// ============================================================================
//  triangle.h  --  三角形 + Möller–Trumbore + AoS / SoA 两套布局与访问器
// ----------------------------------------------------------------------------
//  要证明「换个内存布局能变快」，所以这里同时保留两套布局：
//
//    AoS (TriangleAoS)  : 一个三角形 40 字节连续存放
//                         [v0.x v0.y v0.z v1.x ... v2.z materialId]
//                         取一个三角形 = 一次 40 字节的连续读；
//                         但 warp 内 32 个线程各取各的三角形时，这些地址互相
//                         隔了 40 字节，一次 128B 事务里能命中的线程很少。
//
//    SoA (TriangleSoA)  : 9 个 float 数组 + 1 个材质数组
//                         v0x[] v0y[] v0z[] v1x[] ... v2z[] materialId[]
//                         同一个数组内的地址是连续的，warp 里访问同一批下标
//                         时每次事务都能装下更多线程的数据（访存合并更好）。
//
//  两套布局的取数入口都是 triAt<UseSoA>()，返回同一个 TriangleAoS POD，
//  所以 Möller–Trumbore 只有一份实现，不存在「两种算法结果不一致」的问题。
//
//  注意：这里不 include <cuda_runtime.h>，纯几何 + 纯布局；设备内存的申请与
//  上传放在 bvh.h 的 GpuTriangles 里，保持「算法」与「搬运」分开。
// ============================================================================
#pragma once

#include <vector>

#include "vec3.h"
#include "rt_math.h"
#include "sphere.h"   // HitRecord（定义在 sphere.h，这里复用同一份）

// ---------------------------------------------------------------------------
//  AoS 布局：一个三角形一条记录
// ---------------------------------------------------------------------------
struct TriangleAoS
{
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
    int  materialId;   // 索引到场景的材质数组
};

// ---------------------------------------------------------------------------
//  SoA 布局：指向设备内存里的 10 条并排数组
// ---------------------------------------------------------------------------
struct TriangleSoA
{
    float* v0x; float* v0y; float* v0z;
    float* v1x; float* v1y; float* v1z;
    float* v2x; float* v2y; float* v2z;
    int*   materialId;
    int    count;

    TriangleSoA() : v0x(0), v0y(0), v0z(0), v1x(0), v1y(0), v1z(0),
                    v2x(0), v2y(0), v2z(0), materialId(0), count(0) {}
};

// ---------------------------------------------------------------------------
//  数据视图：暴力遍历和 BVH 叶子都用同一个入口取三角形
// ---------------------------------------------------------------------------
struct TriStore
{
    const TriangleAoS* aos;    // AoS 数组（设备指针）
    TriangleSoA        soa;    // SoA 数组（设备指针，UseSoA = true 时用）
    int                count;

    TriStore() : aos(0), soa(), count(0) {}
};

// AoS：直接返回那条记录（编译器通常合并成 3 次 128-bit 读）。
RT_HD RT_INLINE TriangleAoS triAtAoS(const TriStore& s, int i)
{
    return s.aos[i];
}

// SoA：把 9 个 float 和材质号收集回来。因为 9 个数组互相独立，
// warp 内每个数组的访问都是连续的，这就是「访存合并」收益的来源。
RT_HD RT_INLINE TriangleAoS triAtSoA(const TriangleSoA& s, int i)
{
    TriangleAoS t;
    t.v0 = Vec3(s.v0x[i], s.v0y[i], s.v0z[i]);
    t.v1 = Vec3(s.v1x[i], s.v1y[i], s.v1z[i]);
    t.v2 = Vec3(s.v2x[i], s.v2y[i], s.v2z[i]);
    t.materialId = s.materialId[i];
    return t;
}

// 编译期选择布局：UseSoA = false 走 AoS，true 走 SoA。
// 两个分支都必须能编译（模板实例化时会检查），所以这里用 if 而不是 #if。
template <bool UseSoA>
RT_HD RT_INLINE TriangleAoS triAt(const TriStore& s, int i)
{
    if (UseSoA) return triAtSoA(s.soa, i);
    return triAtAoS(s, i);
}

// ---------------------------------------------------------------------------
//  Möller–Trumbore（光线-三角形求交的标准算法）
// ---------------------------------------------------------------------------
//  e1 = v1-v0; e2 = v2-v0; p = cross(d, e2); det = dot(e1, p);
//  if (|det| < eps) return false;              // 光线与三角形平面平行
//  inv = 1/det; s = o - v0; u = dot(s, p)*inv; if (u<0||u>1) return false;
//  q = cross(s, e1); v = dot(d, q)*inv; if (v<0||u+v>1) return false;
//  t = dot(e2, q)*inv;                          // 命中
//
//  det 就是「行列式」，即三个边向量张成的平行六面体体积；det 接近 0 说明
//  光线方向落在三角形平面内（或者三角形退化成一条线），此时 u/v 会溢出，
//  必须先剔除。u、v 是重心坐标，u+v <= 1 自然把交点限制在三角形内部。
RT_HD RT_INLINE bool mollerTrumbore(const TriangleAoS& tri, const Ray& r,
                                    float tMin, float tMax, float* tHit)
{
    const Vec3 e1 = tri.v1 - tri.v0;
    const Vec3 e2 = tri.v2 - tri.v0;

    const Vec3  p = cross(r.direction, e2);
    const float det = dot(e1, p);
    if (fabsf(det) < 1.0e-12f) return false;               // 平行或退化

    const float inv = 1.0f / det;
    const Vec3  s = r.origin - tri.v0;

    const float u = dot(s, p) * inv;
    if (u < 0.0f || u > 1.0f) return false;

    const Vec3  q = cross(s, e1);
    const float v = dot(r.direction, q) * inv;
    if (v < 0.0f || u + v > 1.0f) return false;

    const float t = dot(e2, q) * inv;
    if (t <= tMin || t >= tMax) return false;

    *tHit = t;
    return true;
}

// 完整求交：填 HitRecord（含双面法线）。
// 三角形没有 vn，所以法线用叉积算，是每个面片的平面法线；
// 再按光线方向翻转，否则从背面看过去三角形是全黑的。
// 绕向因此不影响成像：OBJ 里面的朝向不一致也不会出现黑面。
RT_HD RT_INLINE bool hitTriangle(const TriangleAoS& tri, const Ray& r,
                                 float tMin, float tMax, HitRecord* rec)
{
    float t;
    if (!mollerTrumbore(tri, r, tMin, tMax, &t)) return false;

    Vec3 n = normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
    if (dot(n, r.direction) > 0.0f) n = -n;                // 双面法线

    rec->t = t;
    rec->point = r.pointAt(t);
    rec->normal = n;
    rec->materialId = tri.materialId;
    return true;
}

// 只判断「有没有命中」的版本：阴影光线只需要这个，
// 省掉法线归一化（sqrt + 除法）和写回 HitRecord 的开销。
RT_HD RT_INLINE bool hitTriangleAny(const TriangleAoS& tri, const Ray& r, float tMin, float tMax)
{
    float t;
    return mollerTrumbore(tri, r, tMin, tMax, &t);
}

// ---------------------------------------------------------------------------
//  主机侧：AoS -> SoA 打包
// ---------------------------------------------------------------------------
// 打包成「9 条 float 数组 + 1 条 int 数组」的两块连续内存，这样上传到设备
// 只需要两次 cudaMemcpy，指针也整齐。
inline void packTriangleSoA(const std::vector<TriangleAoS>& tris,
                            std::vector<float>* soaFloats,
                            std::vector<int>* soaMaterialIds)
{
    const size_t n = tris.size();
    soaFloats->assign(n * 9u, 0.0f);
    soaMaterialIds->assign(n, 0);

    for (size_t i = 0; i < n; ++i)
    {
        const TriangleAoS& t = tris[i];
        (*soaFloats)[i]          = t.v0.x;   // v0x[]
        (*soaFloats)[n + i]      = t.v0.y;   // v0y[]
        (*soaFloats)[2u * n + i] = t.v0.z;   // v0z[]
        (*soaFloats)[3u * n + i] = t.v1.x;
        (*soaFloats)[4u * n + i] = t.v1.y;
        (*soaFloats)[5u * n + i] = t.v1.z;
        (*soaFloats)[6u * n + i] = t.v2.x;
        (*soaFloats)[7u * n + i] = t.v2.y;
        (*soaFloats)[8u * n + i] = t.v2.z;
        (*soaMaterialIds)[i] = t.materialId;
    }
}

// 把一块连续内存切成 9+1 个数组指针，填充 TriangleSoA。
inline TriangleSoA bindTriangleSoA(float* base, int* materialIds, int count)
{
    TriangleSoA s;
    s.count = count;
    s.v0x = base;               s.v0y = base + count;       s.v0z = base + 2 * count;
    s.v1x = base + 3 * count;   s.v1y = base + 4 * count;   s.v1z = base + 5 * count;
    s.v2x = base + 6 * count;   s.v2y = base + 7 * count;   s.v2z = base + 8 * count;
    s.materialId = materialIds;
    return s;
}
