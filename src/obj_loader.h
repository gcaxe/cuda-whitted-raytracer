// ============================================================================
//  obj_loader.h  --  OBJ 解析 + 程序化兜底网格 + scale/translate/rotateY 变换
// ----------------------------------------------------------------------------
//  经验：渲染器最容易卡住的地方不是求交，而是「没有模型文件」。
//  所以这里提供两条路，程序启动时任选其一，永远能出图：
//
//    A) loadObj()              -- 只解析 v / f（支持 a、a/b、a//b、a/b/c、
//                                 负下标、多边形扇形三角化），忽略 vt/vn/mtllib。
//                                 Bunny 这类模型只有 v 和 f，法线用叉积现算。
//    B) generateProceduralMesh()-- 网络不可用时的兜底：用参数方程拼一只「类 Bunny」
//                                 的椭球 + 两只耳朵 + 尾巴，约 6.8 万三角形。
//
//  真模型直接随仓库提供：默认 --obj 就是 ..\..\objs\bunny_10k.obj（斯坦福兔子，
//  5,002 顶点 / 10,000 面，只有 v 和 f，正好是本文件支持的子集），不联网也能跑。
//  加载失败会自动回退到程序化网格，所以路径写错也不会没图看。
//  （同一目录下的 teapot.obj 暂不使用。）
//
//  变换只有三个参数（scale / translate / rotateY），顺序是固定的：
//
//      p' = RotY(scale * p + translate)
//
//  先缩放、再平移（把模型摆正、底面落到 y=0、xz 居中），最后绕世界 Y 轴旋转。
//  顺序写死是有意的：模型已经居中在原点，所以绕 Y 轴旋转不会把它转出画面。
//  这也正是 --frames 转台能绕着一个固定中心转的前提。
// ============================================================================
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

#include "vec3.h"
#include "rt_math.h"
#include "triangle.h"
#include "bvh.h"      // AABB（网格包围盒 / BVH 建树都要用）

// ---------------------------------------------------------------------------
//  TriMesh：一台主机上的三角形汤 + 它的包围盒
// ---------------------------------------------------------------------------
struct TriMesh
{
    std::vector<TriangleAoS> triangles;
    AABB                     bounds;
    int                      materialId;

    TriMesh() : triangles(), bounds(), materialId(0) {}

    int triangleCount() const { return (int)triangles.size(); }

    void recomputeBounds()
    {
        AABB b;
        for (size_t i = 0; i < triangles.size(); ++i)
        {
            b.grow(triangles[i].v0);
            b.grow(triangles[i].v1);
            b.grow(triangles[i].v2);
        }
        bounds = b;
    }

    // p' = RotY(scale * p + translate)
    void applyTransform(float scale, const Vec3& translate, float rotateYDeg)
    {
        const float rad = radians(rotateYDeg);
        const float cs = cosf(rad);
        const float sn = sinf(rad);

        for (size_t i = 0; i < triangles.size(); ++i)
        {
            Vec3* v[3] = { &triangles[i].v0, &triangles[i].v1, &triangles[i].v2 };
            for (int k = 0; k < 3; ++k)
            {
                const Vec3 p = (*v[k]) * scale + translate;
                *v[k] = Vec3(cs * p.x + sn * p.z, p.y, -sn * p.x + cs * p.z);
            }
        }
        recomputeBounds();
    }

    // 自动摆位：缩放到指定高度、xz 居中、底面贴 y=0，再绕 Y 轴转 rotateYDeg。
    // OBJ 模型的原始尺度和位置完全不可控（Bunny 只有 0.15 单位高还偏在一边），
    // 所以「先量包围盒再摆位」比手工调常数可靠得多。
    void fitToHeight(float targetHeight, float rotateYDeg)
    {
        recomputeBounds();
        const float h = bounds.bmax.y - bounds.bmin.y;
        if (h <= 1.0e-9f) return;

        const float s = targetHeight / h;
        const Vec3  c = bounds.centroid();
        const Vec3  t(-c.x * s, -bounds.bmin.y * s, -c.z * s);
        applyTransform(s, t, rotateYDeg);
    }
};

// ---------------------------------------------------------------------------
//  OBJ 下标解析
// ---------------------------------------------------------------------------
// token 形如 "12"、"12/34"、"12//56"、"12/34/56"，只要第一个整数。
// OBJ 的下标是 1-based，负数表示「从末尾数」：-1 是最后一个顶点。
inline bool parseObjIndex(const char* token, int vertexCount, int* out)
{
    if (token == NULL || token[0] == '\0') return false;

    char* end = NULL;
    const long raw = strtol(token, &end, 10);
    if (end == token) return false;                 // 这个 token 里没有整数

    const int index = (raw < 0) ? (vertexCount + (int)raw) : ((int)raw - 1);
    if (index < 0 || index >= vertexCount) return false;

    *out = index;
    return true;
}

// ---------------------------------------------------------------------------
//  A) 读 OBJ（只认 v / f）
// ---------------------------------------------------------------------------
// 返回值：true = 读到了至少一个三角形。失败时 errOut 里是中文原因。
inline bool loadObj(const char* path, TriMesh* mesh, int materialId, char* errOut, size_t errSize)
{
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    mesh->triangles.clear();
    mesh->materialId = materialId;

    std::ifstream in(path);
    if (!in.is_open())
    {
        snprintf(errOut, errSize, "打不开文件（回退到程序化网格）");
        return false;
    }

    std::vector<Vec3> positions;
    positions.reserve(1u << 16);

    char line[1024];
    int  lineNo = 0;
    long skippedFaces = 0;

    while (in.getline(line, sizeof(line)))
    {
        ++lineNo;
        if (line[0] == '\0' || line[0] == '#') continue;

        // ---- 顶点 ----
        if (line[0] == 'v' && (line[1] == ' ' || line[1] == '\t'))
        {
            float x = 0.0f, y = 0.0f, z = 0.0f;
            if (sscanf_s(line + 1, "%f %f %f", &x, &y, &z) == 3)
                positions.push_back(Vec3(x, y, z));
            continue;
        }

        // ---- 面 ----
        if (line[0] == 'f' && (line[1] == ' ' || line[1] == '\t'))
        {
            int  index[16];
            int  count = 0;
            bool bad = false;

            char* ctx = NULL;
            for (char* tok = strtok_s(line + 1, " \t\r\n", &ctx);
                 tok != NULL;
                 tok = strtok_s(NULL, " \t\r\n", &ctx))
            {
                if (count >= (int)(sizeof(index) / sizeof(index[0]))) { bad = true; break; }
                int v = 0;
                if (!parseObjIndex(tok, (int)positions.size(), &v)) { bad = true; break; }
                index[count++] = v;
            }

            if (bad || count < 3)
            {
                ++skippedFaces;                     // 面里混了 vt/vn 之外的脏数据，整面丢掉
                continue;
            }

            // 多边形扇形三角化：(0,1,2) (0,2,3) (0,3,4) ...
            for (int k = 1; k + 1 < count; ++k)
            {
                TriangleAoS tri;
                tri.v0 = positions[index[0]];
                tri.v1 = positions[index[k]];
                tri.v2 = positions[index[k + 1]];
                tri.materialId = materialId;
                mesh->triangles.push_back(tri);
            }
            continue;
        }

        // 其余关键字（vt / vn / o / g / s / usemtl / mtllib ...）一律忽略：
        // 着色只用面法线，材质由程序指定，读进来也用不上。
    }

    mesh->recomputeBounds();

    const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("  OBJ : %s\n", path);
    printf("        顶点=%d  三角形=%d  跳过面=%d  行数=%d  解析=%.1f ms\n",
           (int)positions.size(), mesh->triangleCount(), (int)skippedFaces, lineNo, ms);
    printf("        包围盒 min=(%.3f, %.3f, %.3f) max=(%.3f, %.3f, %.3f)\n",
           mesh->bounds.bmin.x, mesh->bounds.bmin.y, mesh->bounds.bmin.z,
           mesh->bounds.bmax.x, mesh->bounds.bmax.y, mesh->bounds.bmax.z);

    if (mesh->triangleCount() == 0)
    {
        snprintf(errOut, errSize, "文件里没有可用的 f 面（回退到程序化网格）");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
//  A2) assimp 加载器（可选路径，实现在 obj_assimp.cpp）
// ---------------------------------------------------------------------------
// 这里**只声明**，不 include <assimp/...>：
//   assimp 的头会拖进几万行模板。一旦被 kernel.cu 间接 include，nvcc 的 device
//   编译阶段也要解析它们 —— 既明显拖慢编译，也容易和 CUDA 的类型/宏冲突。
//   实现放在 obj_assimp.cpp（普通 C++ 编译单元），nvcc 完全看不到。
//
// 与手写 loadObj 的签名保持一致（同样的 5 个参数），调用处可以直接替换。
// 成功返回 true 并填充 triangles + bounds + materialId；失败返回 false 并写 errOut。
bool loadObjWithAssimp(const char* path, TriMesh* mesh, int materialId,
                       char* errOut, size_t errSize);

// assimp 版本字符串（如 "5.3.1"）；未链接 assimp 时由实现方返回空串。
const char* assimpVersionString();

// ---------------------------------------------------------------------------
//  B) 程序化兜底网格：椭球 + 两只耳朵 + 尾巴
// ---------------------------------------------------------------------------
// 「类 Bunny」不需要真的像兔子，只要满足三件事：
//   1. 形状可辨认（有耳朵、有尾巴，一眼不是个球）；
//   2. 三角形数量落在 5~8 万，能体现 BVH 的加速效果；
//   3. 表面有轻微起伏，光照下有明暗变化，不是一坨死白的球。
// 参数方程 -> 网格的规则很简单：极点是三角形扇，其余每圈四边形拆两个三角形，
// 这样不会产生零面积的退化三角形（退化面会让法线变成 NaN）。
inline void appendEllipsoid(std::vector<TriangleAoS>* tris,
                            const Vec3& center, const Vec3& radii,
                            int nu, int nv, float bumpAmp, int materialId)
{
    if (nu < 3) nu = 3;
    if (nv < 3) nv = 3;

    // 顶点网格：(nv+1) 圈 × nu 个经度
    std::vector<Vec3> grid((size_t)(nv + 1) * (size_t)nu);
    for (int j = 0; j <= nv; ++j)
    {
        const float theta = ((float)j / (float)nv) * RT_PI;      // 0 = 北极, pi = 南极
        const float st = sinf(theta);
        const float ct = cosf(theta);
        for (int i = 0; i < nu; ++i)
        {
            const float phi = ((float)i / (float)nu) * 2.0f * RT_PI;

            // 低频径向扰动：让轮廓不是完美椭圆（一眼能看出是程序生成的都无所谓，
            // 关键是有起伏之后高光和明暗交界线才有东西可看）。
            float r = 1.0f;
            if (bumpAmp > 0.0f)
                r += bumpAmp * (0.6f * sinf(5.0f * phi) * sinf(3.0f * theta) + 0.4f * sinf(7.0f * theta));

            Vec3 p(st * cosf(phi) * radii.x * r,
                   ct * radii.y * r,
                   st * sinf(phi) * radii.z * r);
            grid[(size_t)j * (size_t)nu + (size_t)i] = p + center;
        }
    }

    TriangleAoS tri;
    tri.materialId = materialId;

    for (int j = 0; j < nv; ++j)
    {
        for (int i = 0; i < nu; ++i)
        {
            const int i2 = (i + 1) % nu;
            const Vec3& a = grid[(size_t)j * (size_t)nu + (size_t)i];
            const Vec3& b = grid[(size_t)j * (size_t)nu + (size_t)i2];
            const Vec3& c = grid[(size_t)(j + 1) * (size_t)nu + (size_t)i2];
            const Vec3& d = grid[(size_t)(j + 1) * (size_t)nu + (size_t)i];

            if (j == 0)
            {
                // 北极：a、b 是同一个点 -> 只出一个三角形，跳过退化面
                tri.v0 = a; tri.v1 = c; tri.v2 = d;
                tris->push_back(tri);
            }
            else if (j == nv - 1)
            {
                // 南极：c、d 是同一个点
                tri.v0 = a; tri.v1 = b; tri.v2 = c;
                tris->push_back(tri);
            }
            else
            {
                tri.v0 = a; tri.v1 = b; tri.v2 = c;
                tris->push_back(tri);
                tri.v0 = a; tri.v1 = c; tri.v2 = d;
                tris->push_back(tri);
            }
        }
    }
}

// 生成整只「类 Bunny」。
// 三角形预算分配：身体 92%，两只耳朵和尾巴占剩下的 8%。
inline void generateProceduralMesh(TriMesh* mesh, int targetTriangles, int materialId)
{
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    mesh->triangles.clear();
    mesh->materialId = materialId;
    mesh->triangles.reserve((size_t)targetTriangles + 4096u);

    // 身体的三角形数 = 2*nu*(nv-1)；取 nu = nv = n，于是 2n^2 ≈ 0.92*target。
    const float bodyTarget = (float)targetTriangles * 0.92f;
    int n = (int)(sqrtf(bodyTarget * 0.5f)) + 1;
    if (n < 12)  n = 12;
    if (n > 400) n = 400;

    const int bodyBefore = mesh->triangleCount();
    appendEllipsoid(&mesh->triangles, Vec3(0.0f, 0.0f, 0.0f), Vec3(1.00f, 0.86f, 1.12f),
                    n, n, 0.05f, materialId);
    const int bodyCount = mesh->triangleCount() - bodyBefore;

    // 两只耳朵：细长椭球，底部埋进身体里，尖朝上
    const int earBefore = mesh->triangleCount();
    appendEllipsoid(&mesh->triangles, Vec3(-0.34f, 0.74f, -0.10f), Vec3(0.13f, 0.46f, 0.13f),
                    28, 20, 0.02f, materialId);
    appendEllipsoid(&mesh->triangles, Vec3(0.34f, 0.74f, -0.10f), Vec3(0.13f, 0.46f, 0.13f),
                    28, 20, 0.02f, materialId);
    const int earCount = mesh->triangleCount() - earBefore;

    // 尾巴：身后一个小球
    const int tailBefore = mesh->triangleCount();
    appendEllipsoid(&mesh->triangles, Vec3(0.0f, -0.16f, 1.06f), Vec3(0.20f, 0.20f, 0.22f),
                    20, 14, 0.0f, materialId);
    const int tailCount = mesh->triangleCount() - tailBefore;

    mesh->recomputeBounds();

    const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("  程序化兜底网格: 身体 %d + 耳朵 %d + 尾巴 %d = %d 个三角形  生成=%.1f ms\n",
           bodyCount, earCount, tailCount, mesh->triangleCount(), ms);
    printf("        包围盒 min=(%.3f, %.3f, %.3f) max=(%.3f, %.3f, %.3f)\n",
           mesh->bounds.bmin.x, mesh->bounds.bmin.y, mesh->bounds.bmin.z,
           mesh->bounds.bmax.x, mesh->bounds.bmax.y, mesh->bounds.bmax.z);
}
