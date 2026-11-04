// ============================================================================
//  materials.h  --  材质系统 + Blinn-Phong 局部着色 + 阴影光线
// ----------------------------------------------------------------------------
//  为什么要有 MaterialType 而不是几个浮点参数：
//    Diffuse / Mirror / Glass 三者的差别不只是数值，而是「局部着色之后还要
//    不要再发光线、发几条、怎么混合」。用 enum 分派一次，渲染器的主循环就能
//    写成三个分支，读代码的人一眼能看出 Whitted 模型的结构；
//    反过来，用 if (reflectivity > 0) 之类的散装判断，加一种材质就要改五处。
//
//  为什么材质按值传进 kernel：
//    场景里只有几种材质，把它们放进一个定长数组随 kernel 参数传递，会落在
//    常量内存里；warp 内所有线程读同一个材质时是一次广播，比解引用设备指针快。
//
//  设备代码注意：这里所有函数都用 fminf/fmaxf 而不是 vec3.h 的 clampf()，
//  因为 clampf()/radians() 是主机端 inline 函数（不能被 __global__ 调用）。
// ============================================================================
#pragma once

#include "vec3.h"
#include "rt_math.h"

// ---------------------------------------------------------------------------
//  材质类型
// ---------------------------------------------------------------------------
enum MaterialType
{
    MAT_DIFFUSE = 0,   // 郎伯 + 高光，不发二次光线
    MAT_MIRROR  = 1,   // 局部项与反射光线按 reflectivity 混合
    MAT_GLASS   = 2    // Snell 折射 + Schlick Fresnel，在全反射时退化成镜面
};

// ---------------------------------------------------------------------------
//  Material
// ---------------------------------------------------------------------------
// 字段顺序按 16 字节一组排（Vec3 + int / 4 个 float / 4 个 float），
// 一共 48 字节：warp 里所有线程读同一个材质时，一辆 128B 缓存行能装下
// 两个多材质，常量内存的广播读取也更整齐。
struct Material
{
    Vec3  albedo;        // 基色（镜面/玻璃也拿它做染色）
    int   type;          // MaterialType
    float ka;            // 环境光系数（本渲染器没有 GI，用天光近似代替）
    float kd;            // 漫反射系数
    float ks;            // 高光系数
    float shininess;     // Blinn-Phong 指数
    float reflectivity;  // 反射分支的混合权重（0 = 不反射，1 = 纯镜面）
    float ior;           // 折射率（玻璃 1.50 / 水 1.33 / 空气 1.00）
    float checkerScale;  // > 0 时对世界坐标做程序化棋盘（不是纹理贴图）
    float pad;           // 补齐到 48 字节
};

// ---- 构造函数（只在主机端搭场景时用）-------------------------------------
inline Material makeDiffuse(const Vec3& albedo, float ka = 0.16f, float kd = 0.85f,
                            float ks = 0.25f, float shininess = 48.0f)
{
    Material m;
    m.albedo = albedo;
    m.type = MAT_DIFFUSE;
    m.ka = ka; m.kd = kd; m.ks = ks; m.shininess = shininess;
    m.reflectivity = 0.0f;
    m.ior = 1.0f;
    m.checkerScale = 0.0f;
    m.pad = 0.0f;
    return m;
}

inline Material makeMirror(const Vec3& tint, float reflectivity = 0.85f,
                           float ks = 0.35f, float shininess = 200.0f)
{
    Material m = makeDiffuse(tint, 0.10f, 0.20f, ks, shininess);
    m.type = MAT_MIRROR;
    m.reflectivity = reflectivity;
    return m;
}

inline Material makeGlass(const Vec3& tint, float ior = 1.50f)
{
    Material m = makeDiffuse(tint, 0.02f, 0.02f, 0.20f, 200.0f);
    m.type = MAT_GLASS;
    m.reflectivity = 1.0f;     // 玻璃两分支都发，权重由 Fresnel 决定
    m.ior = ior;
    return m;
}

// 定长材质表，随 kernel 参数按值传递。
#define RT_MAX_MATERIALS 8

struct MaterialSet
{
    Material items[RT_MAX_MATERIALS];
    int      count;

    MaterialSet() : count(0) {}

    int add(const Material& m)
    {
        if (count >= RT_MAX_MATERIALS) return -1;
        items[count] = m;
        return count++;
    }
};

// ---------------------------------------------------------------------------
//  程序化棋盘（地面用）
// ---------------------------------------------------------------------------
// 故意不做纹理采样：已知限制是「没有 UV / 没有贴图」，
// 这里只按世界坐标 xz 切格子，换来一个能看清阴影和反射的地面。
RT_HD RT_INLINE Vec3 applyChecker(const Material& m, const Vec3& p, const Vec3& albedo)
{
    if (m.checkerScale <= 0.0f) return albedo;
    const int ix = (int)floorf(p.x * m.checkerScale);
    const int iz = (int)floorf(p.z * m.checkerScale);
    return ((ix + iz) & 1) ? albedo * 0.22f : albedo;
}

// ---------------------------------------------------------------------------
//  环境项（单独一个函数：多光源时才不会把环境项叠加 N 次）
// ---------------------------------------------------------------------------
// 本渲染器没有全局光照，环境项用一个半球天光近似：背光面和阴影里不是死黑，
// 还能看出物体的固有色。albedo 由调用方传（地面会先过一层程序化棋盘）。
RT_HD RT_INLINE Vec3 ambientTerm(const Material& m, const Vec3& albedo, const Vec3& ambientColor)
{
    return albedo * (m.ka * ambientColor);
}

// ---------------------------------------------------------------------------
//  Blinn-Phong：一个光源的漫反射 + 高光
// ---------------------------------------------------------------------------
//   * 返回的是「这个光源贡献的局部项」，环境项由 ambientTerm() 单独加一次；
//   * 被遮挡（inShadow）或背光（n·l <= 0）时返回黑色 —— 硬阴影就是这么来的；
//   * 高光只在 n·l > 0 时算，否则背光面会出现「高光穿帮」；
//   * pow(负数, 非整数) 是 NaN，所以 ndoth 必须先夹紧。
RT_HD RT_INLINE Vec3 shadeBlinnPhong(const Material& m, const Vec3& albedo, const Vec3& p,
                                     const Vec3& n, const Vec3& v, const Vec3& lightPos,
                                     const Vec3& lightColor, bool inShadow)
{
    const Vec3  l = normalize(lightPos - p);
    const float ndotl = dot(n, l);
    if (inShadow || ndotl <= 0.0f) return Vec3(0.0f, 0.0f, 0.0f);

    Vec3 color = albedo * (m.kd * ndotl) * lightColor;

    const Vec3  h = normalize(l + v);
    const float ndoth = fmaxf(0.0f, dot(n, h));
    color += lightColor * (m.ks * powf(ndoth, m.shininess));
    return color;
}

// ---------------------------------------------------------------------------
//  阴影光线（epsilon 的处理写在这里）
// ---------------------------------------------------------------------------
// 为什么做成模板：遮挡查询需要认识「场景」（球 + 网格 + BVH + 共享内存根缓存），
// 而 materials.h 不该知道这些细节。把查询本身抽象成一个可调用对象传进来，
// 材质系统就只负责「从 p 出发、朝光源、区间是多少」这三件事。
// 调用方在 kernel.cu 里提供 Occluder（一个带 operator() 的小结构体）。
template <typename OccluderT>
RT_HD RT_INLINE bool inShadow(const OccluderT& occluded, const Vec3& p, const Vec3& n, const Vec3& lightPos)
{
    const Vec3  toLight = lightPos - p;
    const float dist = length(toLight);
    if (dist <= RT_SHADOW_EPS) return false;

    const Vec3 dir = toLight / dist;

    // 起点沿法线抬起 RT_SHADOW_EPS：
    //   不抬 -> 光线与自己的表面相交，出现条纹状的 shadow acne；
    //   抬太多 -> 接地阴影与物体分离（light leak / peter-panning）。
    // 终点也缩进 RT_SHADOW_EPS，避免把「光源正好在表面另一侧」算成遮挡。
    const Ray shadowRay(p + n * RT_SHADOW_EPS, dir);
    return occluded(shadowRay, RT_SHADOW_EPS, dist - RT_SHADOW_EPS);
}
