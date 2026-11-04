// ============================================================================
//  rt_math.h  --  ray, camera and the handful of constants the tracer needs
// ----------------------------------------------------------------------------
//  Week 1 introduces this file: the camera turns a pixel index into a primary
//  ray.  Everything above `distance()` / `randomInUnit*` is used from week 1
//  onwards; the random helpers are only needed from week 6 (multi-sample
//  anti-aliasing), so they can be ignored until then.
// ============================================================================
#pragma once

#include "vec3.h"

// ---------------------------------------------------------------------------
//  constants
// ---------------------------------------------------------------------------
// 向上取整除法（host / device 通用）。
// 用途：把「N 个像素按 block 宽度要分成多少个 block」这类网格计算写清楚：
//     (n + b - 1) / b   等价于   cdiv(n, b)
// 原来每处都手写 (w + block.x - 1) / block.x，既啰嗦又容易写错括号。
// 只对**非负 n、正 b** 成立 —— 本项目的像素数与网格尺寸都满足这个前提。
RT_HD RT_INLINE int cdiv(int n, int b)
{
    return (b > 0) ? ((n + b - 1) / b) : 0;
}
#define RT_PI        3.14159265358979323846f
#define RT_INFINITY  (1e30f)     // "far enough" for us; avoids INF arithmetic
#define RT_EPS       1.0e-4f     // general ray/plane epsilon
#define RT_SHADOW_EPS 1.0e-3f    // push the shadow/reflection ray off the surface
                                 // (week 2: too small -> shadow acne, too big -> light leak)

// ---------------------------------------------------------------------------
//  ray
// ---------------------------------------------------------------------------
struct Ray
{
    Vec3 origin;
    Vec3 direction;

    RT_HD Ray() : origin(), direction(0.0f, 0.0f, -1.0f) {}
    // 构造时**立即归一化**方向。这样 direction 永远是单位向量，带来两个保证：
//   1. 交点参数 t 直接等于沿光线的真实距离（深度图、阴影距离都依赖这一点）；
//   2. reflect / refract / Schlick 里的点积都是真正的余弦，不因长度而失真。
// 代价是每条光线一次 sqrt —— 但每像素只构造「主光线 + 若干反射/折射」约 10 条，
// 相比 BVH 遍历的几千次访存可以忽略。
RT_HD Ray(const Vec3& o, const Vec3& d) : origin(o), direction(normalize(d)) {}

    // NOTE: deliberately not called `at()` -- a member named `at` collides with
    // the parameter name in MSVC's lookup for `ray.at(t)`.
    RT_HD Vec3 pointAt(float t) const { return origin + direction * t; }
};

// ---------------------------------------------------------------------------
//  camera  (pinhole)
// ---------------------------------------------------------------------------
struct Camera
{
    Vec3 origin;
    Vec3 lowerLeftCorner;
    Vec3 horizontal;
    Vec3 vertical;

    RT_HD Camera() : origin(), lowerLeftCorner(-2.0f, -1.0f, -1.0f),
                     horizontal(4.0f, 0.0f, 0.0f), vertical(0.0f, 2.0f, 0.0f) {}

    // lookfrom/lookat: world-space points, vup: world up (usually (0,1,0)),
    // vfovDeg: vertical field of view in degrees, aspect: width / height.
    // The camera looks down -w, exactly like "Ray Tracing in One Weekend".
    RT_HD Camera(const Vec3& lookfrom, const Vec3& lookat, const Vec3& vup,
                 float vfovDeg, float aspect)
    {
        const float theta = vfovDeg * (RT_PI / 180.0f);
        const float halfHeight = tanf(theta * 0.5f);
        const float halfWidth = aspect * halfHeight;

        const Vec3 w = normalize(lookfrom - lookat);   // points *backwards*
        const Vec3 u = normalize(cross(vup, w));
        const Vec3 v = cross(w, u);

        origin = lookfrom;
        horizontal = u * (2.0f * halfWidth);
        vertical = v * (2.0f * halfHeight);
        lowerLeftCorner = origin - u * halfWidth - v * halfHeight - w;
    }

    // s, t in [0,1]: (0,0) is the bottom-left corner of the image, (1,1) top-right.
    RT_HD Ray getRay(float s, float t) const
    {
        return Ray(origin, normalize(lowerLeftCorner + horizontal * s + vertical * t - origin));
    }
};

// ---------------------------------------------------------------------------
//  pixel -> [0,1] screen coordinates
// ---------------------------------------------------------------------------
// Adds the +0.5 pixel-center offset, which is what makes the image symmetric
// instead of shifted half a pixel to the upper left.
RT_HD RT_INLINE void pixelToUV(int x, int y, int width, int height, float* u, float* v)
{
    *u = ((float)x + 0.5f) / (float)width;
    *v = 1.0f - ((float)y + 0.5f) / (float)height;   // flip: row 0 is the top row
}

// ---------------------------------------------------------------------------
//  world point -> pixel  (added in week 4)
// ---------------------------------------------------------------------------
// The shadow self-test needs to know which pixel a known world-space point
// lands on, so it can read that pixel back and check it is dark.
// This is the inverse of getRay + pixelToUV for points in front of the camera.
// Returns false when the point is behind the camera plane or outside the image.
//
// 【这个函数重写过一次，坑值得记下来】
// 旧实现把投影当成了**仿射变换**：解 oc - lowerLeftCorner = s*H + t*V（2x2 正规方程）。
// 但 getRay 的射线是 origin + λ*(ll + s*H + t*V - origin)，只有 λ=1（点恰好落在
// 近平面）时才有 p - ll = s*H + t*V。对一般远近的点解出的 s/t 是错的：点一旦偏远
// 或偏离视轴，t 就跑到 [0,1] 之外，函数返回 false，调用方拿到未初始化的
// 0xCCCCCCCC（Debug 填充值）。早期版本就是这样炸的：网格中心 (0,0.669,0) 明明在
// 画面正中，旧算法算出 t=-0.37。
//
// 正确做法是**透视投影**：先把点转到相机坐标（右 u / 上 v / 前 -w），再做透视除法
// xc/zc、yc/zc，最后映射到 [0,1]。Camera 里没存 fov，但 horizontal/vertical 是
// 视口向量，长度正好是 2*halfWidth / 2*halfHeight，可以直接反推。
inline bool worldToPixel(const Camera& cam, const Vec3& p, int width, int height,
                         int* px, int* py)
{
    const float hLen = length(cam.horizontal);      // = 2 * halfWidth
    const float vLen = length(cam.vertical);        // = 2 * halfHeight
    if (hLen < 1e-12f || vLen < 1e-12f) return false;

    const Vec3 u = cam.horizontal * (1.0f / hLen);  // 右（单位向量）
    const Vec3 v = cam.vertical * (1.0f / vLen);    // 上（单位向量）
    const Vec3 w = cross(u, v);                     // 从目标指向相机；相机看向 -w

    const Vec3  oc = p - cam.origin;
    const float zc = -dot(oc, w);                   // 沿视线方向的深度
    if (zc <= 1.0e-6f) return false;                // 在相机平面上或背后，投不出来

    // 透视除法：屏幕偏移 = (相机坐标分量 / 深度) / 视口尺寸 + 0.5
    const float s = 0.5f + dot(oc, u) / zc / hLen;
    const float t = 0.5f + dot(oc, v) / zc / vLen;

    const int ix = (int)(s * (float)width);
    const int iy = (int)((1.0f - t) * (float)height);   // 屏幕上 y 朝下
    if (ix < 0 || ix >= width || iy < 0 || iy >= height) return false;

    *px = ix;
    *py = iy;
    return true;
}

// ---------------------------------------------------------------------------
//  reflection / refraction  (week 3)
// ---------------------------------------------------------------------------
// n must be the *unit outward* normal at the hit point.
RT_HD RT_INLINE Vec3 reflect(const Vec3& d, const Vec3& n)
{
    return d - n * (2.0f * dot(d, n));
}

// Snell's law. Returns false on total internal reflection (no refracted ray).
// n is the outward normal; etaRatio = etaIncident / etaTransmitted.
RT_HD RT_INLINE bool refract(const Vec3& d, const Vec3& n, float etaRatio, Vec3* refracted)
{
    const float cosThetaI = -dot(d, n);                       // d points into the surface
    const float k = 1.0f - etaRatio * etaRatio * (1.0f - cosThetaI * cosThetaI);
    if (k < 0.0f) return false;                               // total internal reflection
    *refracted = normalize(d * etaRatio + n * (etaRatio * cosThetaI - sqrtf(k)));
    return true;
}

// Schlick's approximation of the Fresnel reflectance (week 3).
RT_HD RT_INLINE float schlickReflectance(float cosTheta, float ior)
{
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    const float m = 1.0f - cosTheta;
    return r0 + (1.0f - r0) * (m * m * m * m * m);
}

// ---------------------------------------------------------------------------
//  tiny PRNG for stochastic anti-aliasing / soft shadows (week 6, optional)
// ---------------------------------------------------------------------------
// Hash based, so the result only depends on the pixel -- a re-render at the
// same settings is bit-for-bit reproducible, which matters when comparing
// optimisations.
RT_HD RT_INLINE unsigned int hashUint(unsigned int x)
{
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

RT_HD RT_INLINE float randomFloat(unsigned int* state)
{
    *state = hashUint(*state);
    return (float)(*state & 0x00FFFFFFu) / (float)0x01000000u;   // [0,1)
}
