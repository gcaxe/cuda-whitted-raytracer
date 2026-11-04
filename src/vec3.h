// ============================================================================
//  vec3.h  --  minimal 3D vector math for the Whitted ray tracer
// ----------------------------------------------------------------------------
//  Design notes:
//    * Everything is inline and marked __host__ __device__ so the very same
//      code runs on the CPU (scene setup / BVH build) and in the CUDA kernel.
//    * float3 from <vector_types.h> is intentionally NOT used as the vector
//      type: CUDA's float3 has no operators, and this struct has the exact
//      same 12-byte layout, so we keep the math readable.
//    * Only <math.h> is included, which nvcc maps to a fast device version
//      of sqrt/floor/... automatically.
// ============================================================================
#pragma once

#include <math.h>

#ifdef __CUDACC__
#include <vector_types.h>   // uchar4 (CUDA vector type); the .cu file is the only compile unit
#define RT_HD __host__ __device__
#define RT_INLINE __forceinline__
#else
#define RT_HD
#define RT_INLINE inline
#endif

struct Vec3
{
    float x, y, z;

    RT_HD Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    RT_HD explicit Vec3(float s) : x(s), y(s), z(s) {}
    RT_HD Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    RT_HD Vec3 operator-() const { return Vec3(-x, -y, -z); }

    RT_HD Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    RT_HD Vec3& operator-=(const Vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    RT_HD Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    RT_HD Vec3& operator/=(float s) { const float inv = 1.0f / s; x *= inv; y *= inv; z *= inv; return *this; }
    RT_HD Vec3& operator*=(const Vec3& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
};

// ---- component-wise arithmetic --------------------------------------------
RT_HD RT_INLINE Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
RT_HD RT_INLINE Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
RT_HD RT_INLINE Vec3 operator*(const Vec3& a, const Vec3& b) { return Vec3(a.x * b.x, a.y * b.y, a.z * b.z); }
RT_HD RT_INLINE Vec3 operator/(const Vec3& a, const Vec3& b) { return Vec3(a.x / b.x, a.y / b.y, a.z / b.z); }

// ---- scalar arithmetic ----------------------------------------------------
RT_HD RT_INLINE Vec3 operator*(float s, const Vec3& v) { return Vec3(s * v.x, s * v.y, s * v.z); }
RT_HD RT_INLINE Vec3 operator*(const Vec3& v, float s) { return Vec3(v.x * s, v.y * s, v.z * s); }
RT_HD RT_INLINE Vec3 operator/(const Vec3& v, float s) { const float inv = 1.0f / s; return Vec3(v.x * inv, v.y * inv, v.z * inv); }

// ---- products and lengths -------------------------------------------------
RT_HD RT_INLINE float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

RT_HD RT_INLINE Vec3 cross(const Vec3& a, const Vec3& b)
{
    return Vec3(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}

RT_HD RT_INLINE float lengthSquared(const Vec3& v) { return dot(v, v); }
RT_HD RT_INLINE float length(const Vec3& v) { return sqrtf(dot(v, v)); }
// 单位化。**零向量有保护**：长度过小时返回 (0,0,-1)，而不是除零产生 NaN。
// 这条保护是 rt_math.h 里 Ray 构造函数归一化的前提 —— 只要有一处传进零向量，
// 没有保护就会让整张图被 NaN 污染；而 NaN 参与的浮点比较全是 false，
// 表现是「某些像素莫名其妙全黑」，极难排查。
RT_HD RT_INLINE Vec3 normalize(const Vec3& v)
{
    const float len = length(v);
    return (len > 1.0e-8f) ? (v / len) : Vec3(0.0f, 0.0f, -1.0f);
}
RT_HD RT_INLINE Vec3 lerp(const Vec3& a, const Vec3& b, float t) { return a * (1.0f - t) + b * t; }
RT_HD RT_INLINE Vec3 minVec(const Vec3& a, const Vec3& b) { return Vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
RT_HD RT_INLINE Vec3 maxVec(const Vec3& a, const Vec3& b) { return Vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }

// ---- small scalar helpers -------------------------------------------------
// NOTE: these are RT_HD, not plain host `inline`.  Device code needs them too
// (clampf in toUchar4() below, which every kernel calls), and nvcc rejects
// calling a __host__-only function from a __global__ one:
//     "calling a __host__ function from a __global__ function is not allowed".
// Keeping one RT_HD implementation means host scene setup and the kernel share
// exactly the same maths.
RT_HD RT_INLINE float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
RT_HD RT_INLINE float radians(float degrees) { return degrees * 0.01745329251994329577f; }

// ---- color helpers --------------------------------------------------------
// Works on linear [0,1] color values; gamma is applied once at the end of the
// pixel shader so intermediate math stays in linear space.
RT_HD RT_INLINE Vec3 gammaCorrect(const Vec3& c, float invGamma)
{
    return Vec3(powf(fmaxf(c.x, 0.0f), invGamma),
                powf(fmaxf(c.y, 0.0f), invGamma),
                powf(fmaxf(c.z, 0.0f), invGamma));
}

// Pack a linear color into an 8-bit BGRA-ish uchar4 (x=r, y=g, z=b, w=a=255).
RT_HD RT_INLINE uchar4 toUchar4(const Vec3& c, float invGamma)
{
    const Vec3 g = gammaCorrect(c, invGamma);
    uchar4 out;
    out.x = (unsigned char)(clampf(g.x, 0.0f, 1.0f) * 255.0f + 0.5f);
    out.y = (unsigned char)(clampf(g.y, 0.0f, 1.0f) * 255.0f + 0.5f);
    out.z = (unsigned char)(clampf(g.z, 0.0f, 1.0f) * 255.0f + 0.5f);
    out.w = 255;
    return out;
}
