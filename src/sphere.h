// ============================================================================
//  sphere.h  --  ray/sphere intersection + the HitRecord all primitives fill
// ----------------------------------------------------------------------------
//  Maths: substitute the ray  p(t) = o + t*d  into |p - c|^2 = r^2
//
//      a = d.d
//      b = 2 * d.(o - c)
//      cc = (o-c).(o-c) - r^2
//      disc = b*b - 4*a*cc
//
//  disc < 0            -> the ray misses the sphere entirely
//  otherwise t = (-b -+ sqrt(disc)) / (2a):  minus = near root, plus = far root
//
//  The three classic mistakes, all avoided below:
//    * using a root with t <= 0 (a hit *behind* the camera makes the image
//      look "inside out" and is very hard to spot);
//    * forgetting to normalise d (then a != 1 and every derived distance is
//      scaled wrong);
//    * taking the far root when the camera is inside the sphere, which makes
//      the near surface disappear.
//
//  Week 6 change: the scene is passed *by value* in a fixed-size array instead
//  of as a `const Sphere*` pointing at a host array.  A host pointer inside a
//  kernel argument is not dereferenceable on the device, and week 6 really
//  does ship spheres + a mesh + a BVH to the GPU in one launch.  A small
//  by-value array lands in constant memory, which is the fastest place for
//  data every thread reads.
// ============================================================================
#pragma once

#include "vec3.h"
#include "rt_math.h"

// ---------------------------------------------------------------------------
//  Hit record: what the shading code needs to know about an intersection
// ---------------------------------------------------------------------------
// `materialId` indexes into the scene's material array.  (Week 1 called this
// field `objectId`; now that spheres and mesh triangles live in one scene,
// "which material shades me" is the only thing the shading code needs.)
struct HitRecord
{
    float t;           // distance along the ray
    Vec3  point;       // world-space hit position
    Vec3  normal;      // unit surface normal
    int   materialId;  // index into the material array

    RT_HD HitRecord() : t(0.0f), point(), normal(0.0f, 1.0f, 0.0f), materialId(0) {}
};

// ---------------------------------------------------------------------------
//  Sphere primitive
// ---------------------------------------------------------------------------
// 地面为什么用一个大球：半径 1000、球心在 y=-1000 的球，在可见范围内就是一个
// 平面。这样阴影、反射、包围盒全部复用同一套球体求交代码，不用额外写平面图元
// （常规做法）。
struct Sphere
{
    Vec3  center;
    float radius;
    int   materialId;

    RT_HD Sphere() : center(), radius(1.0f), materialId(0) {}
    RT_HD Sphere(const Vec3& c, float r, int id = 0) : center(c), radius(r), materialId(id) {}
};

// Fixed upper bound so the whole set can travel by value into the kernel.
#define RT_MAX_SPHERES 8

struct SphereSet
{
    Sphere items[RT_MAX_SPHERES];
    int    count;

    SphereSet() : count(0) {}

    // Host-side helper: silently ignores spheres past the cap (and says so once).
    bool add(const Sphere& s)
    {
        if (count >= RT_MAX_SPHERES) return false;
        items[count++] = s;
        return true;
    }
};

// ---------------------------------------------------------------------------
//  Nearest-hit test against one sphere
// ---------------------------------------------------------------------------
// Returns true and fills `rec` when the sphere is hit with t inside
// [tMin, tMax].  Passing tMax = current closest distance gives free
// "keep the nearest" behaviour when several primitives are tested.
RT_HD RT_INLINE bool hitSphere(const Sphere& s, const Ray& r, float tMin, float tMax, HitRecord* rec)
{
    const Vec3  oc = r.origin - s.center;
    const float a = dot(r.direction, r.direction);          // == 1 if d is unit length
    const float halfB = dot(oc, r.direction);               // b/2, saves a few multiplies
    const float c = dot(oc, oc) - s.radius * s.radius;
    const float disc = halfB * halfB - a * c;

    if (disc < 0.0f) return false;                          // no real root -> miss

    const float sqrtDisc = sqrtf(disc);
    const float invA = 1.0f / a;

    // near root first: t0 <= t1 because sqrtDisc >= 0
    float t = (-halfB - sqrtDisc) * invA;
    if (t <= tMin || t >= tMax)
    {
        t = (-halfB + sqrtDisc) * invA;                     // camera inside / near root behind us
        if (t <= tMin || t >= tMax) return false;
    }

    rec->t = t;
    rec->point = r.pointAt(t);
    rec->normal = (rec->point - s.center) / s.radius;       // already unit length
    rec->materialId = s.materialId;
    return true;
}

// Nearest hit over the whole set (tMax shrinks as closer hits are found).
RT_HD RT_INLINE bool hitSpheres(const SphereSet& set, const Ray& r, float tMin, float tMax, HitRecord* rec)
{
    bool  hitAnything = false;
    float closest = tMax;

    for (int i = 0; i < set.count; ++i)
    {
        HitRecord temp;
        if (hitSphere(set.items[i], r, tMin, closest, &temp))
        {
            hitAnything = true;
            closest = temp.t;                               // shrink the interval
            *rec = temp;
        }
    }
    return hitAnything;
}

// Any-hit variant (no nearest search, early out) -- this is the version the
// shadow rays use, where only "is something in the way?" matters.
RT_HD RT_INLINE bool occludedSpheres(const SphereSet& set, const Ray& r, float tMin, float tMax)
{
    for (int i = 0; i < set.count; ++i)
    {
        HitRecord temp;
        if (hitSphere(set.items[i], r, tMin, tMax, &temp))
            return true;
    }
    return false;
}
