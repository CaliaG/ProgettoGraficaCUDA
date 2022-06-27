#pragma once

#include "../core/ray.h"
#include "../math/aabb.h"

class material;

struct hit_record {
    float t;
    float u;
    float v;
    point3D p;
    vector3D normal;
    material *mat_ptr;
};

class object  {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const = 0;
};
