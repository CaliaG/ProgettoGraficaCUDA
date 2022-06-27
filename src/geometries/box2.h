#pragma once

#include "object.h"
#include "aarect.h"

class box2: public object {
public:
    __device__ box2() {
        p_min = point3D(0,0,0);
        p_max = point3D(1,1,1);
    }
    __device__ box2(const point3D& p0, const point3D& p1, material* mat);

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const{
        box = aabb(p_min, p_max);
        return true;
    }

    point3D p_min, p_max;
    object* list_ptr;
};

__device__ box2::box2(const point3D& p0, const point3D& p1, material* mat) {
    p_min = p0;
    p_max = p1;

    object** list = new object*[6];
    list[0] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), mat);
    list[1] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), mat);
    list[2] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), mat);
    list[3] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), mat);
    list[4] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), mat);
    list[5] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), mat);
    list_ptr = new object_list(list, 6);
}

__device__ bool box2::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    return list_ptr->hit(r, t_min, t_max, rec);
}
