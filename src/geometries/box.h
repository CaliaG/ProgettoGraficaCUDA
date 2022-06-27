#pragma once

#include "object.h"
#include "object_list.h"
#include "rectangle.h"
#include "../utils/utils.h"

class flipNormals: public object {
public:
    __device__ flipNormals(object* p): ptr(p) {}

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        if (ptr->hit(r, t_min, t_max, rec)) {
            rec.normal = -rec.normal;
            return true;
        }
        else {
            return false;
        }
    }

    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const {
        return ptr->bounding_box(t0, t1, box);
    }

    object* ptr;
};

class box: public object {
public:
    __device__ box() {
        p_min = point3D(0,0,0);
        p_max = point3D(1,1,1);
    }
    __device__ box(const point3D& p0, const point3D& p1, material* mat);

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const{
        box = aabb(p_min, p_max);
        return true;
    }

    point3D p_min, p_max;
    object* list_ptr;
};

__device__ box::box(const point3D& p0, const point3D& p1, material* mat) {
    p_min = p0;
    p_max = p1;

    object** list = new object*[6];
    list[0] =                (new rectangleXY(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), mat));
    list[1] = new flipNormals(new rectangleXY(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), mat));
    list[2] =                (new rectangleXZ(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), mat));
    list[3] = new flipNormals(new rectangleXZ(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), mat));
    list[4] =                (new rectangleYZ(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), mat));
    list[5] = new flipNormals(new rectangleYZ(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), mat));
    list_ptr = new object_list(list, 6);
}

__device__ bool box::hit(const ray& r, float t_min, float t_max, hit_record& rec) const{
    return list_ptr->hit(r, t_min, t_max, rec);
}
