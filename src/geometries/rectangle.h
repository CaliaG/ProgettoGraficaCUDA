#pragma once

#include "object.h"

/**
 * Rectangle along XY axes
 */
class rectangleXY: public object {
public:
    __device__ rectangleXY() {};
    __device__ rectangleXY(float _x0, float _x1, float _y0, float _y1, float _k, material* mat):
        x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mat_ptr(mat) {};
    
    __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const{
        box = aabb(vector3D(x0, y0, k - 0.0001), vector3D(x1, y1, k + 0.0001));
        return true;
    }

    float x0, x1, y0, y1, k;
    material* mat_ptr;
};

__device__ bool rectangleXY::hit(const ray& r, float t0, float t1, hit_record& rec) const{
    float t = (k - r.origin().z()) / r.direction().z();
    if (t < t0 || t > t1) return false;

    float x = r.origin().x() + t * r.direction().x();
    float y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1) return false;

    rec.u = (x - x0) / (x1 - x0);
    rec.v = (y - y0) / (y1 - y0);
    rec.t = t;
    rec.mat_ptr = mat_ptr;
    rec.p = r.point_at_parameter(t);
    rec.normal = vector3D(0, 0, 1);
    return true;
}

/**
 * Rectangle along XZ axes
 */
class rectangleXZ: public object {
public:
    __device__ rectangleXZ() {};
    __device__ rectangleXZ(float _x0, float _x1, float _z0, float _z1, float _k, material* mat):
        x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mat_ptr(mat) {};
    
    __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const{
        box = aabb(vector3D(x0, k - 0.0001, z0), vector3D(x1, k + 0.0001, z1));
        return true;
    }

    float x0, x1, z0, z1, k;
    material* mat_ptr;
};

__device__ bool rectangleXZ::hit(const ray& r, float t0, float t1, hit_record& rec) const{
    float t = (k - r.origin().y()) / r.direction().y();
    if(t < t0 || t > t1) return false;

    float x = r.origin().x() + t * r.direction().x();
    float z = r.origin().z() + t * r.direction().z();
    if(x < x0 || x > x1 || z < z0 || z > z1) return false;

    rec.u = (x - x0) / (x1 - x0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    rec.mat_ptr = mat_ptr;
    rec.p = r.point_at_parameter(t);
    rec.normal = vector3D(0, 1, 0);
    return true;
}


/**
 * Rectangle along YZ axes
 */
class rectangleYZ: public object {
public:
    __device__ rectangleYZ() {};
    __device__ rectangleYZ(float _y0, float _y1, float _z0, float _z1, float _k, material* mat):
        y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mat_ptr(mat) {};
    
    __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const{
        box = aabb(vector3D(k - 0.0001, y0, z0), vector3D(k + 0.0001, y1, z1));
        return true;
    }

    float y0, y1, z0, z1, k;
    material* mat_ptr;
};

__device__ bool rectangleYZ::hit(const ray& r, float t0, float t1, hit_record& rec) const{
    float t = (k - r.origin().x()) / r.direction().x();
    if(t < t0 || t > t1) return false;

    float y = r.origin().y() + t * r.direction().y();
    float z = r.origin().z() + t * r.direction().z();
    if(y < y0 || y > y1 || z < z0 || z > z1) return false;

    rec.u = (y - y0) / (y1 - y0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    rec.mat_ptr = mat_ptr;
    rec.p = r.point_at_parameter(t);
    rec.normal = vector3D(1, 0, 0);
    return true;
}
