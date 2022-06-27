#pragma once

#include "object.h"

class sphere: public object  {
public:
    __device__ sphere() {}
    __device__ sphere(point3D cen, float r, material *m) : center(cen), radius(r), mat_ptr(m)  {};

    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;
    
    point3D center;
    float radius;
    material *mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vector3D oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

__device__ bool sphere::bounding_box(float t0, float t1, aabb& box) const{
    box = aabb(center - vector3D(radius, radius, radius), center + vector3D(radius, radius, radius));
    return true;
}

class movingSphere: public object {
public:
    __device__ movingSphere() {}
    __device__ movingSphere(point3D cen0, point3D cen1, float t0, float t1, float r, material *m){
        center0 = cen0;
        center1 = cen1;
        time0   = t0;
        time1   = t1;
        radius  = r;
        mat_ptr = m;
    };
    
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;

    __device__ point3D center(float time) const;

    // centers at time0 and time1
    point3D center0;
    point3D center1;
    float time0;
    float time1;
    float radius;
    material *mat_ptr;
};

__device__ point3D movingSphere::center(float time) const{
    return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

__device__ bool movingSphere::bounding_box(float t0, float t1, aabb& box) const {
    aabb box0(center(t0) - vector3D(radius, radius, radius), center(t0) + vector3D(radius, radius, radius));
    aabb box1(center(t1) - vector3D(radius, radius, radius), center(t1) + vector3D(radius, radius, radius));
    box = surrounding_box(box0, box1);
    return true;
}

__device__ bool movingSphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const{
    vector3D oc = r.origin() - center(r.time());
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0){
        float sol = (-b - sqrt(discriminant)) / a;
        if (sol < t_max && sol > t_min){
            rec.t = sol;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center(r.time())) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        sol = (-b + sqrt(discriminant)) / a;
        if (sol < t_max && sol > t_min){
            rec.t = sol;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center(r.time())) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}