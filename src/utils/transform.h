#pragma once

#include "../geometries/object.h"

class Translate: public object {
public:
    __device__ Translate(object* p, const vector3D& displacement): ptr(p), offset(displacement) {}

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;

    object* ptr;
    vector3D offset;
};

__device__ bool Translate::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    ray moved_r(r.origin() - offset, r.direction());
    if (ptr->hit(moved_r, t_min, t_max, rec)){
        rec.p += offset;
        return true;
    }
    return false;
}

__device__ bool Translate::bounding_box(float t0, float t1, aabb& box) const {
    if (ptr->bounding_box(t0, t1, box)){
        box = aabb(box.min() + offset, box.max() + offset);
        return true;
    }
    return false;
}

class Rotate: public object {
public:
    __device__ Rotate(object* p, float angle);

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        return false;
    }
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const {
        box = bbox;
        return hasbox;
    }

    float sin_theta;
    float cos_theta;
    bool hasbox;
    aabb bbox;
    object* ptr;
};

__device__ Rotate::Rotate(object* p, float angle): ptr(p) {
    float radians = (M_PI / 180.) * angle;
    sin_theta = sin(radians);
    cos_theta = cos(radians);
    hasbox = ptr->bounding_box(0, 1, bbox);
    vector3D min( FLT_MAX,  FLT_MAX,  FLT_MAX);
    vector3D max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                float x = i * bbox.max().x() + (1-i) * bbox.min().x();
                float y = j * bbox.max().y() + (1-j) * bbox.min().y();
                float z = k * bbox.max().z() + (1-k) * bbox.min().z();
                float newx =  cos_theta * x + sin_theta * z;
                float newz = -sin_theta * x + cos_theta * z;
                vector3D tester(newx, y, newz);
                for (int c = 0; c < 3; c++){
                    if (tester[c] > max[c]) max[c] = tester[c];
                    if (tester[c] < min[c]) min[c] = tester[c];
                }
            }
        }
    }
    bbox = aabb(min, max);
}

class RotateY: public Rotate {
public:
    __device__ RotateY(object* p, float angle) : Rotate(p, angle) {}

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
};

__device__ bool RotateY::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vector3D origin = r.origin();
    vector3D direction = r.direction();
    origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
    origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];
    direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
    direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];
    ray rotate_r(origin, direction);
    if (ptr->hit(rotate_r, t_min, t_max, rec)) {
        vector3D p = rec.p;
        vector3D normal = rec.normal;
        p[0] =  cos_theta * rec.p[0] + sin_theta * rec.p[2];
        p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];
        normal[0] =  cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
        normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];
        rec.p = p;
        rec.normal = normal;
        return true;
    }
    return false;
}
