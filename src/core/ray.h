#pragma once

#include "../math/vector3D.h"

class ray {
public:
    __host__ __device__ ray() {}
    __host__ __device__ ray(const point3D& _origin, const vector3D& _direction) { o = _origin; d = _direction; }

    __host__ __device__ point3D origin() const { return o; }
    __host__ __device__ vector3D direction() const { return d; }
    __host__ __device__ float time() const { return tm; }
    __host__ __device__ vector3D point_at_parameter(float t) const { return o + t * d; }

    point3D o;
    vector3D d;
    float tm;
};
