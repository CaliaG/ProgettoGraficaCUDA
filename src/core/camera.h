#pragma once

#include "../external.h"
#include "../utils/utils.h"
#include "ray.h"

class camera {
public:
    __device__ camera() {}
    __device__ camera(vector3D lookfrom, vector3D lookat, vector3D vup, float vfov, float aspect) {
        float theta = degree_to_radian(vfov);
        float half_height = tan(theta / 2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
		lower_left_corner = -half_width * u - half_height * v - w;
        horizontal = 2.0f * half_width * u;
        vertical = 2.0f * half_height * v;
    }

    __device__ virtual ray get_ray(float s, float t, curandState *local_rand_state);

    vector3D origin;
    vector3D lower_left_corner;
    vector3D horizontal;
    vector3D vertical;
    vector3D u, v, w;
    float lens_radius;
};

__device__ ray camera::get_ray(float s, float t, curandState *local_rand_state) {
		return ray(origin, lower_left_corner + s * horizontal + t * vertical);
}

class camera_depth : public camera {
public:
    __device__ camera_depth() {}
    __device__ camera_depth(vector3D lookfrom, vector3D lookat, vector3D vup, float vfov, float aspect, float aperture, float focus_dist) {
        lens_radius = aperture / 2.0f;
        float theta = degree_to_radian(vfov);
        float half_height = tan(theta / 2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin  - half_width * focus_dist * u -half_height * focus_dist * v - focus_dist * w;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;
    }

    __device__ virtual ray get_ray(float s, float t, curandState *local_rand_state);

    vector3D origin;
    vector3D lower_left_corner;
    vector3D horizontal;
    vector3D vertical;
    vector3D u, v, w;
    float lens_radius;
};

__device__ ray camera_depth::get_ray(float s, float t, curandState *local_rand_state) {
    vector3D rd = lens_radius * random_in_unit_disk(local_rand_state);
    vector3D offset = u * rd.x() + v * rd.y();
    return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
}