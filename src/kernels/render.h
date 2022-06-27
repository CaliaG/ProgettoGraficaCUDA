#pragma once

#include "../core/camera.h"
#include "../core/ray.h"
#include "../math/vector3D.h"
#include "../geometries/object.h"
#include "../external.h"
#include "../material.h"
#include "../utils/bvh.h"
#include "../utils/utils.h"

__device__ color notHitted(ray& cur_ray, color& cur_attenuation, Texture** background) {
    vector3D unit_direction = unit_vector(cur_ray.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    //color c = (1.0f - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
    
    // -- BACKGROUND (HDR)
    float u, v;
    get_spherical_uv(unit_direction, u, v);
    color c = background[0]->value(u, v, unit_direction);

    return cur_attenuation * c;

    // -- return world color
    //return cur_attenuation;
}

// -- WORLD
__device__ vector3D shot(const ray& r, object **world, int bounces, curandState *local_rand_state, Texture** background) {
    ray cur_ray = r;
    vector3D cur_attenuation = vector3D(1.0,1.0,1.0);
    for (int i = 0; i < bounces; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            color attenuation;
            color emit = rec.mat_ptr->emit(rec) + color(0.1f, 0.1f, 0.1f); // bloomy effect
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, /*local_rand_state*/random_in_unit_sphere(local_rand_state))) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return emit;
            }
        } else {
            return notHitted(cur_ray, cur_attenuation, background);
        }
    }
    return vector3D(0.f, 0.f, 0.f); // exceeded recursion
}

__global__ void render(vector3D *fb, int max_x, int max_y, int ns, int bounces, camera **cam, object **world, curandState *rand_state, Texture** background) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    color col(0,0,0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += shot(r, world, bounces, &local_rand_state, background);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

// -- BVH
__device__ vector3D shot(const ray& r, BVHNode* bvh_root, int bounces, curandState *local_rand_state, Texture** background) {
    ray cur_ray = r;
    vector3D cur_attenuation = vector3D(1.0,1.0,1.0);
    for (int i = 0; i < bounces; i++) {
        hit_record rec;
        if (hit_BVH(bvh_root, cur_ray, .0001, FLT_MAX, rec)) {
            ray scattered;
            color attenuation;
            color emit = rec.mat_ptr->emit(rec) + color(0.1f, 0.1f, 0.1f); // bloomy effect
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, /*local_rand_state*/random_in_unit_sphere(local_rand_state))) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return emit;
            }
        } else {
            return notHitted(cur_ray, cur_attenuation, background);
        }
    }
    return vector3D(0.f, 0.f, 0.f); // exceeded recursion
}

__global__ void render(vector3D *fb, int max_x, int max_y, int ns, int bounces, camera **cam, BVHNode* bvh_root, curandState *rand_state, Texture** background) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    color col(0,0,0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += shot(r, bvh_root, bounces, &local_rand_state, background);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    curand_init(1984, 0, 0, rand_state);
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j*max_x + i;
    int seed = 2022;
    curand_init(seed + pixel_index, 0, 0, &rand_state[pixel_index]);
}
