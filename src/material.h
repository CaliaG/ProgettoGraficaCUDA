#pragma once

struct hit_record;

#include "core/ray.h"
#include "geometries/object.h"
#include "utils/utils.h"
#include "texture.h"

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f-r0) * pow((1.0f - cosine),5.0f);
}

__device__ bool refract(const vector3D& v, const vector3D& n, float ni_over_nt, vector3D& refracted) {
    vector3D uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    return false;
}

__device__ vector3D reflect(const vector3D& v, const vector3D& n) {
    return v - 2.0f * dot(v, n) * n;
}

class material  {
public:
    __device__ material() {}

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, /*curandState *local_rand_state*/ vector3D random) const {
        return false;
    }
    __device__ virtual color emit(const hit_record& hrec) const {
        return color(0.f, 0.f, 0.f);
    }
};

class lambertian : public material {
public:
    __device__ lambertian(const color& a) : albedo(a) {}

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, /*curandState *local_rand_state*/ vector3D random) const  {
        vector3D target = rec.p + rec.normal /*+ random_in_unit_sphere(local_rand_state)*/ + random;
        scattered = ray(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    }

    color albedo;
};

class lambertianTexture : public material {
public:
	__device__ lambertianTexture(Texture *a) : albedo(a) {}
	
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, /*curandState *local_rand_state*/ vector3D random) const {
		vector3D target = rec.normal /*+ random_in_unit_sphere(local_rand_state)*/+ random;
		scattered = ray(rec.p, target);
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return true;
	}

	Texture *albedo;
};

class metal : public material {
public:
    __device__ metal(const color& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, /*curandState *local_rand_state*/ vector3D random) const  {
        vector3D reflected = reflect(normalize(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz /** random_in_unit_sphere(local_rand_state)*/* random);
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

    color albedo;
    float fuzz = 0.f;
};

class dielectric : public material {
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, /*curandState *local_rand_state*/ vector3D random) const  {
        vector3D outward_normal;
        vector3D reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vector3D(1.0f, 1.0f, 1.0f);
        vector3D refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ref_idx*ref_idx * (1 - cosine * cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
            reflect_prob = schlick(cosine, ref_idx);
        }
        else {
            reflect_prob = 1.0f;
        }
        scattered = ray(rec.p, refracted);
        return true;
    }

    float ref_idx;
};

class emitter : public material {
public:
    __device__ emitter(Texture* tex, float intensity = 1.f) : albedo(tex), _intensity(intensity) {}
    __device__ emitter(color c, float intensity = 1.f) : albedo(new constant_texture(c)), _intensity(intensity) {}

    __device__ virtual color emit(const hit_record& hrec) const {
        return albedo->value(hrec.u, hrec.v, hrec.p) * _intensity;
    }

    Texture* albedo;
    float _intensity;
};

class glossy: public material {
public:
    // Fuzz texture interpreted as the magnitude of the fuzz texture.
    __device__ glossy(Texture* a, Texture* f): albedo(a), fuzz(f){}

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, /*curandState *local_rand_state*/ vector3D random) const  {
        vector3D reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        float fuzz_factor = (fuzz->value(rec.u, rec.v, rec.p)).length();
        scattered = ray(rec.p, reflected+fuzz_factor*random);
        return true;
    }
 
    Texture* albedo;
    Texture* fuzz;
};

// See https://people.sc.fsu.edu/~jburkardt/data/mtl/mtl.html
// The MTL format is based on the Phong shading model, so this uses a bit of reinterpretation
// See https://www.scratchapixel.com/lessons/3d-basic-rendering/phong-shader-BRDF , and https://www.psychopy.org/api/visual/phongmaterial.html , and http://vr.cs.uiuc.edu/node198.html
// There are a few properties, which we allow to vary based on textures: 
// diffuse color: albedo for lambertian 
// specular color: albedo for metal
// emissive color: emissive :)
//
// sharpness map: remapped to fuzz := 1-log_10(sharpness)/4, sharpness clamped to [1, 10000]
//
// How to decide what happens? |color_for_type| / (sum ^type |color|), i.e. if color components add to 1, everything is fine, if not it is normalized
//
class mtl_material: public material {
public:
    __device__ mtl_material(
            Texture* diffuse_a, 
            Texture* specular_a, 
            Texture* emissive_a,
            Texture* transparency_map, 
            Texture* sharpness_map, 
            int illum,
            float random_f) : 
        emissive_text(emissive_a), 
        diffuse_text(diffuse_a), 
        specular_text(specular_a), 
        transparency_text(transparency_map), 
        roughness_text(new roughness_from_sharpness_texture(sharpness_map, 1, 10000)),
        random_float(random_f)
    {
        diffuse_mat = new lambertianTexture(diffuse_text);
        specular_mat = new glossy(specular_text, roughness_text);
        emissive_mat = new emitter(emissive_text);
    }

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, /*curandState *local_rand_state*/ vector3D random) const  {
        float transp_prob = transparency_prob(rec.u, rec.v, rec.p);
        if (transp_prob > random_float) {
            attenuation = transparency_text->value(rec.u, rec.v, rec.p);
            scattered = ray(rec.p, r_in.direction());
            return false;
        }
        return choose_mat(rec.u, rec.v, rec.p)->scatter(r_in, rec, attenuation, scattered, random);
    }

    __device__ virtual color emit(const hit_record& hrec) const {
        return emissive_mat->emit(hrec);
    }

private:
    material* emissive_mat;
    material* diffuse_mat;
    material* specular_mat;
    float random_float;

    __device__ inline float transparency_prob(float u, float v, const point3D& p) const {
        float diff = diffuse_text->value(u, v, p).length();
        float spec = specular_text->value(u, v, p).length();
        float transp = transparency_text->value(u, v, p).length();
        return transp / (transp+diff+spec+0.00001);
    }

    __device__ inline float diffuse_prob(float u, float v, const point3D& p) const {
        float diff = diffuse_text->value(u, v, p).length();
        float spec = specular_text->value(u, v, p).length();
        return diff / (diff+spec+0.00001);
    }

    __device__ inline material* choose_mat(float u, float v, const point3D& p) const {
        if (diffuse_prob(u, v, p) > random_float) {
            return diffuse_mat;
        } else {
            return specular_mat;
        }
    }

public:
    Texture* emissive_text;
    Texture* diffuse_text;
    Texture* specular_text;
    Texture* transparency_text;
    Texture* roughness_text;
};
