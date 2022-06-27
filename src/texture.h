#pragma once

#include "math/vector3D.h"
#include "utils/perlin_noise.h"

__device__ static void get_spherical_uv(const point3D &p, float &u, float &v) {
    auto theta = acos(-p.y());
    auto phi = atan2(-p.z(), p.x()) + M_PI;

    u = phi / (2 * M_PI);
    v = theta / M_PI;
}

class Texture {
public:
	__device__ virtual color value(float u, float v, const point3D& p) const = 0;
};

class constant_texture : public Texture {
public:
	__device__ constant_texture() { }
	__device__ constant_texture(color c) : color(c) { }
	
	__device__ virtual color value(float u, float v, const point3D& p) const {
		return color;
	}
	
	color color;
};

class checker_texture : public Texture {
public:
	__device__ checker_texture() { }
	__device__ checker_texture(Texture *t0, Texture *t1) : even(t0), odd(t1) { }
	
	__device__ virtual color value(float u, float v, const point3D& p) const;
	
	Texture *odd;
	Texture *even;
};

__device__ color checker_texture::value(float u, float v, const point3D& p) const {
	float sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
	if (sines < 0) {
		return odd->value(u, v, p);
	}
	else {
		return even->value(u, v, p);
	}
}

class image_texture : public Texture {
public:
    __device__ image_texture() {}
    __device__ image_texture(float* buffer, int width, int height) : _data(buffer), _width(width), _height(height) {}

    __device__ virtual color value(float u, float v, const color& p) const {
        int i = u * _width;
        int j = (1 - v) * _height - 0.001;
        if (i < 0) i = 0;
        if (j < 0) j = 0;
        if (i > _width - 1) i = _width - 1;
        if (j > _height - 1) j = _height - 1;

		int index = j * _width + i;
        float r = _data[index * 3 + 0];
        float g = _data[index * 3 + 1];
        float b = _data[index * 3 + 2];

        return color(r, g, b);
    }
	
private:
    float* _data;
    int _width = 0;
    int _height = 0;
};

enum class noise_type : uint8_t {
    PERLIN,
    TURBULANCE,
    MARBLE,
    UNKNOWN
};

class noise_texture : public Texture {
public:
    __device__ noise_texture(noise_type ntype = noise_type::PERLIN, float density = 4.f)
        : _density(density), _ntype(ntype) {
        if (_density <= 0.f) {
            _density = 4.f;
        }
    }
    __device__ virtual color value(float u, float v, const point3D& p) const override {
        if (_ntype == noise_type::PERLIN) {
            return vector3D(1, 1, 1) * _noise.noise(p * _density);
        } else if (_ntype == noise_type::TURBULANCE) {
            return vector3D(1.f) * 0.5 * _noise.turbulance_noise(p * _density);
        } else if (_ntype == noise_type::MARBLE) {
            float value = 0.5f * (1 + __sinf((p.z() * _density + 7 * _noise.turbulance_noise(p))));

            color color1(0.925, 0.816, 0.78);
            color color2(0.349/2, 0.431/2, 0.498/2);
            return color1 * value + color2 * (1 - value);
        } else {
            return color(1, 1, 1);
        }
    }
private:
    perlin_noise _noise;
    float _density;
    noise_type _ntype;
};

class wood_texture : public Texture {
public:
     __device__ wood_texture(
        const color& color1,
        const color& color2,
        float density = 4.f,
        float hardness = 50.f
    ) : _color1(color1), _color2(color2), _density(density), _hardness(hardness)
    {
        if (_density <= 0.f) { //avoid division by zero
            _density = 4.f;
        }
    }
     __device__ virtual inline color value(float u, float v, const point3D& p) const override {
        float n = _hardness * _noise.noise(vector3D(p.x(), p.y(), p.z()) / _density);
        n -= floorf(n);
        return (_color1 * n) + (_color2 * (1.f - n));
    }
private:
    float _density;
    float _hardness;
    color _color1;
    color _color2;
    perlin_noise _noise;
};

class roughness_from_sharpness_texture: public Texture {
public:
    __device__ roughness_from_sharpness_texture() {}

    __device__ roughness_from_sharpness_texture(Texture* sharpness_map, float min_v, float max_v)
                    : sharpness_text(sharpness_map), l_min_val(log(min_v)), l_max_val(log(max_v)) {}

    __device__ virtual inline color value(float u, float v, const point3D& p) const override {
        return color(1, 0, 0) * clamp(
                log(sharpness_text->value(u, v, p).length()+0.00001), l_min_val, l_max_val)
            / (l_max_val-l_min_val);
    }

public:
    Texture* sharpness_text;
private: 
    float l_min_val, l_max_val;
};
