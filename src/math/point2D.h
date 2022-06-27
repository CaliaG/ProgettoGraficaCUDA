#pragma once

#include "../external.h"

class point2D  {

public:
    __host__ __device__ point2D() {}
    __host__ __device__ point2D(float a, float b) { x = a; y = b; }

	__host__ __device__ inline float operator [](int i) const { return ((&x)[i]); }
	__host__ __device__ inline float& operator [](int i) { return ((&x)[i]); }

	union {
		float x, u;
	};
	union {
		float y, v;
	};
};

__host__ __device__ inline point2D operator*(const point2D& v, float s) {
	return (point2D(v.x * s, v.y * s));
}

__host__ __device__ inline point2D operator*(float s, const point2D& v) {
	return (point2D(v.x * s, v.y * s));
}

__host__ __device__ inline point2D operator+(const point2D& a, const point2D& b) {
	return (point2D(a.x + b.x, a.y + b.y));
}
