#pragma once

#include "../external.h"

class vector3D  {

public:
    __host__ __device__ vector3D() {}
    __host__ __device__ vector3D(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    __host__ __device__ vector3D(float e1) { e[0] = e1; e[1] = e1; e[2] = e1; }

    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return x(); }
    __host__ __device__ inline float g() const { return y(); }
    __host__ __device__ inline float b() const { return z(); }

    __host__ __device__ inline const vector3D& operator+() const { return *this; }
    __host__ __device__ inline vector3D operator-() const { return vector3D(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; };

    __host__ __device__ inline vector3D& operator+=(const vector3D &v2);
    __host__ __device__ inline vector3D& operator-=(const vector3D &v2);
    __host__ __device__ inline vector3D& operator*=(const vector3D &v2);
    __host__ __device__ inline vector3D& operator/=(const vector3D &v2);
    __host__ __device__ inline vector3D& operator*=(const float t);
    __host__ __device__ inline vector3D& operator/=(const float t);
    __host__ __device__ inline vector3D& operator+=(const float t);
    __host__ __device__ inline vector3D& operator-=(const float t);

    __host__ __device__ inline float squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    __host__ __device__ inline float length() const { return sqrt(squared_length()); }
    __host__ __device__ inline void make_unit_vector();

    float e[3];
};

inline std::istream& operator>>(std::istream &is, vector3D &t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vector3D &t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

__host__ __device__ inline void vector3D::make_unit_vector() {
    float k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline vector3D operator+(const vector3D &v1, const vector3D &v2) {
    return vector3D(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vector3D operator-(const vector3D &v1, const vector3D &v2) {
    return vector3D(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vector3D operator*(const vector3D &v1, const vector3D &v2) {
    return vector3D(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vector3D operator/(const vector3D &v1, const vector3D &v2) {
    return vector3D(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vector3D operator*(float t, const vector3D &v) {
    return vector3D(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vector3D operator/(const vector3D &v, float t) {
    return vector3D(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__host__ __device__ inline vector3D operator*(const vector3D &v, float t) {
    return vector3D(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vector3D operator+(const vector3D &v1, float t) {
    return vector3D(v1.x() + t, v1.y() + t, v1.z() + t);
}

__host__ __device__ inline vector3D operator-(const vector3D &v1, float t) {
    return vector3D(v1.x() - t, v1.y() - t, v1.z() - t);
}

__host__ __device__ inline float dot(const vector3D &v1, const vector3D &v2) {
    return v1.e[0] *v2.e[0] + v1.e[1] *v2.e[1]  + v1.e[2] *v2.e[2];
}

__host__ __device__ inline vector3D cross(const vector3D &v1, const vector3D &v2) {
    return vector3D( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
                     (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
                     (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}

__host__ __device__ inline vector3D& vector3D::operator+=(const vector3D &v){
    e[0]  += v.e[0];
    e[1]  += v.e[1];
    e[2]  += v.e[2];
    return *this;
}

__host__ __device__ inline vector3D& vector3D::operator*=(const vector3D &v){
    e[0]  *= v.e[0];
    e[1]  *= v.e[1];
    e[2]  *= v.e[2];
    return *this;
}

__host__ __device__ inline vector3D& vector3D::operator/=(const vector3D &v){
    e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];
    return *this;
}

__host__ __device__ inline vector3D& vector3D::operator-=(const vector3D& v) {
    e[0]  -= v.e[0];
    e[1]  -= v.e[1];
    e[2]  -= v.e[2];
    return *this;
}

__host__ __device__ inline vector3D& vector3D::operator*=(const float t) {
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    return *this;
}

__host__ __device__ inline vector3D& vector3D::operator/=(const float t) {
    float k = 1.0/t;

    e[0]  *= k;
    e[1]  *= k;
    e[2]  *= k;
    return *this;
}

__host__ __device__ inline vector3D &vector3D::operator+=(const float t) {
    e[0] += t;
    e[1] += t;
    e[2] += t;
    return *this;
}
__host__ __device__ inline vector3D &vector3D::operator-=(const float t) {
    e[0] -= t;
    e[1] -= t;
    e[2] -= t;
    return *this;
}

__host__ __device__ inline vector3D unit_vector(const vector3D v) {
    return v / v.length();
}

__host__ __device__ inline float magnitude(const vector3D& v) {
	return v.length();
}

__host__ __device__ inline vector3D normalize(const vector3D& v) {
	return v / magnitude(v);
}

__host__ __device__ vector3D min_vec(const vector3D &v1, const vector3D &v2) {
    float xmin = fmin(v1.x(), v2.x());
    float ymin = fmin(v1.y(), v2.y());
    float zmin = fmin(v1.z(), v2.z());
    return vector3D(xmin, ymin, zmin);
}
__host__ __device__ vector3D max_vec(const vector3D v1, const vector3D v2) {
    float xmax = fmax(v1.x(), v2.x());
    float ymax = fmax(v1.y(), v2.y());
    float zmax = fmax(v1.z(), v2.z());
    return vector3D(xmax, ymax, zmax);
}

using point3D = vector3D;
using color = vector3D;
