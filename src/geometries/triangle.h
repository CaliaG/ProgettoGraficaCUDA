#pragma once

#include "object.h"
#include "../math/point2D.h"

class triangle: public object {
public:
    __device__ triangle() : EPSILON(0.00001) {}
    __device__ triangle(point3D vs[3], material* _mat) : EPSILON(0.000001) {
        for (int i = 0; i < 3; i++) {
            vertices[i] = vs[i];
        }
		norm = normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
        mat = _mat;
    };
    __device__ triangle(point3D p1, point3D p2, point3D p3, material* _mat) : EPSILON(0.000001) {
        vertices[0] = p1;
        vertices[1] = p2;
        vertices[2] = p3;
		norm = normalize(cross(p2 - p1, p3 - p1));
        mat = _mat;
    };
    __device__ triangle(point3D p1, point3D p2, point3D p3, material* _mat, point2D tex0, point2D tex1, point2D tex2) : EPSILON(0.000001) {
        vertices[0] = p1;
        vertices[1] = p2;
        vertices[2] = p3;
		norm = normalize(cross(p2 - p1, p3 - p1));
        mat = _mat;
        uv0 = tex0;
        uv1 = tex1;
        uv2 = tex2;
    };

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;

    const float EPSILON;

    point3D vertices[3];
    material* mat;
	point2D uv0, uv1, uv2;
	vector3D norm;
};

__device__ bool triangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    point3D vertex0 = vertices[0];
    point3D vertex1 = vertices[1];
    point3D vertex2 = vertices[2];

    point3D edge1, edge2, h, s, q;
    float a, f, u, v;

    edge1 = vertex1 - vertex0;
    edge2 = vertex2 - vertex0;
    h = cross(r.direction(), edge2);
    a = dot(edge1, h);

    // Almost parallel to the triangle
    if (a > -EPSILON && a < EPSILON) return false;

    f = 1.0f / a;
    s = r.origin() - vertex0;
    u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f) return false;

    q = cross(s, edge1);
    v = f * dot(r.direction(), q);
    if (v < 0.0f || u + v > 1.0f) return false;

    // Compute final t
    float t = f * dot(edge2, q);

	if (t > t_min && t < t_max) {
		if (t > EPSILON) { // ray intersection
			rec.normal = norm;
			rec.t = dot((vertex0 - r.o), rec.normal) / dot(r.direction(), rec.normal);
			// rec.p = r.point_at_parameter(rec.t);
			point3D intersPoint = r.o + (normalize(r.d) * (t * magnitude(r.d)));
			rec.p = intersPoint;
            rec.mat_ptr = mat;
			rec.u = u * uv1.x + v * uv2.x + (1 - u - v) * uv0.x;
			rec.v = u * uv1.y + v * uv2.y + (1 - u - v) * uv0.y;
			return true;
		}
	}

    return false;
}

__device__ bool triangle::bounding_box(float t0, float t1, aabb& bbox) const {
    float minX = min(vertices[0][0], min(vertices[1][0], vertices[2][0]));
    float minY = min(vertices[0][1], min(vertices[1][1], vertices[2][1]));
    float minZ = min(vertices[0][2], min(vertices[1][2], vertices[2][2]));

    float maxX = max(vertices[0][0], max(vertices[1][0], vertices[2][0]));
    float maxY = max(vertices[0][1], max(vertices[1][1], vertices[2][1]));
    float maxZ = max(vertices[0][2], max(vertices[1][2], vertices[2][2]));

    bbox = aabb(point3D(minX, minY, minZ), point3D(maxX, maxY, maxZ));
    return true;
}
