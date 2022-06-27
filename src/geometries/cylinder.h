#pragma once

#include <math.h> 
#include "object.h"
#include "../math/vector3D.h"

class cylinder : public object {
public:

	__device__ cylinder() {}
	__device__ cylinder(const float bottom, const float top, const float r) : y0(bottom), y1(top), radius(r), inv_radius(1.0f/r) {};
	__device__ cylinder(const float bottom, const float top, const float r, const material *m) : y0(bottom), y1(top), radius(r), inv_radius(1.0f/r) {};
	
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;

	__device__ virtual void textCoord(hit_record& rec) const;

private:
	float y0;				// bottom y value
	float y1;				// top y value
	float radius;			// radius
	float inv_radius;  		// one over the radius	
};

__device__ bool cylinder::bounding_box(float t0, float t1, aabb& box) const {
	box = aabb(point3D(-radius, y0, -radius), point3D(radius, y1, radius));
	return true;
}

__device__ inline void cylinder::textCoord(hit_record& rec) const {
	float phi = atan2(rec.p.x(), rec.p.z());
	if (phi < 0.0) phi += 2 * M_PI;
	rec.u = phi * (1.0 / (2 * M_PI));
	rec.v = rec.p.y() / (y1 - y0);
}

__device__ bool cylinder::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	float temp;
	float ox = r.origin().x();
	float oy = r.origin().y();
	float oz = r.origin().z();
	float dx = r.direction().x();
	float dy = r.direction().y();
	float dz = r.direction().z();

	float a = dx * dx + dz * dz;
	float b = 2.0f * (ox * dx + oz * dz);
	float c = ox * ox + oz * oz - radius * radius;
	float disc = b * b - 4.0f * a * c;

	if (disc < 0.0)
		return(false);
	else {
		float e = sqrt(disc);
		float denom = 2.0f * a;
		temp = (-b - e) / denom;    // smaller root

		if (temp < t_max && temp > t_min) {
			float yhit = oy + temp * dy;

			if (yhit > y0 && yhit < y1) {
				rec.t = temp;
				rec.normal = normalize(vector3D((ox + temp * dx) * inv_radius, 0.0f, (oz + temp * dz) * inv_radius));
				rec.p = r.point_at_parameter(rec.t);
				//rec.m = mat;

				// test for hitting from inside
				if (dot(-r.direction(), rec.normal) < 0.0f)
					rec.normal = -rec.normal;
				
				textCoord(rec);

				return (true);
			}
		}

		temp = (-b + e) / denom;    // larger root
		
		if (temp < t_max && temp > t_min) {
			float yhit = oy + temp * dy;

			if (yhit > y0 && yhit < y1) {
				rec.t = temp;
				rec.normal = normalize(vector3D((ox + temp * dx) * inv_radius, 0.0f, (oz + temp * dz) * inv_radius));
				rec.p = r.point_at_parameter(rec.t);
				//rec.m = mat;

				// test for hitting inside surface
				if (dot(-r.direction(), rec.normal) < 0.0f)
					rec.normal = -rec.normal;

				textCoord(rec);

				return (true);
			}
		}
	}

	return (false);
}
