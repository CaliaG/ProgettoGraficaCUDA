#pragma once

#include "../external.h"
#include "../core/ray.h"
#include "../utils/utils.h"
#include "../math/vector3D.h"

class aabb {
public:
  __device__ aabb() {}
  __device__ aabb(const point3D &a, const point3D &b) {
      _min = a;
      _max = b;
      cent = (_max - _min) / 2.0;
      volume = compute_box_volume(a, b);
  }

  __device__ point3D min() const { return _min; }
  __device__ point3D max() const { return _max; }

  __device__ void set_min(const point3D &min) { _min = min; }
  __device__ void set_max(const point3D &max) { _max = max; }

  __device__ float compute_box_volume(const point3D &a, const point3D &b) const {
      point3D minv(dfmin(a.x(), b.x()),
                dfmin(a.y(), b.y()),
                dfmin(a.z(), b.z()));
      point3D maxv(dfmax(a.x(), b.x()),
                dfmax(a.y(), b.y()),
                dfmax(a.z(), b.z()));
      return (maxv.x() - minv.x()) * (maxv.y() - minv.y()) * (maxv.z() - minv.z());
  }

  __device__ bool hit(const ray &r, float tmin, float tmax) const {
      for (int a = 0; a < 3; a++) {
          float t0 = dfmin((_min[a] - r.origin()[a]) / r.direction()[a],
                           (_max[a] - r.origin()[a]) / r.direction()[a]);
          float t1 = dfmax((_min[a] - r.origin()[a]) / r.direction()[a],
                           (_max[a] - r.origin()[a]) / r.direction()[a]);
          tmin = dfmax(t0, tmin);
          tmax = dfmin(t1, tmax);
          if (tmax <= tmin) return false;
      }
      return true;
  }

  __device__ point3D center() const { return cent; }

public:
    point3D _min;
    point3D _max;
    float volume;
    point3D cent;
};

// get union of two aabb, for temporary use
__device__ aabb surrounding_box(aabb b1, aabb b2) {
    point3D b1min = b1.min();
    point3D b2min = b2.min();
    point3D small(dfmin(b1min.x(), b2min.x()),
                  dfmin(b1min.y(), b2min.y()),
                  dfmin(b1min.z(), b2min.z()));

    point3D b1max = b1.max();
    point3D b2max = b2.max();

    point3D big(dfmax(b1max.x(), b2max.x()),
                dfmax(b1max.y(), b2max.y()),
                dfmax(b1max.z(), b2max.z()));

    return aabb(small, big);
}
