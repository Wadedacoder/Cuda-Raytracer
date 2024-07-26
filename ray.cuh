#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

class Ray {
    public:
        __host__ __device__ Ray() {}
        __host__ __device__ Ray(const vec3& origin, const vec3& direction) : orig(origin), dir(direction) {}

        __host__ __device__ vec3 origin() const { return orig; }
        __host__ __device__ vec3 direction() const { return dir; }

        __host__ __device__ vec3 at(double t) const {
            return orig + t * dir;
        }

    private:
        vec3 orig;
        vec3 dir;
};

#endif