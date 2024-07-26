#ifndef VEC3_CUH
#define VEC3_CUH
class vec3 {
    private:
        float e[3];
    
    public:
    __host__ __device__ vec3() {}
    __host__ __device__ vec3(float e0, float e1, float e2) {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
    }
    
    __host__ __device__ float x() const { return e[0]; }
    __host__ __device__ float y() const { return e[1]; }
    __host__ __device__ float z() const { return e[2]; }

    __host__ __device__  vec3& operator*= (const float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__  vec3& operator/= (const float t) {
        return *this *= 1/t;
    }

    __host__ __device__  vec3 operator- () const {
        return vec3(-e[0], -e[1], -e[2]);
    }

    __host__ __device__  vec3& operator+= (const vec3 &v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__  vec3& operator-= (const vec3 &v) {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        return *this;
    }

    __host__ __device__ double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    __host__ __device__ double length() const {
        return sqrt(length_squared());
    }
};


__host__ __device__ inline vec3 operator+ (const vec3 &u, const vec3 &v) {
    return vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

__host__ __device__ inline vec3 operator- (const vec3 &u, const vec3 &v) {
    return vec3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

__host__ __device__ inline vec3 operator* (const vec3 &u, const vec3 &v) {
    return vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

__host__ __device__ inline vec3 operator* (float t, const vec3 &v) {
    return vec3(t*v.x(), t*v.y(), t*v.z());
}

__host__ __device__ inline vec3 operator* (const vec3 &v, float t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/ (vec3 v, float t) {
    return (1/t) * v;
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
    return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.y() * v.z() - u.z() * v.y(),
                u.z() * v.x() - u.x() * v.z(),
                u.x() * v.y() - u.y() * v.x());
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}



#endif
    