#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <string.h>
#include <fstream>
#include <iostream>
#include "stbi_image_write.h"
#include <cuda_runtime.h>
#include <chrono>
#include "vec3.cuh"
#include "ray.cuh"

// make a timer
class Timer {
public:
    Timer() {
        start = std::chrono::high_resolution_clock::now();
    }

    void reset() {
        start = std::chrono::high_resolution_clock::now();
    }

    float elapsed() {
        std::chrono::duration<float> duration = std::chrono::high_resolution_clock::now() - start;
        return duration.count();
    }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start;

};



__device__ vec3 color(Ray* R){
    vec3 unit_direction = unit_vector(R->direction());
    float t = 0.5 * (unit_direction.y() + 1.0);

    // make a sphere 
    vec3 sphere_center = vec3(0, 0, -1);
    float sphere_radius = 0.5;

    vec3 oc = R->origin() - sphere_center;
    float a = dot(R->direction(), R->direction());
    float b = 2.0 * dot(oc, R->direction());
    float c = dot(oc, oc) - sphere_radius * sphere_radius;
    float discriminant = b*b - 4*a*c;

    if(discriminant > 0){
        float temp = (-b - sqrt(discriminant)) / (2.0 * a);
        if(temp > 0){
            vec3 N = unit_vector(R->at(temp) - sphere_center);
            return 0.5 * vec3(N.x() + 1, N.y() + 1, N.z() + 1);
        }
    }

    return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__host__ __device__ struct img_info{
    int width;
    int height;
    
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;

    vec3 pixel00_loc;
    vec3 viewport_upper_left;

    vec3 camera_center;
};


// kernel function where ray tracing is done

__global__ void raytrace(vec3* image, img_info img) {
    // Calculate the pixel's index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= img.width || y >= img.height) {
        return;
    }

    vec3 pixel_loc = img.pixel00_loc + x * img.pixel_delta_u + y * img.pixel_delta_v;
    vec3 direction = pixel_loc - img.camera_center;
    Ray R(img.camera_center, direction);

    image[y * img.width + x] = color(&R);
}


int main(int argc, char* argv[]) {
    // Image size
    const int width = 400;
    const float aspect_ratio = 16.0 / 9.0;
    const int height = static_cast<int>(width / aspect_ratio);
    const int channels = 3;
    const int size = width * height * channels;

    // view port
    float viewport_height = 2.0;
    float focal_length = 1.0;
    float viewport_width = viewport_height * static_cast<float>(width) / height;
    vec3 camera_center = vec3(0, 0, 0);

    vec3 viewport_u = vec3(viewport_width, 0, 0);
    vec3 viewport_v = vec3(0, -viewport_height, 0);

    vec3 pixel_delta_u = viewport_u / width;
    vec3 pixel_delta_v = viewport_v / height;

    vec3 viewport_upper_left = camera_center
                             - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    vec3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    img_info img = {width, height, pixel_delta_u, pixel_delta_v, pixel00_loc, viewport_upper_left, camera_center};

    // Allocate memory for the image
    vec3* image = new vec3[size];

    // Allocate memory on the device
    vec3* d_image;
    cudaMalloc(&d_image, size * sizeof(vec3));

    // timer
    Timer timer;
    // Launch the kernel
    dim3 threads(16, 16);
    dim3 blocks((width) / threads.x, (height ) / threads.y);
    raytrace<<<blocks, threads>>>(d_image, img);

    // print the time taken
    std::cout << "Time taken: " << timer.elapsed() << "s" << std::endl;

    // Copy the image back to the host
    cudaMemcpy(image, d_image, size * sizeof(vec3), cudaMemcpyDeviceToHost);

    // use stb_image_write to save the image
    unsigned char* image_c = new unsigned char[width * height * channels];

    // copy the image to the buffer
    for(int i = 0; i < width * height; i++) {
        image_c[i * channels + 0] = (unsigned char)(255.0 * image[i].x());
        image_c[i * channels + 1] = (unsigned char)(255.0 * image[i].y());
        image_c[i * channels + 2] = (unsigned char)(255.0 * image[i].z());
    }

    // save the image
    stbi_write_png("output.png", width, height, channels, image_c, width * channels);

    // Free the device memory
    cudaFree(d_image);
    

    // Free the image memory
    delete[] image;
    delete[] image_c;

    return 0;
}