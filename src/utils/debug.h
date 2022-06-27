#pragma once

#include "../external.h"

void cuda_control(cudaError_t res, const char *const fn, const char *const f, const int l) {
    if (res != cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA ERROR :: " << static_cast<unsigned int>(res) << std::endl
            << cudaGetErrorName(res) << " file: " << f << std::endl
            << " line: " << l << std::endl
            << " function: " << fn << std::endl;
        cudaDeviceReset();
        std::string s = ss.str();
        throw std::runtime_error(s.c_str());
        exit(99);
    } 
}

#define CUDA_CONTROL(v) cuda_control((v), #v, __FILE__, __LINE__)

void get_heap_size() {
    size_t limit;
    CUDA_CONTROL(cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize));
    //std::cerr << "• thread limit: " << static_cast<unsigned int>(limit) << std::endl;
    printf("• thread limit: %lu MB\n", limit / (1024 * 1024));
}

void set_heap_size(size_t rsize) {
    CUDA_CONTROL(cudaDeviceSetLimit(cudaLimitMallocHeapSize, rsize));
}

void set_1GB_heap_size() {
    size_t rsize = 1024ULL*1024ULL*1024ULL*1ULL;  // allocate 1GB
    CUDA_CONTROL(cudaDeviceSetLimit(cudaLimitMallocHeapSize, rsize));
}

void gpu_memory() {
    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount(&num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        cudaSetDevice(gpu_id);
        int id;
        cudaGetDevice(&id);
        cudaMemGetInfo(&free, &total);
        printf("• GPU %i > memory[free = %lu MB, total = %lu MB]\n", id, free / (1024 * 1024), total / (1024 * 1024));
    }
}
