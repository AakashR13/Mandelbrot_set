#ifndef GPU_UTILS__H
#define GPU_UTILS__H

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <chrono>


#define ENABLE_CUDA_CHECK 0
#define ENABLE_STREAMS 0
#define N_STREAMS 3


void checkCudaError(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) 
                  << " \"" << cudaGetErrorString(result) << "\" in " << func << std::endl;
        exit(EXIT_FAILURE);
    }
}

#if ENABLE_CUDA_CHECK
    #define CUDA_CHECK(val) checkCudaError((val), #val, __FILE__, __LINE__)
#else
    #define CUDA_CHECK(val) val 
#endif
//

int _ConvertSMVer2Cores(int major, int minor) {
    typedef struct {
        int SM;
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        { 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128 }, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128 }, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64  }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128 }, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128 }, // Pascal Generation (SM 6.2) GP10x class
        { 0x70, 64  }, // Volta Generation (SM 7.0) GV100 class
        { 0x72, 64  }, // Volta Generation (SM 7.2) GV11B class
        { 0x75, 64  }, // Turing Generation (SM 7.5) TU10x class
        { 0x80, 64  }, // Ampere Generation (SM 8.0) GA100 class
        { 0x86, 128 }, // Ampere Generation (SM 8.6) GA10x class
        { 0x87, 128 }, // Ampere Generation (SM 8.7) GA10x class
        { 0x90, 128 }, // Hopper Generation (SM 9.0) GH100 class
        { -1, -1 }
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    std::cerr << "MapSMtoCores undefined SM version " << major << "." << minor << std::endl;
    return -1;
}

void printDeviceProperties(int device, std::ofstream &out) {
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));

    out << "Device " << device << ": " << deviceProp.name << std::endl;
    out << "  Total amount of global memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
    out << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
    out << "  CUDA Cores/MP: " << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) << std::endl;
    out << "  Total CUDA Cores: " << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount << std::endl;
    out << "  GPU Max Clock rate: " << deviceProp.clockRate * 1e-3f << " MHz" << std::endl;
    out << "  Memory Clock rate: " << deviceProp.memoryClockRate * 1e-3f << " MHz" << std::endl;
    out << "  Memory Bus Width: " << deviceProp.memoryBusWidth << "-bit" << std::endl;
    out << "  L2 Cache Size: " << deviceProp.l2CacheSize << " bytes" << std::endl;
    out << "  Max Texture Dimension Size (x,y,z): 1D=(" << deviceProp.maxTexture1D << "), 2D=(" << deviceProp.maxTexture2D[0] << "," << deviceProp.maxTexture2D[1] << "), 3D=(" << deviceProp.maxTexture3D[0] << "," << deviceProp.maxTexture3D[1] << "," << deviceProp.maxTexture3D[2] << ")" << std::endl;
    out << "  Max Layered Texture Size (dim) x layers: 1D=(" << deviceProp.maxTexture1DLayered[0] << ") x " << deviceProp.maxTexture1DLayered[1] << ", 2D=(" << deviceProp.maxTexture2DLayered[0] << "," << deviceProp.maxTexture2DLayered[1] << ") x " << deviceProp.maxTexture2DLayered[2] << std::endl;
    out << "  Total amount of constant memory: " << deviceProp.totalConstMem << " bytes" << std::endl;
    out << "  Total amount of shared memory per block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
    out << "  Total number of registers available per block: " << deviceProp.regsPerBlock << std::endl;
    out << "  Warp size: " << deviceProp.warpSize << std::endl;
    out << "  Maximum number of threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    out << "  Maximum number of threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    out << "  Max dimension size of a thread block (x,y,z): (" << deviceProp.maxThreadsDim[0] << "," << deviceProp.maxThreadsDim[1] << "," << deviceProp.maxThreadsDim[2] << ")" << std::endl;
    out << "  Max dimension size of a grid size (x,y,z): (" << deviceProp.maxGridSize[0] << "," << deviceProp.maxGridSize[1] << "," << deviceProp.maxGridSize[2] << ")" << std::endl;
    out << "  Concurrent copy and kernel execution: " << (deviceProp.deviceOverlap ? "Yes" : "No") << std::endl;
    out << "  Number of asynchronous engines: " << deviceProp.asyncEngineCount << std::endl;
    out << "  Unified addressing: " << (deviceProp.unifiedAddressing ? "Yes" : "No") << std::endl;
    out << "  Memory pitch: " << deviceProp.memPitch << std::endl;
    out << "-------------------------------------------------------------\n";
}

int getDeviceProps(const char* filepath){
    int nDevices;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));

    // std::cout << "Number of devices: " << nDevices << std::endl;
    std::ofstream out(filepath);
    if(!out.is_open()) {
        std::cerr << "Failed to open output file" << std::endl;
        return 1;
    }
    out << "Found " << nDevices << " CUDA-capable device(s)\n";
    for (int device = 0; device < nDevices; ++device) {
        printDeviceProperties(device, out);
    }

    out.close();
    return 0;
}
    
#define DURATION(val, fname) { \
    auto start = std::chrono::steady_clock::now(); \
    val \
    auto end = std::chrono::steady_clock::now(); \
    std::cout << "Time to generate " << fname << " = " << std::chrono::duration<double, std::milli>(end - start).count() << " [ms]" << std::endl; \
}

#define DURATION_MSG(val, fname, msg) { \
    auto start = std::chrono::steady_clock::now(); \
    val \
    auto end = std::chrono::steady_clock::now(); \
    std::cout << msg << " " << fname << " = " << std::chrono::duration<double, std::milli>(end - start).count() << " [ms]" << std::endl; \
}

// template<class T>
// __inline__ __host__ CUDART_DEVICE cudaError_t cudaOccupancyMaxPotentialBlockSize(
//     int    *minGridSize,
//     int    *blockSize,
//     T       func,
//     size_t  dynamicSMemSize = 0,
//     int     blockSizeLimit = 0)
// {
//     return cudaOccupancyMaxPotentialBlockSizeVariableSMem(minGridSize, blockSize, func, __cudaOccupancyB2DHelper(dynamicSMemSize), blockSizeLimit);
// }

// minGridSize     = Suggested min grid size to achieve a full machine launch.
// blockSize       = Suggested block size to achieve maximum occupancy.
// func            = Kernel function.
// dynamicSMemSize = Size of dynamically allocated shared memory. Of course, it is known at runtime before any kernel launch. The size of the statically allocated shared memory is not needed as it is inferred by the properties of func.
// blockSizeLimit  = Maximum size for each block. In the case of 1D kernels, it can coincide with the number of input elements.

#endif