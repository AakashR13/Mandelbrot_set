#include <iostream>
#include <vector>
#include <chrono>
#include <functional>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include "window.h"
#include "save_image.h"
#include "utils.h"
#include "params.h"

#define ENABLE_CUDA_CHECK 0
#define ENABLE_STREAMS 0


// Kernel Params
struct FractalParams {
    int scr_width;
    int scr_height;
    double fr_x_max;
    double fr_x_min;
    double fr_y_max;
    double fr_y_min;

    FractalParams(const window<int>& scr, const window<double>& fract)
    :   scr_width(scr.width()), scr_height(scr.height()),
        fr_x_max(fract.x_max()), fr_x_min(fract.x_min()),
        fr_y_max(fract.y_max()), fr_y_min(fract.y_min()) {}
};


// Alias for complex type
struct Complex {
    double real;
    double imag;

    __forceinline__ __host__ __device__ Complex(double r=0.0, double i=0.0) : real(r), imag(i) {}

    __forceinline__ __host__ __device__ double magnitude() const {
        return sqrt(real * real + imag * imag);
    }

    __forceinline__ __host__ __device__ Complex operator*(const Complex& other) const {
        return Complex(real * other.real - imag * other.imag, real * other.imag + imag * other.real);
    }

    __forceinline__ __host__ __device__ Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);    
    }
};

typedef Complex (*fractal_func_t)(Complex, Complex);

__forceinline__ __device__ Complex mandelbrot_func(Complex z, Complex c) {
    return z * z + c;
}

__forceinline__ __device__ Complex triple_mandelbrot_func(Complex z, Complex c) {
    return z * z * z + c;
}

__device__ fractal_func_t fractal_functions[] = {mandelbrot_func, triple_mandelbrot_func};

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

__host__ void getDeviceProps() {
    int nDevices;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));

    std::cout << "Number of devices: " << nDevices << std::endl;

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "Device Number: " << i << "\n";
        std::cout << "  Device name: " << prop.name << "\n";
        std::cout << "  Memory Clock Rate (MHz): " << prop.memoryClockRate/1024 << "\n";
        std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << "\n";
        std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth/8) / 1.0e6 << "\n";
        std::cout << "  Total global memory (Gbytes): " << static_cast<float>(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0 << "\n";
        std::cout << "  Shared memory per block (Kbytes): " << static_cast<float>(prop.sharedMemPerBlock) / 1024.0 << "\n";
        std::cout << "  minor-major: " << prop.minor << "-" << prop.major << "\n";
        std::cout << "  Warp-size: " << prop.warpSize << "\n";
        std::cout << "  Concurrent kernels: " << (prop.concurrentKernels ? "yes" : "no") << "\n";
        std::cout << "  Concurrent computation/communication: " << (prop.deviceOverlap ? "yes" : "no") << "\n\n";
    }
}

// Convert a pixel coordinate to the complex domain
    __forceinline__ __device__ Complex scale(FractalParams params, Complex c) {
        return Complex(c.real / static_cast<double>(params.scr_width) * (params.fr_x_max - params.fr_x_min) + params.fr_x_min,
                    c.imag / static_cast<double>(params.scr_height) * (params.fr_y_max - params.fr_y_min) + params.fr_y_min);
    }

    // Check if a point is in the set or escapes to infinity, return the number of iterations
    __forceinline__ __device__ int escape(Complex c,fractal_func_t func) {
        Complex z(0);
        int iter = 0;

        while (z.magnitude() < 2.0 && iter < iter_max) {
            z = func(z, c);
            iter++;
        }

        return iter;
    }

    // Loop over each pixel from our image and check if the points associated with this pixel escape to infinity
    __global__ void get_number_iterations(FractalParams params, int *colors, int func_idx) {

        int tix = blockDim.x * blockIdx.x + threadIdx.x;
        int tiy = blockDim.y * blockIdx.y + threadIdx.y;

        if (tiy < params.scr_height && tix < params.scr_width) {
            Complex c(static_cast<double>(tix), static_cast<double>(tiy));
            c = scale(params, c);

            int idx = tiy * params.scr_width + tix;
            fractal_func_t func = fractal_functions[func_idx];
            colors[idx] = escape(c, func);
        }
    }

void fractal(window<int> &scr, window<double> &fract, std::vector<int> &colors,
             int func_idx, const char *fname, bool smooth_color) {
    
    FractalParams params(scr,fract);

    auto start = std::chrono::steady_clock::now();
    int *d_colors;
    CUDA_CHECK(cudaMalloc(&d_colors, colors.size() * sizeof(int)));

    dim3 threadsPerBlock(32, 32);	
    dim3 numBlocks((params.scr_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (params.scr_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        #if ENABLE_STREAMS
            //TODO
            int N = 8; //number of streams
            cudaStream_t streams[N];
            dim3 numBlocksPerStream(numBlocks.x, numBlocks.y / N);

            for (int i = 0; i < N; i++)     
                CUDA_CHECK(cudaStreamCreate(&streams[i]));

            for(int i=0;i<N;i++){   
                size_t offx = 0;
                size_t offy = (scr_height/N)*i;
                get_number_iterations<<<numBlocksPerStream,threadsPerBlock,0,streams[i]>>>(params, d_colors+(offy*scr_width+offx),func_idx);
                #if ENABLE_CUDA_CHECK   
                    CUDA_CHECK(cudaGetLastError());         
                #endif
                CUDA_CHECK(cudaMemcpyAsync(colors.data()+ (offy*scr_width+offx),d_colors+(offy*scr_width+offx), ((scr_height*scr_width/(N*N))*sizeof(int)), cudaMemcpyDeviceToHost,streams[i%N]));
            }

            for (int i = 0; i < N; i++) {
                CUDA_CHECK(cudaStreamSynchronize(streams[i]));
                CUDA_CHECK(cudaStreamDestroy(streams[i]));
            }
        #else

            get_number_iterations<<<numBlocks, threadsPerBlock>>>(params, d_colors, func_idx);
            #if ENABLE_CUDA_CHECK   
                CUDA_CHECK(cudaGetLastError());         
            #else
            #endif

            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(colors.data(), d_colors, colors.size() * sizeof(int), cudaMemcpyDeviceToHost));
        #endif
    auto end = std::chrono::steady_clock::now();

    std::cout << "Time to generate " << fname << " = " << std::chrono::duration<double, std::milli>(end - start).count() << " [ms]" << std::endl;
    
    // Save (show) the result as an image
    plot(scr, colors, iter_max, fname, smooth_color);

    // Output details to a text file
    std::ofstream out("./res/fractal_details_GPU.txt", std::ios::app);
    if (out.is_open()) {
        out << "File name: " << fname << "\n";
        out << "GPU Accelerated: true" << "\n";
        out << "Streams: " << (ENABLE_STREAMS ? "enabled" : "disabled") << "\n";
        out << "Debug Mode: " << (ENABLE_CUDA_CHECK ? "enabled" : "disabled") << "\n";
        out << "Time to generate: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
        out << "Iterations: " << iter_max    << "\n";
        out << "Smooth color: " << (smooth_color ? "true" : "false") << "\n";
        out << "----------------------------------------\n";
        out.close();
    } else {
        std::cerr << "Unable to open file for writing" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_colors));
}

void mandelbrot() {
    // Define the size of the image
    window<int> scr(0, win_width, 0, win_height);
    // The domain in which we test for points
    window<double> fract(-2.2, 1.2, -1.7, 1.7);

    const char *fname = "./res/mandelbrot_acc.png";
    bool smooth_color = true;
    std::vector<int> colors(scr.size());

    fractal(scr, fract, colors, 0, fname, smooth_color);
}

void triple_mandelbrot() {
    // Define the size of the image
    window<int> scr(0, win_width, 0, win_height);
    // The domain in which we test for points
    window<double> fract(-1.5, 1.5, -1.5, 1.5);

    const char *fname = "./res/triple_mandelbrot_acc.png";
    bool smooth_color = true;
    std::vector<int> colors(scr.size());

    fractal(scr, fract, colors, 1, fname, smooth_color);
}
void prewarm_gpu(size_t n=1){
    int* dummy;
    cudaMalloc(&dummy,n*sizeof(int));
    cudaFree(dummy);
    
}
int main() {
    // getDeviceProps();
    prewarm_gpu(win_width * win_height);
    mandelbrot();
    triple_mandelbrot();
}