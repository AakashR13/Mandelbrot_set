#include <vector>
#include <chrono>
#include <functional>
#include <cmath>
#include <cuda_runtime.h>

#include "gpu_utils.cuh"
#include "window.h"
#include "save_image.h"
#include "params.h"


// Kernel Params
struct FractalParams {
    int scr_width;
    int scr_height;
    double fr_x_max;
    double fr_x_min;
    double fr_y_max;
    double fr_y_min;
    // double x_scale;
    // double y_scale;
    __host__ __device__ FractalParams() {}

    FractalParams(const window<int>& scr, const window<double>& fract)
    :   scr_width(scr.width()), scr_height(scr.height()),
        fr_x_max(fract.x_max()), fr_x_min(fract.x_min()),
        fr_y_max(fract.y_max()), fr_y_min(fract.y_min())
        // x_scale(static_cast<double>(scr.width()) * (fract.x_max()-fract.x_min())),
        // y_scale(static_cast<double>(scr.height()) * (fract.y_max()-fract.y_min())) 
        {}

    // void print() const {
    //     printf("scr_width: %d\n", scr_width);
    //     printf("scr_height: %d\n", scr_height);
    //     printf("fr_x_max: %.2f\n", fr_x_max);
    //     printf("fr_x_min: %.2f\n", fr_x_min);
    //     printf("fr_y_max: %.2f\n", fr_y_max);
    //     printf("fr_y_min: %.2f\n", fr_y_min);
    //     printf("x_scale: %.2f\n", x_scale);
    //     printf("y_scale: %.2f\n", y_scale);
    // }
};

    __constant__ FractalParams d_params;

    __global__ void just_basic_mandelbrot(int* colors, int offset=0) {
        int tix = blockDim.x * blockIdx.x + threadIdx.x + offset;
        double x = tix % win_width;
        double y = tix / win_width;

        if(tix < d_params.scr_height * d_params.scr_width){
            double c_real = (x) / static_cast<double>(d_params.scr_width) * (d_params.fr_x_max - d_params.fr_x_min) + d_params.fr_x_min;
            double c_imag = (y) / static_cast<double>(d_params.scr_height) * (d_params.fr_y_max - d_params.fr_y_min) + d_params.fr_y_min;

            double z_real=0,z_imag=0;
            double z_real2=0,z_imag2=0;
            // double z_real_temp=0, z_imag_temp=0; 
            int iter = 0;

            while(iter < iter_max && z_imag*z_imag + z_real*z_real < 4){
                z_real2 = z_real*z_real;
                z_imag2 = z_imag*z_imag;
                z_imag = 2 * z_real * z_imag + c_imag;
                z_real = z_real2 - z_imag2 + c_real;
                // z_imag_temp = fma(2*z_real2,z_imag,c_imag);
                // z_real_temp = fma(z_real,z_real,fma(-z_imag,z_imag,c_real));
                // z_imag = z_imag_temp;
                // z_real = z_real_temp;
                iter++;
            }
            colors[tix-offset] = iter;
        }
    }


        __global__ void just_triple_mandelbrot(int* colors, int offset=0) {
        int tix = blockDim.x * blockIdx.x + threadIdx.x + offset;
        double x = tix % win_width;
        double y = tix / win_width;

        if(tix < d_params.scr_height * d_params.scr_width){
            double c_real = (x) / static_cast<double>(d_params.scr_width) * (d_params.fr_x_max - d_params.fr_x_min) + d_params.fr_x_min;
            double c_imag = (y) / static_cast<double>(d_params.scr_height) * (d_params.fr_y_max - d_params.fr_y_min) + d_params.fr_y_min;

            double z_real=0,z_imag=0;
            double z_real2=0,z_imag2=0;
            double z_real_temp=0, z_imag_temp=0;
            int iter = 0;

            while(iter < iter_max && z_imag*z_imag + z_real*z_real < 4 ){
                z_real2 = z_real * z_real;
                z_imag2 = z_imag * z_imag;
                z_real_temp = z_real * (z_real2 - 3 * z_imag2) + c_real;
                z_imag_temp = z_imag * (3 * z_real2 - z_imag2) + c_imag;
                z_real = z_real_temp;
                z_imag = z_imag_temp;
                iter++;
            }
            colors[tix-offset] = iter;
        }
    }

void fractal(window<int> &scr, window<double> &fract, std::vector<int> &colors,
             int func_idx, const char *fname, bool smooth_color) {
    
    FractalParams h_params(scr,fract);
    // h_params.print();
    // printf("GPU x_scale: %.2f\n",static_cast<double>(h_params.scr_width) * (h_params.fr_x_max - h_params.fr_x_min) + h_params.fr_x_min);
    // printf("GPU y_scale: %.2f\n",static_cast<double>(h_params.scr_height) * (h_params.fr_y_max - h_params.fr_y_min) + h_params.fr_y_min);

    CUDA_CHECK(cudaMemcpyToSymbol(d_params,&h_params,sizeof(FractalParams)));

    auto start = std::chrono::steady_clock::now();
    int *d_colors;
    CUDA_CHECK(cudaMalloc(&d_colors, colors.size() * sizeof(int)));

    //
    //
    dim3 threadsPerBlock(256);
    dim3 numBlocks((h_params.scr_width*h_params.scr_height+threadsPerBlock.x-1)/threadsPerBlock.x);
    //
    //

        #if ENABLE_STREAMS
            //TODO
            int N = 3; //number of streams
            cudaStream_t streams[N];
            dim3 numBlocksPerStream(numBlocks.x/N);

            for (int i = 0; i < N; i++)     
                CUDA_CHECK(cudaStreamCreate(&streams[i]));

            int step = ((params.scr_width*params.scr_height+N-1)/N);

            for(int i=0;i<N;i++){   
                size_t offx = step * i;
                get_number_iterations<<<numBlocksPerStream,threadsPerBlock,0,streams[i%N]>>>(params, d_colors+(offx),func_idx,offx);

                #if ENABLE_CUDA_CHECK   
                    CUDA_CHECK(cudaGetLastError());         
                #endif

                CUDA_CHECK(cudaMemcpyAsync(colors.data()+offx,d_colors+offx, (step*sizeof(int)), cudaMemcpyDeviceToHost,streams[i%N]));
            }

            for (int i = 0; i < N; i++) {
                CUDA_CHECK(cudaStreamSynchronize(streams[i]));
                CUDA_CHECK(cudaStreamDestroy(streams[i]));
            }
        #else
            if(func_idx == 0)
                just_basic_mandelbrot<<<numBlocks,threadsPerBlock>>>(d_colors);
            if(func_idx == 1)
                just_triple_mandelbrot<<<numBlocks,threadsPerBlock>>>(d_colors);
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
    std::ofstream out("./reports/fractal_details_GPU.txt", std::ios::app);
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
    if(getDeviceProps("./reports/device_properties.txt")) return 1;
    prewarm_gpu(win_width * win_height);
    mandelbrot();
    triple_mandelbrot();
}