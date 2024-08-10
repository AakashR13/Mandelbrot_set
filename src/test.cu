#include "gpu_utils.cuh"

__global__ void test_fma(double real, double imag){
    double z_real = 0, z_imag = 0;
    double z_real2 = 0, z_imag2 = 0;
    double z_real_temp = 0, z_imag_temp = 0; 
    double z_real_fma = 0, z_imag_fma = 0;

    for(int i = 0; i < 10; i++){
        // with fma
        z_real_temp = fma(-z_imag, z_imag, real); // z_real part
        z_real_temp = fma(z_real, z_real, z_real_temp);
        z_imag_temp = fma(2 * z_real, z_imag, imag); // z_imag part
        
        z_imag_fma = z_imag_temp;
        z_real_fma = z_real_temp;

        // without fma
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2 * z_real * z_imag + imag;
        z_real = z_real2 - z_imag2 + real;


        printf("Iteration %d:\tfma: (%lf, %lf)\tnon-fma: (%lf, %lf)\n", 
                i, z_real_fma, z_imag_fma, z_real, z_imag);
    }
}

int main(){
    double real = -0.743643887037151;
    double imag = 0.131825904205330;
    test_fma<<<1,1>>>(real,imag);
    cudaDeviceSynchronize();
}