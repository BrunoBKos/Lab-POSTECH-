#include "gelu.h"

__global__ void gelu_par(float* input_vector, size_t N, float* output_vector);

int gelu(float* input_vector, size_t N, float* output_vector) {

    // device variables
    float* d_input_vector; 
    float* d_output_vector;

    int num_blks = N / NTHREADS;

    // device memory reserve
    cudaMalloc(&d_input_vector, N*sizeof(float));
    cudaMalloc(&d_output_vector, N*sizeof(float));

    // device memory initialitation
    cudaMemcpy(d_input_vector, input_vector, N*sizeof(float), cudaMemcpyHostToDevice);

    // kernel call
    gelu_par<<<num_blks, NTHREADS>>>(d_input_vector, N, d_output_vector);
    
    // results recovery
    cudaMemcpy(output_vector, d_output_vector, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input_vector); 
    cudaFree(d_output_vector);

    return 0;

}

// GPU Kernel for the relu activation function
__global__ void gelu_par(float* input_vector, size_t N, float* output_vector) {

    int th_id = threadIdx.x + blockDim.x * blockIdx.x; 
    float x = input_vector[th_id];
    float inter = ((((0.044715*x*x) + 1)*x)*0.7978845608);
    output_vector[th_id] = ((tanhf(inter)+1)*0.5*x);

} 