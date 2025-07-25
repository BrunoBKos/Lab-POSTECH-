#include "relu.h"

__global__ void relu_par(float* input_vector, size_t N, float* output_vector);

int bias(float* input_vector, size_t N, float* output_vector) {

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
    relu_par<<<num_blks, NTHREADS>>>(d_input_vector, N, d_output_vector);
    
    // results recovery
    cudaMemcpy(output_vector, d_output_vector, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input_vector); 
    cudaFree(d_output_vector);

    return 0;

}

// GPU Kernel for the relu activation function
__global__ void relu_par(float* input_vector, size_t N, float* output_vector) {

    int th_id = threadIdx.x + blockDim.x * blockIdx.x; 
    output_vector[th_id] = input_vector[th_id] > 0 ? input_vector[th_id] : 0;

} 