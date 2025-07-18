
#include "bias.h"

__global__ void bias_par(float* input_vector, size_t N, float* bias_vector, float* output_vector);

int bias(float* input_vector, size_t N, float* bias_vector, float* output_vector) {

    // device variables
    float* d_input_vector; 
    float* d_bias_vector;
    float* d_output_vector;

    int num_blks = N / NTHREADS;

    // device memory reserve
    cudaMalloc(&d_input_vector, N*sizeof(float));
    cudaMalloc(&d_bias_vector, N*sizeof(float));
    cudaMalloc(&d_output_vector, N*sizeof(float));

    // device memory initialitation
    cudaMemcpy(d_input_vector, input_vector, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_vector, bias_vector, N*sizeof(float), cudaMemcpyHostToDevice);

    // kernel call
    bias_par<<<num_blks, NTHREADS>>>(d_input_vector, N, d_bias_vector, d_output_vector);
    
    // results recovery
    cudaMemcpy(output_vector, d_output_vector, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input_vector); 
    cudaFree(d_bias_vector);
    cudaFree(d_output_vector);

    return 0;

}

// GPU Kernel for the bias addition
__global__ void bias_par(float* input_vector, size_t N, float* bias_vector, float* output_vector) {

    int th_id = threadIdx.x + blockDim.x * blockIdx.x; 
    output_vector[th_id] = input_vector[th_id] + bias_vector[th_id];

} 