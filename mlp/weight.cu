
#include "weight.h"

__global__ void weight_par(float* input_vector, size_t input_size, float* weight, float* output_vector, size_t output_size);

__global__ void weight_par_mp(float* input_vector, size_t input_size, float* weight, float* output_vector, size_t output_size);

__global__ void weight_par_sm(float* input_vector, size_t input_size, float* weight, float* output_vector, size_t output_size);

__global__ void weight_par_sm_mp(float* input_vector, size_t input_size, float* weight, float* output_vector, size_t output_size);

int weigth(float* input_vector, size_t input_size, float* weight, float* output_vector, size_t output_size, int sm) {

    // device variables
    float* d_input_vector; 
    float* d_weight;
    float* d_output_vector;

    int num_blks = output_size / NTHREADS;

    // device memory reserve
    cudaMalloc(&d_input_vector, input_size*sizeof(float));
    cudaMalloc(&d_weight, input_size*output_size*sizeof(float));
    cudaMalloc(&d_output_vector, output_size*sizeof(float));

    // device memory initialitation
    cudaMemcpy(d_input_vector, input_vector, input_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, input_size*output_size*sizeof(float), cudaMemcpyHostToDevice);

    // kernel call
    switch(sm & 3) {
        case 0 : // no shared memory no memory protection
            weight_par<<<num_blks, NTHREADS>>>(d_input_vector, input_size, d_weight, d_output_vector, output_size);
            break;
        case 1 : // shared memory no memory protection
            weight_par_sm<<<num_blks, NTHREADS>>>(d_input_vector, input_size, d_weight, d_output_vector, output_size);
            break;
        case 2 : // no shared memory but memory protection
            weight_par_mp<<<num_blks, NTHREADS>>>(d_input_vector, input_size, d_weight, d_output_vector, output_size);
            break;
        case 3 : // shared memory and memory protection
            weight_par_sm_mp<<<num_blks, NTHREADS>>>(d_input_vector, input_size, d_weight, d_output_vector, output_size);
            break;
    }
    // results recovery
    cudaMemcpy(output_vector, d_output_vector, output_size*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input_vector); 
    cudaFree(d_weight);
    cudaFree(d_output_vector);

    return 0;

}


// GPU Kernel without shared memory and no memory protection
__global__ void weight_par(float* input_vector, size_t input_size, float* weight, float* output_vector, size_t output_size) {
    
    int th_id = threadIdx.x + blockDim.x * blockIdx.x; 
    float acum = 0;
    if(th_id < output_size) {
        for(int i = 0; i < input_size; i++) {
            acum += input_vector[i]*weight[th_id*input_size + i];
        }
        output_vector[th_id] = acum;
    }
}

// GPU Kernel without shared memory but with memory protection
__global__ void weight_par_mp(float* input_vector, size_t input_size, float* weight, float* output_vector, size_t output_size) {
    
    int th_id = threadIdx.x + blockDim.x * blockIdx.x; 
    float acum = 0;
    for(int i = 0; i < input_size; i++) {
        acum += input_vector[i]*weight[th_id*input_size + i];
    }
    output_vector[th_id] = acum;

}

// GPU Kernel with shared memory but no memory protection
__global__ void weight_par_sm(float* input_vector, size_t input_size, float* weight, float* output_vector, size_t output_size) {
    
    __shared__ int s_input[NTHREADS];
    int i, j;
    int th_id = threadIdx.x + blockDim.x * blockIdx.x; 
    int n = (input_size/blockDim.x);
    float acum = 0;
    for(i = 0; i < n; i++) {
        // initialize shared memory
        s_input[threadIdx.x] = input_vector[threadIdx.x + (i*blockDim.x)];
        // thread synchronitation
        __syncthreads(); 
        for(j = 0;  j < blockDim.x; j++) {
            acum += s_input[j]*weight[j + (th_id*input_size)];
        }
        // thread synchronitation
        __syncthreads();
    }
    // results storing
    output_vector[th_id] = acum;

}


// GPU Kernel with shared memory and memory protection
__global__ void weight_par_sm_mp(float* input_vector, size_t input_size, float* weight, float* output_vector, size_t output_size) {
    
    __shared__ int s_input[NTHREADS];
    int i, j;
    int th_id = threadIdx.x + blockDim.x * blockIdx.x; 
    int n = ((input_size/blockDim.x) + (input_size % blockDim.x ? 1 : 0)) ;
    float acum = 0;
    for(i = 0; i < n; i++) {
        // initialize shared memory
        if((threadIdx.x + (i*blockDim.x)) < input_size)
            s_input[threadIdx.x] = input_vector[threadIdx.x + (i*blockDim.x)];
        // thread synchronitation
        __syncthreads(); 
        for(j = 0;  j < blockDim.x; j++) {
            if((j + (i*blockDim.x)) < input_size)
                acum += s_input[j]*weight[j + (th_id*input_size)];
        }
        // thread synchronitation
        __syncthreads();
    }
    // results storing
    if(th_id < output_size)
        output_vector[th_id] = acum;

}
