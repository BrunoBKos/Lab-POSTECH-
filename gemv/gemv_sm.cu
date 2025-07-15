
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define Elems 2048
#define NTHREADS 1024

// headers
void gemv_sec(float* weight, float* input, float* output, size_t n);

__global__ void gemv_par(float* weight, float* input, float* output, size_t n);

int compare(float* a, float* b, size_t n);

void initialize(float* a, size_t n);

// main function
int main(void) {

  // host vectors
  float* h_weight;
  float* h_input;
  float* h_output_sec;
  float* h_output_par;

  // device vectors
  float* d_weight;
  float* d_input;
  float* d_output;

  // variables
  int num_blks;

  // mempory reserve in CPU
  h_weight = (float*) malloc(Elems*Elems*sizeof(float));
  h_input = (float*) malloc(Elems*sizeof(float));
  h_output_sec = (float*) malloc(Elems*sizeof(float));
  h_output_par = (float*) malloc(Elems*sizeof(float));

  // memory reserve in GPU
  cudaMalloc(&d_weight, Elems*Elems*sizeof(float));
  cudaMalloc(&d_input, Elems*sizeof(float));
  cudaMalloc(&d_output, Elems*sizeof(float));

  // initialization
  initialize(h_weight, Elems*Elems);
  initialize(h_input, Elems);

  //device copy
  cudaMemcpy(d_weight, h_weight, Elems*Elems*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, h_input, Elems*sizeof(float), cudaMemcpyHostToDevice);

  // call to kernel (to make the kernel still)
  num_blks = Elems/NTHREADS;
  gemv_par<<<num_blks, NTHREADS>>>(d_weight, d_input, d_output, Elems);

  // secuential calculation of the results
  gemv_sec(h_weight, h_input, h_output_sec, Elems);

  // recovery of the parallel results
  cudaMemcpy(h_output_par, d_output, Elems*sizeof(float), cudaMemcpyDeviceToHost);

  // return of the results
  int ret = compare(h_output_sec, h_output_par, Elems); 
  if(ret) {
    printf("Error on the execution: (diferent results after index: %d)\n", ret);
    printf("Expected:");
    for(int i = 0; i < 6; i++) printf(" %f", h_output_sec[i+ret]);
    printf("\n");
    printf("Given:   ");
    for(int i = 0; i < 6; i++) printf(" %f", h_output_par[i+ret]);
    printf("\n");
    return 1;
  }
  else {
    printf("Correct execution: Success\n");
    return 0;
  }
}

// CPU function for the secuential product between matrix and vector
void gemv_sec(float* weight, float* input, float* output, size_t n) {
  int i, j;
  float acum = 0;
  for(i = 0; i < n; i++) {
    acum = 0;
    for(j = 0; j < n; j++) {
        acum += (weight[(i*n) + j] * input[j]);
    }
    output[i] = acum;
  }
}

// GPU kenel (vesion with sm)
__global__ void gemv_par(float* weight, float* input, float* output, size_t n) {
    __shared__ int s_input[NTHREADS];
    int i, j;
    int th_id = threadIdx.x + blockDim.x*blockIdx.x; 
    float acum = 0;
    for(i = 0; i < (n/blockDim.x); i++) {
        s_input[threadIdx.x] = input[threadIdx.x + (i*blockDim.x)];
        __syncthreads(); 
        for(j = 0; j < NTHREADS; j++) {
          acum += (weight[(th_id*n) + (i*blockDim.x) + j]*s_input[j]);
        }
        __syncthreads();
    }
    output[th_id] = acum;
}

// auxiliar function to compare the values of two diferent vectors (to do: use memcmp)
int compare(float* a, float* b, size_t n) {
  int i;
  for(i = 0; i < n; i++)
    if(a[i] != b[i]) break;
  return (i - n) ? i : 0;
}

// auxiliar function to initialize vectors with random values
void initialize(float* vect, size_t n){
    int i;
    for (i = 0; i < n; i++) {
	    vect[i] = (float) (rand() % 1000);
    }
}

