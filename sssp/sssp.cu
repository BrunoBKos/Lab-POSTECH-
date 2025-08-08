#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "spcsr_generator.h"

#define ROWS 128
#define BIGNUM 1048576.0
#define NTHREADS 1024

// headers
void sssp_sec(csr_spm_t* csr_spm, float* input, float* output);

__global__ void sssp_par(int* rows, int num_rows, float* data, int* cols, float* input, float* output);

int compare(float* a, float* b, size_t n);

void initialize(float* a, size_t n);

void init_input_vector(csr_spm_t* csr_spm, float* vector);

// main function
int main(void) {

  // host vectors
  csr_spm_t* csr_spm;
  float* h_input_vector;
  float* h_output_vector_sec;
  float* h_output_vector_par; 

  // device vectors
  int* d_rows;
  int* d_cols;
  float* d_data;
  float* d_input_vector;
  float* d_output_vector;

  // variables
  int num_blks = (ROWS+NTHREADS-1)/NTHREADS;

  // mempory reserve in CPU
  csr_spm             = spmcsr_gen(ROWS);
  h_input_vector      = (float*) malloc(ROWS*sizeof(float));
  h_output_vector_sec = (float*) malloc(ROWS*sizeof(float));
  h_output_vector_par = (float*) malloc(ROWS*sizeof(float));

  // memory reserve in GPU
  cudaMalloc(&d_rows, (ROWS+1)*sizeof(int));
  cudaMalloc(&d_cols, (((*csr_spm).rows)[ROWS])*sizeof(int));
  cudaMalloc(&d_data, (((*csr_spm).rows)[ROWS])*sizeof(float));
  cudaMalloc(&d_input_vector, ROWS*sizeof(float));
  cudaMalloc(&d_output_vector, ROWS*sizeof(float));

  // initialitation
  init_input_vector(csr_spm, h_input_vector);

  //device copy
  cudaMemcpy(d_rows, (*csr_spm).rows, ROWS*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cols, (*csr_spm).cols, (((*csr_spm).rows)[ROWS])*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, (*csr_spm).data, (((*csr_spm).rows)[ROWS])*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_vector, h_input_vector, ROWS*sizeof(float), cudaMemcpyHostToDevice);

  // call to GPU kernel
  sssp_par<<<num_blks, NTHREADS>>>(d_rows, ROWS, d_data, d_cols, d_input_vector, d_output_vector);

  // secuential calculation of the results
  sssp_sec(csr_spm, h_input_vector, h_output_vector_sec);

  // recovery of the parallel results
  cudaMemcpy(h_output_vector_par, d_output_vector, ROWS*sizeof(float), cudaMemcpyDeviceToHost);

  // return of the results
  float rel_err = 0;
  for(int i = 0; i < ROWS; i++) {
      rel_err += abs((h_output_vector_par[i]-h_output_vector_sec[i])/h_output_vector_sec[i]);
  }
  rel_err /= ROWS;
  if(rel_err > 0.15) printf("relative error greater than 0.15; relative error: %f\n", rel_err); 
  
  for(int i = 0; i < ROWS; i++)
    printf(" %f,", h_output_vector_sec[i]);
  printf("\n");
  for(int i = 0; i < ROWS; i++)
    printf(" %f,", h_output_vector_par[i]);
  printf("\n");

  // resources release (host)
  free_csr_spm(csr_spm);
  if(h_input_vector) free(h_input_vector);
  if(h_output_vector_par) free(h_output_vector_par);
  if(h_output_vector_sec) free(h_output_vector_sec);
  
  // resources release (device)
  cudaFree(d_rows);
  cudaFree(d_cols);
  cudaFree(d_data);
  cudaFree(d_input_vector);
  cudaFree(d_output_vector);

  return 0;
}

////////////////////////////////////////////////////
// CPU (secuential) version of the spmv algorithm //
////////////////////////////////////////////////////

void sssp_sec(csr_spm_t* csr_spm, float* input, float* output) {

  int num_rows = (*csr_spm).num_rows;
  int* rows = (*csr_spm).rows;
  int* cols = (*csr_spm).cols;
  float* data = (*csr_spm).data;

  for(int i = 0; i < num_rows; i++) {
    float min = input[i];
    for(int j = rows[i]; j < rows[i+1]; j++) {
      float min_update = (data[j] + input[(cols[j])]);
      if(min_update < min) min = min_update;
    }
    output[i] = min;
  }

}

///////////////////////////////////////////////////////////
// GPU Kernels of the spmv algorithym (parallel version) //
///////////////////////////////////////////////////////////

__global__ void sssp_par(int* rows, int num_rows, float* data, int* cols, float* input, float* output) {

  int th_id = threadIdx.x + blockDim.x*blockIdx.x;
  if(th_id < num_rows) {
    float min = input[th_id];
    for(int i = rows[th_id]; i < rows[th_id+1]; i++) {
      float min_update = (data[i] + input[(cols[i])]);
      if(min_update < min) min = min_update;
    }
    output[th_id] = min;
  }
}

////////////////////////
// auxiliar functions //
////////////////////////

// auxiliar function to compare the values of two diferent vectors (to do: use memcmp)
int compare(float* a, float* b, size_t n) {
  int i;
  for(i = 0; i < n; i++)
    if(a[i] != b[i]) break;
  return (i - n) ? i : 0;
}

// auxiliar function to initialize vector with random positive values less than BINS
void initialize(float* vect, size_t n){ for(int i = 0; i < n; i++) vect[i] = ((float) (rand() % 1024)); }



void init_input_vector(csr_spm_t* csr_spm, float* vector) {

  int num_rows = (*csr_spm).num_rows;
  int* rows = (*csr_spm).rows;
  int* cols = (*csr_spm).cols;
  float* data = (*csr_spm).data;
  
  for(int i = 0; i < num_rows; i++) {
    vector[i] = BIGNUM;
  }
  
  for(int i = 0; i < rows[1]; i++) {
    vector[(cols[i])] = data[i];
  }

}