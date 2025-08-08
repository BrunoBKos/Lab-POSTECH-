#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "spcsr_generator.h"

#define ROWS 128
#define NTHREADS 1024

// headers
void spmv_sec(csr_spm_t* csr_spm, float* input, float* output);

__global__ void spmv_par(int* rows, int num_rows, float* data, int* cols, float* input, float* output);

int compare(float* a, float* b, size_t n);

void initialize(float* a, size_t n);

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
  int num_blks = ROWS/NTHREADS;

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
  initialize(h_input_vector, ROWS);

  //device copy
  cudaMemcpy(d_rows, (*csr_spm).rows, ROWS*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, (*csr_spm).cols, (((*csr_spm).rows)[ROWS])*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cols, (*csr_spm).data, (((*csr_spm).rows)[ROWS])*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_vector, h_input_vector, ROWS*sizeof(float), cudaMemcpyHostToDevice);

  // call to GPU kernel
  spmv_par<<<num_blks, NTHREADS>>>(d_rows, ROWS, d_data, d_cols, d_input_vector, d_output_vector);

  // secuential calculation of the results
  spmv_sec(csr_spm, h_input_vector, h_output_vector_sec);

  // recovery of the parallel results
  cudaMemcpy(h_output_vector_par, d_output_vector, ROWS*sizeof(float), cudaMemcpyDeviceToHost);

  // return of the results
  float rel_err = 0;
  for(int i = 0; i < ROWS; i++) {
      rel_err += abs((h_output_vector_par[i]-h_output_vector_sec[i])/h_output_vector_sec[i]);
  }
  rel_err /= ROWS;
  if(rel_err > 0.15) printf("relative error greater than 0.15; relative error: %f\n", rel_err); 
  
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

void spmv_sec(csr_spm_t* csr_spm, float* input, float* output) {

  int num_rows = (*csr_spm).num_rows;
  int* rows = (*csr_spm).rows;
  int* cols = (*csr_spm).cols;
  float* data = (*csr_spm).data;

  for(int i = 0; i < num_rows; i++) {
    float acum = 0;
    for(int j = rows[i]; j < rows[i+1]; j++) {
      acum += data[j]*(input[(cols[j])]);
    }
    output[i] = acum;
  }

}

///////////////////////////////////////////////////////////
// GPU Kernels of the spmv algorithym (parallel version) //
///////////////////////////////////////////////////////////

__global__ void spmv_par(int* rows, int num_rows, float* data, int* cols, float* input, float* output) {

  int th_id = threadIdx.x + blockDim.x*blockIdx.x;
  if(th_id < num_rows) {
    float acum = 0;
    for(int i = rows[th_id]; i < rows[th_id+1]; i++) {
      acum += data[i]*(input[(cols[i])]);
    }
    output[th_id] = acum;
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

