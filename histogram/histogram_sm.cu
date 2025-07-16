#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define BINS 256
#define Elems 32768
#define NTHREADS 1024

// headers
void histogram_sec(int* input_data, size_t n, int* output_bin);

__global__ void histogram_par(int* input_data, size_t n, int* output_bin);

int compare(int* a, int* b, size_t n);

void initialize(int* a, size_t n);

// main function
int main(void) {

  // host vectors
  int* h_data;
  int* h_out_bin_sec;
  int* h_out_bin_par;

  // device vectors
  int* d_data;
  int* d_out_bin;

  // variables
  int num_blks;

  // mempory reserve in CPU
  h_data = (int*) malloc(Elems*sizeof(int));
  h_out_bin_sec = (int*) malloc(BINS*sizeof(int));
  h_out_bin_par = (int*) malloc(BINS*sizeof(int));

  // memory reserve in GPU
  cudaMalloc(&d_data, Elems*sizeof(int));
  cudaMalloc(&d_out_bin, BINS*sizeof(int));

  // initialization
  initialize(h_data, Elems);
  memset(h_out_bin_sec, 0, BINS*sizeof(int));
  memset(h_out_bin_par, 0, BINS*sizeof(int));

  //device copy
  cudaMemcpy(d_data, h_data, Elems*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_bin, h_out_bin_par, BINS*sizeof(int), cudaMemcpyHostToDevice);

  // call to kernel (to make the kernel still)
  num_blks = Elems/NTHREADS;
  histogram_par<<<num_blks, NTHREADS>>>(d_data, Elems, d_out_bin);

  // secuential calculation of the results
  histogram_sec(h_data, Elems, h_out_bin_sec);

  // recovery of the parallel results
  cudaMemcpy(h_out_bin_par, d_out_bin, BINS*sizeof(int), cudaMemcpyDeviceToHost);

  // return of the results
  int ret = compare(h_out_bin_sec, h_out_bin_par, BINS); 
  if(ret) {
    printf("Error on the execution: (diferent results after index: %d)\n", ret);
    printf("Expected:");
    for(int i = 0; i < 6; i++) printf(" %d", h_out_bin_sec[i+ret]);
    printf("\n");
    printf("Given:   ");
    for(int i = 0; i < 6; i++) printf(" %d", h_out_bin_par[i+ret]);
    printf("\n");
    return 1;
  }
  else {
    printf("Correct execution: Success\n");
    return 0;
  }
}

// CPU function (Histogram)
void histogram_sec(int* input_data, size_t N, int* output_bin) { for(int i = 0; i < N; i++) output_bin[(input_data[i])]++; }

// GPU kenel (vesion with sm)
__global__ void histogram_par(int* input_data, size_t N, int* output_bin) {
  __shared__ int s_bin[BINS];
  int th_id = threadIdx.x + blockDim.x*blockIdx.x; 
  if(th_id < N) atomicAdd(s_bin + input_data[th_id], 1);
  __syncthreads();
  if(threadIdx.x < BINS) atomicAdd(output_bin + threadIdx.x, input_data[threadIdx.x]);
}

// auxiliar function to compare the values of two diferent vectors (to do: use memcmp)
int compare(int* a, int* b, size_t n) {
  int i;
  for(i = 0; i < n; i++)
    if(a[i] != b[i]) break;
  return (i - n) ? i : 0;
}

// auxiliar function to initialize vectors with random values
void initialize(int* vect, size_t n){ for(int i = 0; i < n; i++) vect[i] = ((unsigned int) rand()) % BINS; }

