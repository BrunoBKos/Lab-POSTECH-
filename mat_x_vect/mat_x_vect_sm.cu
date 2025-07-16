#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define Elems 2048
#define NTHREADS 1024

// headers
void mat_x_vect_sec(int* mat, int* vect, int* out, size_t n);

__global__ void mat_x_vect_par(int* mat, int* vect, int* out, size_t n);

int compare(int* a, int* b, size_t n);

void initialize(int* a, size_t n);

// main function
int main(void) {

  // host vectors
  int* h_mat;
  int* h_vect;
  int* h_out_sec;
  int* h_out_par;

  // device vectors
  int* d_mat;
  int* d_vect;
  int* d_out;

  // variables
  int num_blks;

  // mempory reserve in CPU
  h_mat = (int*) malloc(Elems*Elems*sizeof(int));
  h_vect = (int*) malloc(Elems*sizeof(int));
  h_out_sec = (int*) malloc(Elems*sizeof(int));
  h_out_par = (int*) malloc(Elems*sizeof(int));

  // memory reserve in GPU
  cudaMalloc(&d_mat, Elems*Elems*sizeof(int));
  cudaMalloc(&d_vect, Elems*sizeof(int));
  cudaMalloc(&d_out, Elems*sizeof(int));

  // initialization
  initialize(h_mat, Elems*Elems);
  initialize(h_vect, Elems);

  //device copy
  cudaMemcpy(d_mat, h_mat, Elems*Elems*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vect, h_vect, Elems*sizeof(int), cudaMemcpyHostToDevice);

  // call to kernel (to make the kernel still)
  num_blks = Elems/NTHREADS;
  mat_x_vect_par<<<num_blks, NTHREADS>>>(d_mat, d_vect, d_out, Elems);

  // secuential calculation of the results
  mat_x_vect_sec(h_mat, h_vect, h_out_sec, Elems);

  // recovery of the parallel results
  cudaMemcpy(h_out_par, d_out, Elems*sizeof(int), cudaMemcpyDeviceToHost);

  // return of the results
  if(compare(h_out_sec, h_out_par, Elems)) {
    printf("Error on the execution: (diferent results)\n");
    return 1;
  }
  else {
    printf("Correct execution: Success\n");
    return 0;
  }
}

// CPU function for the secuential product between matrix and vector
void mat_x_vect_sec(int* mat, int* vect, int* out, size_t n) {
  int i, j, acum;
  for(i = 0; i < n; i++) {
    acum = 0;
    for(j = 0; j < n; j++) {
        acum += (mat[(i*n) + j] * vect[j]);
    }
    out[i] = acum;
  }
}

// GPU kenel (vesion with sm)
__global__ void mat_x_vect_par(int* mat, int* vec, int* out, size_t n) {
    __shared__ int s_vect[NTHREADS];
    int i, j;
    int acum = 0;
    int th_id = threadIdx.x + blockDim.x*blockIdx.x; 
    for(i = 0; i < (n/blockDim.x); i++) {
        // initialize shared memory
        s_vect[threadIdx.x] = vec[threadIdx.x + (i*blockDim.x)];
        __syncthreads(); 
        // operate
        for(j = 0; j < NTHREADS; j++) {
          acum += (mat[(th_id*n) + (i*blockDim.x) + j]*s_vect[j]);
        }
        __syncthreads();
    }
    // store results
    out[th_id] = acum;
}

// auxiliar function to compare the values of two diferent vectors (to do: use memcmp)
int compare(int* a, int* b, size_t n) {
  int i;
  for(i = 0; i < n; i++)
    if(a[i] != b[i]) break;
  return (i - n);
}

// auxiliar function to initialize vectors with random values
void initialize(int* vect, size_t n){
    int i;
    for (i = 0; i < n; i++) {
	    vect[i] = rand() % 1000;
    }
}

