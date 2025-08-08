#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define Elems 32768
#define NTHREADS 1024

// headers
void softmax_sec(float* input_vector, size_t n, float* output_vector);

__global__ void softmax_par1(float* input_vector, size_t N, float* maxred_vector);

__global__ void softmax_par2(float* input_vector, size_t N, float* output_vector, float maxred, float* sum_red);

__global__ void softmax_par3(float* exps_vector, size_t N, float sumred);

int compare(float* a, float* b, size_t n);

void initialize(float* a, size_t n);

// main function
int main(void) {

  // host vectors
  float* h_input_vector;
  float* h_output_vector_sec;
  float* h_output_vector_par; 
  float* h_reduction;

  // device vectors
  float* d_input_vector;
  float* d_output_vector;
  float* d_reduction;

  // variables
  int num_blks = Elems/NTHREADS;

  // mempory reserve in CPU
  h_input_vector      = (float*) malloc(Elems*sizeof(float));
  h_output_vector_sec = (float*) malloc(Elems*sizeof(float));
  h_output_vector_par = (float*) malloc(Elems*sizeof(float));
  h_reduction         = (float*) malloc(num_blks*sizeof(float));

  // memory reserve in GPU
  cudaMalloc(&d_input_vector, Elems*sizeof(float));
  cudaMalloc(&d_output_vector, Elems*sizeof(float));
  cudaMalloc(&d_reduction, num_blks*sizeof(float));

  // initialitation
  initialize(h_input_vector, Elems);

  //device copy
  cudaMemcpy(d_input_vector, h_input_vector, Elems*sizeof(float), cudaMemcpyHostToDevice);

  // call to the firt kernel ( reduction )
  softmax_par1<<<num_blks/2, NTHREADS>>>(d_input_vector, Elems, d_reduction);

  // recovery of the reduction vector
  cudaMemcpy(h_reduction, d_reduction, num_blks*sizeof(float), cudaMemcpyDeviceToHost);

  float maxred = h_reduction[0];
  for(int i = 1; i < (num_blks/2); i++) {
    if(maxred < h_reduction[i]) maxred = h_reduction[i];
  }

  // call to the second kernel
  softmax_par2<<<num_blks, NTHREADS>>>(d_input_vector, Elems, d_output_vector, maxred, d_reduction);
  
  // recovery of the reduction vector
  cudaMemcpy(h_reduction, d_reduction, num_blks*sizeof(float), cudaMemcpyDeviceToHost);

  float sumred = 0;
  for(int i = 0; i < num_blks; i++) {
    sumred += h_reduction[i];
  }

  // call to last kernel
   softmax_par3<<<num_blks, NTHREADS>>>(d_output_vector, Elems, sumred);

  // secuential calculation of the results
  softmax_sec(h_input_vector, Elems, h_output_vector_sec);

  // recovery of the parallel results
  cudaMemcpy(h_output_vector_par, d_output_vector, Elems*sizeof(float), cudaMemcpyDeviceToHost);

  // return of the results
  int ret = compare(h_output_vector_sec, h_output_vector_par, Elems); // TO DO: Mirar la función de comparación // implementarla con memcmp
  if(ret) {
    printf("Error on the execution: (diferent results after index: %d)\n", ret);
    printf("Expected:");
    for(int i = 0; i < 6; i++) printf(" %f", h_output_vector_sec[i+ret]);
    printf("\n");
    printf("Given:   ");
    for(int i = 0; i < 6; i++) printf(" %f", h_output_vector_par[i+ret]);
    printf("\n");
    exit(1);
  }
  else {
    printf("Correct execution: Success\n");
  }

  // free host memory
  free(h_input_vector);
  free(h_output_vector_sec);
  free(h_output_vector_par); 
  free(h_reduction);

  // free device memory
  cudaFree(d_input_vector);
  cudaFree(d_output_vector);
  cudaFree(d_reduction);

  return 0;
}

///////////////////////////////////////////////////////
// CPU (secuential) version of the softmax algorithm //
///////////////////////////////////////////////////////

void softmax_sec(float* input_vector, size_t N, float* output_vector) {

  // max reduction
  float maxred = input_vector[0];
  for(int i = 0; i < N; i++) {
    if(maxred < input_vector[i]) maxred = input_vector[i];
  }

  // exponential calculation + sum reduction
  float sumred = 0;
  for(int i = 0; i < N; i++) {
    float exp_ac = exp(input_vector[i]-maxred);
    output_vector[i] = exp_ac; 
    sumred += exp_ac;
  }

  // division
  for(int i = 0; i < N; i++) {
    output_vector[i] /= sumred;
  }

}

//////////////////////////////////////////////////////////////
// GPU Kernels of the softmax algorithym (parallel version) //
//////////////////////////////////////////////////////////////

// GPU kenel 1 {zmax}
__global__ void softmax_par1(float* input_vector, size_t N, float* maxred_vector) {
  
  int start = threadIdx.x + (blockDim.x*blockIdx.x*2);
  __shared__ float s_maxred_vector[NTHREADS];

  s_maxred_vector[threadIdx.x] = 
      ((input_vector[start] > input_vector[start+NTHREADS]) ? input_vector[start] : input_vector[start+NTHREADS]);
  
  // reduction
  for(int stride = NTHREADS>>1; stride > 0; stride >>= 1) {
    __syncthreads();
    if(threadIdx.x < stride) {
      if(s_maxred_vector[threadIdx.x] < s_maxred_vector[threadIdx.x+stride]) {
        s_maxred_vector[threadIdx.x] = s_maxred_vector[threadIdx.x+stride];
      }
    }
  }
  if(threadIdx.x == 0) maxred_vector[blockIdx.x] = s_maxred_vector[0];

}

// GPU kernel 2 {e^(zi-zmax) ; sum(e^(zi-zmax))}
__global__ void softmax_par2(float* input_vector, size_t N, float* output_vector, float maxred, float* sum_red) {
  
  int th_id = threadIdx.x + blockDim.x*blockIdx.x;
  __shared__ float s_sumred_vector[NTHREADS];

  // exponential
  if(th_id < N)  {
    float exp_ac = exp(input_vector[th_id]-maxred);
    s_sumred_vector[threadIdx.x] = exp_ac;
    output_vector[th_id] = exp_ac; 
  }

  // reduction TO DO: Finalize de algorythm
  for(int stride = NTHREADS>>1; stride > 0; stride >>= 1) {
    __syncthreads();
    if(threadIdx.x < stride) {
      s_sumred_vector[threadIdx.x] += s_sumred_vector[threadIdx.x+stride];
    }
  }
  __syncthreads();
  if(threadIdx.x == 0) sum_red[blockIdx.x] = s_sumred_vector[0];

}


// GPU kernel 3 {e^(zi-zmax) / sum(e^(zi-zmax))}
__global__ void softmax_par3(float* exps_vector, size_t N, float sumred) {
  
  int th_id = threadIdx.x + blockDim.x*blockIdx.x;
  if(th_id < N) exps_vector[th_id] /= sumred;

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
void initialize(float* vect, size_t n){ for(int i = 0; i < n; i++) vect[i] = ((float) (rand() % 32)); }

