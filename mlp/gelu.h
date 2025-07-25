#ifndef NTHREADS
#define NTHREADS 1024
#endif

#include <sys/types.h>

int gelu(float* input_vector, size_t N, float* output_vector);