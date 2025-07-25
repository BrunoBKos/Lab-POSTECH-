#ifndef NTRHEADS
#define NTHREADS 1024
#endif

#include <sys/types.h>

/*
    GEMV function using gpu cuda programming

    Use of sm: 
        0 no memory protection and no shared memory
        1 for shared memory but no memory protection
        2 for memory protection but no shared memory
        3 for shared memory and memory protection 
    
    input vector == input vector of size input_size
    weigth == weight matrix of size input_size * output_size
    output vector == output vector (can be the same as input vector) of size output_size

*/
int weigth(float* input_vector, size_t input_size, float* weight, float* output_vector, size_t output_size, int sm);
