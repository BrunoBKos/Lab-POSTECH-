#include "relu.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define OUTPUTSIZE 1024

//////////
// headers
//////////

void error_msg(int error);

void initialize_vector_float(float* vector, size_t vector_size);

int init(float** input_vector, float** output_vector_par, float** output_vector_sec, size_t output_size);

////////////////
// main function
////////////////

int main(int argc, char** argv) {

    int res = 0;

    float* input_vector = NULL;
    float* output_vector_par = NULL;
    float* output_vector_sec = NULL; 

    res = init(&input_vector, &output_vector_par, &output_vector_sec, OUTPUTSIZE);
    
    if(res) { error_msg(res); return res; } 

    bias(input_vector, OUTPUTSIZE, output_vector_par);
    
    float rel_err = 0;
    for(int i = 0; i < OUTPUTSIZE; i++) {
        rel_err += abs((output_vector_par[i]-output_vector_sec[i])/output_vector_sec[i]);
    }
    rel_err /= OUTPUTSIZE;

    if(rel_err > 0.15) res = 5; // error code for memory diferences (problems in parallel execution)

    // resources release
    if(input_vector) free(input_vector);
    if(output_vector_par) free(output_vector_par);
    if(output_vector_sec) free(output_vector_sec);
    
    // end of execution msg
    error_msg(res);
    if(!(res - 5)) printf("relative error calculated = %f\n", rel_err);

    return res;

}

/////////////////////
// auxiliar functions
/////////////////////

int init(float** input_vector, float** output_vector_par, float** output_vector_sec, size_t output_size) {
    
    int res = 0;
    
    // resources reserve
    *input_vector = (float*) malloc(output_size*sizeof(float));
    res += ((*input_vector) == NULL);
    *output_vector_par = (float*) malloc(output_size*sizeof(float));
    res += ((*output_vector_par) == NULL); 
    *output_vector_sec = (float*) malloc(output_size*sizeof(float));
    res += ((*output_vector_sec) == NULL); 

    if(res) {
        if(*input_vector) free(*input_vector);
        if(*output_vector_par) free(*output_vector_par);
        if(*output_vector_sec) free(*output_vector_sec);
        return 3; // error code for memory problems
    }
    
    // vector initialization
    initialize_vector_float(*input_vector, output_size);
    memset(*output_vector_par, '\0', output_size*sizeof(float));

    // output calculation (secuential version of vector add)
    for(int i = 0; i < output_size; i++) { 
        (*output_vector_sec)[i] = ((*input_vector)[i] > 0) ? (*input_vector)[i] : 0; 
    }

    // successful return 
    return 0;
}

void error_msg(int error) {
    switch(error) {
        case 0 :
            printf("Success\n");
            break;
        case 3 :
            printf("Memory Error: Problem encountered while reserving memory space\n");
            break;
        case 5 :
            printf("Execution Error: Significant diferences found between the secuetial and parallel resoults\n");
            break;
        default :
            printf("Unexpected Error\n");
    }
}

void initialize_vector_float(float* vector, size_t vector_size) { 

    for(int i = 0; i < vector_size; i++) { 
        vector[i] = (float) rand();
    }
    
}