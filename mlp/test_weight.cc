#include "weight.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define INPUTSIZE 1024
#define OUTPUTSIZE 1024

//////////
// headers
//////////

void error_msg(int error);

void initialize_vector_float(float* vector, size_t vector_size);

int init(float** weight, float** input_vector, float** output_vector_par, 
            float** output_vector_sec, size_t input_size, size_t output_size);

////////////////
// main function
////////////////

int main(int argc, char** argv) {

    int res = 0;
    int sm = 0;

    float* weight = NULL;
    float* input_vector = NULL;
    float* output_vector_par = NULL;
    float* output_vector_sec = NULL; 

    if(argc > 1) sm = atoi(argv[1]);

    res = init(&weight, &input_vector, &output_vector_par, &output_vector_sec, INPUTSIZE, OUTPUTSIZE);
    
    if(res) { error_msg(res); return res; } 
    
    weigth(input_vector, INPUTSIZE, weight, output_vector_par, OUTPUTSIZE, sm);

    float rel_err = 0;
    for(int i = 0; i < OUTPUTSIZE; i++) {
        rel_err += abs((output_vector_par[i]-output_vector_sec[i])/output_vector_sec[i]);
    }
    rel_err /= OUTPUTSIZE;

    if(rel_err > 0.15) res = 5; // error code for memory diferences (problems in parallel execution)

    // resources release
    if(weight) free(weight);
    if(input_vector) free(input_vector);
    if(output_vector_par) free(output_vector_par);
    if(output_vector_sec) free(output_vector_sec);
    
    // end of execution msg
    error_msg(res);
    if(!(res - 5)) printf("relative error calculated = %f", rel_err);

    return res;

}

/////////////////////
// auxiliar functions
/////////////////////

int init(float** weight, float** input_vector, float** output_vector_par, 
            float** output_vector_sec, size_t input_size, size_t output_size) {
    
    int res = 0;
    
    // resources reserve
    *weight = (float*) malloc(input_size*output_size*sizeof(float));
    res += ((*weight) == NULL); 
    *input_vector = (float*) malloc(input_size*sizeof(float));
    res += ((*input_vector) == NULL); 
    *output_vector_par = (float*) malloc(output_size*sizeof(float));
    res += ((*output_vector_par) == NULL); 
    *output_vector_sec = (float*) malloc(output_size*sizeof(float));
    res += ((*output_vector_sec) == NULL); 

    if(res) {
        if(*weight) free(*weight);
        if(*input_vector) free(*input_vector);
        if(*output_vector_par) free(*output_vector_par);
        if(*output_vector_sec) free(*output_vector_sec);
        return 3; // error code for memory problems
    }
    
    // vector initialization
    initialize_vector_float(*weight, input_size*output_size);
    initialize_vector_float(*input_vector, input_size);
    memset(*output_vector_par, '\0', output_size*sizeof(float));

    // output calculation (secuential version of gemv)
    for(int i = 0; i < output_size; i++) { 
        float acum = 0;
        for(int j = 0; j < input_size; j++) {
            acum += ((*input_vector)[j])*((*weight)[j+(input_size*i)]);
        }
        (*output_vector_sec)[i] = acum;
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
            printf("Execution Error: Significant fiferences found between the secuetial and parallel resoults\n");
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