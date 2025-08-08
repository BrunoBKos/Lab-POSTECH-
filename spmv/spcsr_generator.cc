#include "spcsr_generator.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

csr_spm_t* spmcsr_gen(int num_rows) {

    // variables
    csr_spm_t* csr_spm;
    int* rows;
    float* data;
    int* cols;

    // memory reservation
    csr_spm = (csr_spm_t*) calloc(1, sizeof(csr_spm_t));
    if(!csr_spm) return NULL;

    rows = (int*) calloc(1, sizeof(int)*num_rows+1);
    (*csr_spm).rows = rows;
    if(!rows) { free_csr_spm(csr_spm); return NULL; }

    
    // initialitation of rows vector
    int num_elems = 0;
    for(int i = 0; i < num_rows; i++) {
        rows[i] = num_elems;
        num_elems += abs(rand() % (num_rows/16));
    }
    rows[num_rows] = num_elems;

    // memory reservation
    data = (float*) calloc(1, sizeof(float)*num_elems);
    (*csr_spm).data = data;
    cols = (int*) calloc(1, sizeof(int)*num_elems);
    (*csr_spm).cols = cols;
    if(!data || !cols) { free_csr_spm(csr_spm); return NULL; }

    
    // data vector initialitation
    for(int i = 0; i < num_elems; i++) { 
        data[i] = ((float) abs(rand() % 128))/128;
    }
    
    // cols vector initialitation
    for(int i = 0; i < num_rows; i++) {
        int elems = rows[i+1] - rows[i];
        int offset = 0;
        for(int j = rows[i]; j < rows[i+1]; j++) {
            offset += abs(rand() % (((num_rows - elems) - offset) + (j == rows[i])));
            offset += (j != rows[i]);
            cols[j] = offset;
            elems--;
        }
    }

    (*csr_spm).num_rows = num_rows;
    // entrego los resultados
    return csr_spm;
}

void free_csr_spm(csr_spm_t* csr_spm) {
    if(!csr_spm) return;
    if((*csr_spm).rows) free((*csr_spm).rows);
    if((*csr_spm).data) free((*csr_spm).data);
    if((*csr_spm).cols) free((*csr_spm).cols);
    free(csr_spm);
}

void print_csr_spm(csr_spm_t* csr_spm) { 
    
    int num_rows = (*csr_spm).num_rows;
    int* rows = (*csr_spm).rows;
    float* data = (*csr_spm).data;
    int* cols = (*csr_spm).cols;

    // rows vector
    printf("rows vector :");
    for(int i = 0; i < num_rows; i++) {
        if(!(i%32)) printf("\n   ");
        printf(" %d,", rows[i]);
    }
    printf("\n\n");

    // cols vector
    printf("cols vector :");
    for(int i = 0; i < rows[num_rows]; i++) {
        if(!(i%32)) printf("\n   ");
        printf(" %d,", cols[i]);
    }
    printf("\n\n");

    // data vector
    printf("data vector :");
    for(int i = 0; i < rows[num_rows]; i++) {
        if(!(i%32)) printf("\n   ");
        printf(" %f,", data[i]);
    }
    printf("\n\n");

}


float* csr_to_matrix(csr_spm_t* csr_spm) {
    if(!csr_spm) return NULL;
    float* mat = (float*) calloc(sizeof(float), (*csr_spm).num_rows*(*csr_spm).num_rows);
    if(!mat) return NULL;

    for(int i = 0; i < (*csr_spm).num_rows; i++) {
        for(int j = ((*csr_spm).rows)[i]; j < ((*csr_spm).rows)[i+1]; j++) { 
            mat[(i*(*csr_spm).num_rows)+((*csr_spm).cols)[j]] = ((*csr_spm).data)[j];
        }
    }

    return mat;
}

void print_mat(float* mat, int num_rows) {
    for(int i = 0; i < num_rows; i++) {
        printf("row %d:", i);
        for(int j = 0; j < num_rows; j++) {
            printf(" %f,", mat[(i*num_rows) + j]);
        }
        printf("\n");
    }
}