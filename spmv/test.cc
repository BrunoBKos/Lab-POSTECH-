#include "spcsr_generator.h"

#include <stdlib.h>

int main(int argc, char** argv) {

    int res = 0;
    int num_rows = 128;

    if(argc > 1) num_rows = atoi(argv[1]);
    csr_spm_t* csr_spm = spmcsr_gen(num_rows);
    if(!csr_spm) return 1;

    print_csr_spm(csr_spm);

    float* mat = csr_to_matrix(csr_spm);
    if(!mat) { free_csr_spm(csr_spm); return 2; }
    print_mat(mat, num_rows);

    free_csr_spm(csr_spm);
    return 0;
}