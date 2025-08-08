struct csr_spm_s {
    int num_rows;
    int* rows;
    float* data;
    int* cols;
} typedef csr_spm_t;

csr_spm_t* spmcsr_gen(int num_rows);

void free_csr_spm(csr_spm_t* csr_spm);

void print_csr_spm(csr_spm_t* csr_spm);

float* csr_to_matrix(csr_spm_t* csr_spm);

void print_mat(float* mat, int num_rows);