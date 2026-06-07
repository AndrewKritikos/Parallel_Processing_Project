#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <string.h>
#include <cuda_runtime.h>

double get_current_time(){
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1000000.0;
}

// Kernel gia ta stats. Kathe thread kanei mia sthlh (column)
__global__ void compute_block_statistics_kernel(double *buffer, long long current_rows, long long D, 
    double *sum, double *sum_sq, double *min_val, double *max_val){
    
    long long j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < D) {
        for(long long i = 0; i < current_rows; i++){
            double val = buffer[i * D + j];
            sum[j] += val;
            sum_sq[j] += val * val;

            if(val < min_val[j]){
                min_val[j] = val;
            }
            if(val > max_val[j]){
                max_val[j] = val;
            }
        }
    }
}

// Kernel gia to scaling. 1 thread = 1 row
__global__ void scale_block_kernel(double *buffer, long long current_rows, long long D, int mode,
    double *mean, double *std_dev, double *min_val, double *max_val){

    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < current_rows) {
        if(mode == 0){ // standard
            for(long long j = 0; j < D; j++){
                buffer[i * D + j] = (buffer[i * D + j] - mean[j]) / std_dev[j];
            }
        }
        else if(mode == 1){ // minmax
            for(long long j = 0; j < D; j++){
                double range = max_val[j] - min_val[j];
                if(range == 0.0){
                    range = 1.0;
                }
                buffer[i * D + j] = (buffer[i * D + j] - min_val[j]) / range;
            } 
        }
    }
}


int main(int argc, char* argv[]){
    if (argc < 6){
        printf("You need to run the programm with 6 args\n");
        return 0;
    }

    char *input_file = argv[1];
    char *output_file = argv[2];
    long long N = atoll(argv[3]);
    long long D = atoll(argv[4]);
    char *mode = argv[5];
    long long block_rows = (argc == 7) ? atoll(argv[6]) : 100000; 
    
    int mode_int = (strcmp(mode, "standard") == 0) ? 0 : 1;

    // desmeush mnhmhs sto CPU
    double *sum = (double *)calloc(D, sizeof(double)); 
    double *sum_sq = (double *)calloc(D, sizeof(double));
    double *min_val = (double *)malloc(D * sizeof(double));
    double *max_val = (double *)malloc(D * sizeof(double));
    double *mean = (double *)malloc(D * sizeof(double));
    double *std_dev = (double *)malloc(D * sizeof(double));

    for(long long j = 0; j < D; j++){ 
        min_val[j] = DBL_MAX;
        max_val[j] = -DBL_MAX;
    }

    // GPU memory allocations
    double *d_buffer, *d_sum, *d_sum_sq, *d_min_val, *d_max_val, *d_mean, *d_std_dev;
    cudaMalloc((void**)&d_buffer, block_rows * D * sizeof(double));
    cudaMalloc((void**)&d_sum, D * sizeof(double));
    cudaMalloc((void**)&d_sum_sq, D * sizeof(double));
    cudaMalloc((void**)&d_min_val, D * sizeof(double));
    cudaMalloc((void**)&d_max_val, D * sizeof(double));
    cudaMalloc((void**)&d_mean, D * sizeof(double));
    cudaMalloc((void**)&d_std_dev, D * sizeof(double));

    // stelnoume ta arxika stats sth GPU
    cudaMemcpy(d_sum, sum, D * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum_sq, sum_sq, D * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min_val, min_val, D * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_val, max_val, D * sizeof(double), cudaMemcpyHostToDevice);

    FILE *fptr = fopen(input_file, "rb"); 
    if(fptr == NULL){
        printf("Error no file %s found", input_file);
        return 1;
    }

    FILE *output_file_ptr = fopen(output_file, "wb");
    if(output_file_ptr == NULL){
        printf("Error creating output file %s found\n", output_file);
        return 1;
    }

    double *buffer = (double *)malloc(block_rows * D * sizeof(double));
    long long rows_processed = 0;
    
    printf("Starting the statistics calculaions on GPU\n");

    double t1 = get_current_time();
    while(rows_processed < N){
        long long current_rows;
        long long remaining_rows = N - rows_processed;
        if(remaining_rows < block_rows){
            current_rows = remaining_rows;
        } else{
            current_rows = block_rows;
        }
        
        // diavasma apo to arxeio
        fread(buffer, sizeof(double), current_rows * D, fptr);
        
        // copy buffer to GPU
        cudaMemcpy(d_buffer, buffer, current_rows * D * sizeof(double), cudaMemcpyHostToDevice);
        
        // trexoume to prwto kernel (threads = columns)
        int threadsPerBlock = 256;
        int blocks = (D + threadsPerBlock - 1) / threadsPerBlock;
        compute_block_statistics_kernel<<<blocks, threadsPerBlock>>>(d_buffer, current_rows, D, d_sum, d_sum_sq, d_min_val, d_max_val);
        
        cudaDeviceSynchronize(); // perimenoume na teleiwsei
        
        rows_processed += current_rows;
    }
    double t2 = get_current_time();

    printf("Took %f seconds to calculate statistics\n", t2 - t1);
    
    // ferame ta apotelesmata pisw gia to teliko math sto CPU
    cudaMemcpy(sum, d_sum, D * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(sum_sq, d_sum_sq, D * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(min_val, d_min_val, D * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_val, d_max_val, D * sizeof(double), cudaMemcpyDeviceToHost);

    // ipologismos mean k std_dev opws sto serial
    for (long long j = 0; j < D; j++) {
        mean[j] = sum[j] / N;
        double var = (sum_sq[j] / N) - (mean[j] * mean[j]);
        if(var < 0.0){
            var = 0.0;
        }
        std_dev[j] = sqrt(var);
    } 

    // stelnoume mean kai std_dev gia th 2h fash
    cudaMemcpy(d_mean, mean, D * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_std_dev, std_dev, D * sizeof(double), cudaMemcpyHostToDevice);

    rows_processed = 0;
    double t3 = get_current_time();
    rewind(fptr);

    while(rows_processed < N){
        long long current_rows = (N - rows_processed < block_rows) ? (N - rows_processed) : block_rows;
        
        fread(buffer, sizeof(double), current_rows * D, fptr);
        
        cudaMemcpy(d_buffer, buffer, current_rows * D * sizeof(double), cudaMemcpyHostToDevice);

        // trexoume to 2o kernel (threads = rows)
        int threadsPerBlock = 256;
        int blocks = (current_rows + threadsPerBlock - 1) / threadsPerBlock;
        scale_block_kernel<<<blocks, threadsPerBlock>>>(d_buffer, current_rows, D, mode_int, d_mean, d_std_dev, d_min_val, d_max_val);
        
        cudaDeviceSynchronize();

        // get results back
        cudaMemcpy(buffer, d_buffer, current_rows * D * sizeof(double), cudaMemcpyDeviceToHost);

        // grafoume sto file
        fwrite(buffer, sizeof(double), current_rows * D, output_file_ptr);
        rows_processed += current_rows;
    }

    double t4 = get_current_time();
    printf("Writing to the output file took %f seconds\n", t4-t3);

    // free mnhmhs
    free(buffer); free(sum); free(sum_sq); free(min_val); free(max_val); free(mean); free(std_dev);
    
    cudaFree(d_buffer); cudaFree(d_sum); cudaFree(d_sum_sq); cudaFree(d_min_val); 
    cudaFree(d_max_val); cudaFree(d_mean); cudaFree(d_std_dev);
    
    fclose(fptr);
    fclose(output_file_ptr);

    return 0;
}