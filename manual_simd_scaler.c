/*
--------------------------------------------------
Test Results
--------------------------------------------------
__________________________________________________
N = 1000000, D = 32, mode = 'standard'
Took 0.102304 seconds to calculate statistics
Writing to the output file took 0.333699 seconds
__________________________________________________
N = 1000000, D = 32, mode = 'minmax'
Took 0.119621 seconds to calculate statistics
Writing to the output file took 0.371114 seconds
__________________________________________________
N = 5000000, D = 64, mode = 'standard'
Took 1.086308 seconds to calculate statistics
Writing to the output file took 3.561501 seconds
__________________________________________________
N = 5000000, D = 64, mode = 'minmax'
Took 1.084720 seconds to calculate statistics
Writing to the output file took 3.428537 seconds
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <string.h>
#include <immintrin.h>

double get_current_time(){
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1000000.0;
}

void compute_block_statistics(double *buffer, long long current_rows, long long D, 
    double *sum, double *sum_sq, double *min_val, double *max_val){
    
    for(long long i=0; i<current_rows; i++){
        for(long long j=0; j<D; j+=4){ //Implementing AVX so can fit 4 doubles = 256 bits in each process

            
            //load 4 values in each vector for processing
            __m256d val_vec = _mm256_load_pd(&buffer[i * D + j]);
            __m256d sum_vec = _mm256_load_pd(&sum[j]);
            __m256d sum_sq_vec = _mm256_load_pd(&sum_sq[j]);
            __m256d min_vec = _mm256_load_pd(&min_val[j]);
            __m256d max_vec = _mm256_load_pd(&max_val[j]);

            //performing the calculations
            sum_vec = _mm256_add_pd(sum_vec, val_vec);
            __m256d square = _mm256_mul_pd(val_vec, val_vec);
            sum_sq_vec = _mm256_add_pd(sum_sq_vec, square);
            min_vec = _mm256_min_pd(min_vec, val_vec);
            max_vec = _mm256_max_pd(max_vec, val_vec);

            //storing the vectors back to the designated arrays
            _mm256_store_pd(&sum[j], sum_vec);
            _mm256_store_pd(&sum_sq[j], sum_sq_vec);
            _mm256_store_pd(&min_val[j], min_vec);
            _mm256_store_pd(&max_val[j], max_vec);
        }
    }

}

void scale_block(double *buffer, long long current_rows, long long D, char *mode,
    double *mean, double *std_dev, double *min_val, double *max_val){

        if(strcmp(mode, "standard") == 0){
            for(long long i=0; i<current_rows; i++){
                for(long long j=0; j<D; j += 4){//Implementing AVX so can fit 4 doubles = 256 bits in each process

                    //load 4 values in each vector for processing
                    __m256d val_vec = _mm256_load_pd(&buffer[i * D + j]);
                    __m256d mean_vec = _mm256_load_pd(&mean[j]);
                    __m256d std_vec = _mm256_load_pd(&std_dev[j]);
                    
                    //performing the calculations
                    __m256d sub_vec = _mm256_sub_pd(val_vec, mean_vec);
                    __m256d div_vec = _mm256_div_pd(sub_vec, std_vec);

                    //storing the vectors back to the designated arrays
                    _mm256_store_pd(&buffer[i * D + j], div_vec);
                }
            }
        }
        else if(strcmp(mode, "minmax") == 0){
            double *ranges = (double *)aligned_alloc(32, D * sizeof(double));

            __m256d zero_vec = _mm256_setzero_pd();
            __m256d one_vec = _mm256_set1_pd(1.0);

            for(long long j = 0; j < D; j += 4){
                __m256d min_vec = _mm256_load_pd(&min_val[j]);
                __m256d max_vec = _mm256_load_pd(&max_val[j]); 

                __m256d range_vec = _mm256_sub_pd(max_vec, min_vec);
                __m256d mask = _mm256_cmp_pd(range_vec, zero_vec, _CMP_EQ_OQ);
                range_vec = _mm256_blendv_pd(range_vec, one_vec, mask);

                _mm256_store_pd(&ranges[j], range_vec);

            } 
           for(long long i=0; i<current_rows; i++){
                for(long long j=0; j<D; j += 4){//Implementing AVX so can fit 4 doubles = 256 bits in each process

                    //load 4 values in each vector for processing
                    __m256d val_vec = _mm256_load_pd(&buffer[i * D + j]);
                    __m256d min_vec = _mm256_load_pd(&min_val[j]);
                    __m256d range_vec = _mm256_load_pd(&ranges[j]);

                    //performing the calculations
                    __m256d sub_vec = _mm256_sub_pd(val_vec, min_vec);
                    __m256d div_vec = _mm256_div_pd(sub_vec, range_vec);
                    
                    //storing the vectors back to the designated arrays
                    _mm256_store_pd(&buffer[i * D + j], div_vec); 
                }
            } 
            free(ranges);
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
    long long D = atoll(argv[4]); //using long long for better precision with big multiplications
    char *mode = argv[5];
    long long block_rows = (argc == 7) ? atoll(argv[6]) : 100000; //if 7 args when called 7th is blockrows else 100000

    double *sum = (double *)aligned_alloc(32, D *  sizeof(double));//alligned memory allocation
    memset(sum, 0, D * sizeof(double)); //initialisation with 0
    double *sum_sq = (double *)aligned_alloc(32, D *  sizeof(double));//alligned memory allocation
    memset(sum_sq, 0, D * sizeof(double));//initialisation with 0
    double *min_val = (double *)aligned_alloc(32, D * sizeof(double));//alligned memory allocation
    double *max_val = (double *)aligned_alloc(32, D * sizeof(double));//alligned memory allocation
    double *mean = (double *)aligned_alloc(32, D * sizeof(double));//alligned memory allocation
    double *std_dev = (double *)aligned_alloc(32, D * sizeof(double));//alligned memory allocation

    //Initializing the min_val and max_val array with min and max values
    for(long long j = 0; j<D; j+=4){ 

        __m256d min_vec = _mm256_load_pd(&min_val[j]);
        __m256d max_vec = _mm256_load_pd(&max_val[j]);

        min_vec = _mm256_set1_pd(DBL_MAX);
        max_vec = _mm256_set1_pd(-DBL_MAX);
        
        _mm256_store_pd(&min_val[j], min_vec);
        _mm256_store_pd(&max_val[j], max_vec);
    }
    

    //Opening the bin input file
    FILE *fptr = fopen(input_file, "rb"); //rb for reading binary file
    if(fptr == NULL){
        printf("Error no file %s found", input_file);
        return 1;
    }

    FILE *output_file_ptr = fopen(output_file, "wb");
    if(output_file_ptr == NULL){
        printf("Error creating output file %s found\n", output_file);
        return 1;
    }

    //Initializing buffer size in RAM
    double *buffer = (double *)aligned_alloc(32, block_rows * D *sizeof(double));
    long long rows_processed = 0;
    
    printf("Starting the statistics calculaions\n");

    

    double t1 = get_current_time();
    while(rows_processed < N){
        long long current_rows;
        long long remaining_rows = N - rows_processed;
        if(remaining_rows < block_rows){
            current_rows = remaining_rows;
        } else{
            current_rows = block_rows;
        }
        fread(buffer, sizeof(double), current_rows * D, fptr);
        compute_block_statistics(buffer, current_rows, D, sum, sum_sq, min_val, max_val);
        rows_processed += current_rows;
    }
    double t2 = get_current_time();

    printf("Took %f seconds to calculate statistics\n", t2 - t1);
    

    __m256d n_vec = _mm256_set1_pd((double)N);
    __m256d zero_vec = _mm256_setzero_pd();
    for (long long j = 0; j < D; j += 4) {

        __m256d sum_vec = _mm256_load_pd(&sum[j]);
        __m256d sum_sq_vec = _mm256_load_pd(&sum_sq[j]);
        
        __m256d mean_vec = _mm256_div_pd(sum_vec, n_vec);
        _mm256_store_pd(&mean[j], mean_vec);

        __m256d mean_sq_vec = _mm256_mul_pd(mean_vec, mean_vec);
        __m256d div_vec = _mm256_div_pd(sum_sq_vec,n_vec);
        __m256d var_vec = _mm256_sub_pd(div_vec, mean_sq_vec);

        __m256d mask = _mm256_cmp_pd(var_vec, zero_vec, _CMP_LT_OQ);
        var_vec = _mm256_blendv_pd(var_vec, zero_vec, mask);
        __m256d std_dev_vec = _mm256_sqrt_pd(var_vec);
        _mm256_store_pd(&std_dev[j], std_dev_vec);

    } 

    rows_processed = 0;
    double t3 = get_current_time();
    rewind(fptr);

    while(rows_processed < N){
        long long current_rows = (N - rows_processed < block_rows) ? (N - rows_processed) : block_rows;
        
        fread(buffer, sizeof(double), current_rows * D, fptr);
        scale_block(buffer, current_rows, D, mode, mean, std_dev, min_val, max_val);

        fwrite(buffer, sizeof(double), current_rows * D, output_file_ptr);
        rows_processed += current_rows;
    }

    double t4 = get_current_time();
    printf("Writing to the output file took %f seconds\n", t4-t3);


    free(buffer);
    free(sum);
    free(sum_sq);
    free(min_val);
    free(max_val);
    free(mean);
    free(std_dev);
    fclose(fptr);
    fclose(output_file_ptr);


    return 0;
}