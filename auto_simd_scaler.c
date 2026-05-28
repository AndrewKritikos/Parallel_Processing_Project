/*
--------------------------------------------------
Test Results
--------------------------------------------------
__________________________________________________
N = 1000000, D = 32, mode = 'standard'
Took 0.103269 seconds to calculate statistics
Writing to the output file took 0.346605 seconds
__________________________________________________
N = 1000000, D = 32, mode = 'minmax'
Took 0.106092 seconds to calculate statistics
Writing to the output file took 0.343970 seconds
__________________________________________________
N = 5000000, D = 64, mode = 'standard'
Took 1.123213 seconds to calculate statistics
Writing to the output file took 3.581857 seconds
__________________________________________________
N = 5000000, D = 64, mode = 'minmax'
Took 1.126543 seconds to calculate statistics
Writing to the output file took 3.736975 seconds

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <string.h>

double get_current_time(){
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1000000.0;
}

void compute_block_statistics(double *restrict buffer, long long current_rows, long long D, 
    double *restrict sum, double *restrict sum_sq, double *restrict min_val, double *restrict max_val){
    
    for(long long i=0; i<current_rows; i++){
        for(long long j=0; j<D; j++){

            double val = buffer[i * D + j];
            sum[j] += val;
            sum_sq[j] += val * val;

            if(val< min_val[j]){
                min_val[j] = val;
            }
            if(val>max_val[j]){
                max_val[j] = val;
            }

        }
    }

}

void scale_block(double *restrict buffer, long long current_rows, long long D, char *mode,
    double *restrict mean, double *restrict std_dev, double *restrict min_val, double *restrict max_val){

        if(strcmp(mode, "standard") == 0){
            for(long long i=0; i<current_rows; i++){
                for(long long j=0; j<D; j++){
                    buffer[i * D +j] = (buffer[i*D+j] - mean[j]) / std_dev[j];
                }
            }
        }
        else if(strcmp(mode, "minmax") == 0){
           for(long long i=0; i<current_rows; i++){
                for(long long j=0; j<D; j++){
                    double range = max_val[j] - min_val[j];
                    if(range == 0.0){
                        range = 1.0;
                    }
                    buffer[i * D +j] = (buffer[i*D+j] - min_val[j]) / range;
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
    long long block_rows = (argc == 7) ? atoll(argv[6]) : 100000; //if 7 args when called 7th is blockrows else 100000

    double *sum = (double *)calloc(D, sizeof(double)); //memory alloc and initialisation with 0
    double *sum_sq = (double *)calloc(D, sizeof(double));//memory alloc and initialisation with 0
    double *min_val = (double *)malloc(D * sizeof(double));//memory allocation
    double *max_val = (double *)malloc(D * sizeof(double));//memory allocation
    double *mean = (double *)malloc(D * sizeof(double));//memory allocation
    double *std_dev = (double *)malloc(D * sizeof(double));//memory allocation

    //Initializing the min_val and max_val array with min and max values
    for(long long j = 0; j<D; j++){ 
        min_val[j] = DBL_MAX;
        max_val[j] = -DBL_MAX;
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
    double *buffer = (double *)malloc(block_rows * D *sizeof(double));
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
    

    for (long long j = 0; j < D; j++) {
        mean[j] = sum[j] / N;

        double var = (sum_sq[j] / N) - (mean[j] * mean[j]);
        if(var < 0.0){
            var = 0.0;
        }
        std_dev[j] = sqrt(var);
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