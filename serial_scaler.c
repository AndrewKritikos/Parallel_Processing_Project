#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

int compute_block_statistics(double *buffer, long long current_rows, long long D, 
    double *sum, double *sum_sq, double *min_val, double *max_val){
    
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

    return 0;

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

    //Initializing buffer size in RAM
    double *buffer = (double *)malloc(block_rows * D *sizeof(double));
    long long rows_processed = 0;
    
    printf("Starting the statistics calculaions");

    time_t seconds;

    double t1 = time(NULL);
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
    double t2 = time(NULL);

    printf("Took %f seconds to calculate statistics\n", t2 - t1);
    fclose(fptr);

    printf("\n--- Δείγμα Στατιστικών (Πρώτες Στήλες) ---\n");
    long long cols_to_print = (D < 5) ? D : 5;
    for (long long j = 0; j < cols_to_print; j++) {
        printf("Στήλη %lld:\n", j);
        printf("  Min    = %10.4f\n", min_val[j]);
        printf("  Max    = %10.4f\n", max_val[j]);
    }

    free(buffer);
    free(sum);
    free(sum_sq);
    free(min_val);
    free(max_val);
    free(mean);
    free(std_dev);


    return 0;
}