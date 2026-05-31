#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <string.h>
#include <mpi.h>

double get_current_time(){
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1000000.0;
}

void compute_block_statistics(double *buffer, long long current_rows, long long D, 
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

}

void scale_block(double *buffer, long long current_rows, long long D, char *mode,
    double *mean, double *std_dev, double *min_val, double *max_val){

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
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 6){
        if(rank == 0) printf("You need to run the programm with 6 args\n");
        MPI_Finalize();
        return 0;
    }

    char *input_file = argv[1];
    char *output_file = argv[2];
    long long N = atoll(argv[3]);
    long long D = atoll(argv[4]);
    char *mode = argv[5];
    long long block_rows = (argc == 7) ? atoll(argv[6]) : 100000; //if 7 args when called 7th is blockrows else 100000

    //Find out how many rows this specific process will handle
    long long local_N;
    long long local_start_row;
    long long remainder = N % size;
    
    if (rank < remainder) {
        local_N = (N / size) + 1;
        local_start_row = rank * local_N;
    } else {
        local_N = (N / size);
        local_start_row = rank * local_N + remainder;
    }

    //local arrays for this process
    double *local_sum = (double *)calloc(D, sizeof(double)); 
    double *local_sum_sq = (double *)calloc(D, sizeof(double));
    double *local_min_val = (double *)malloc(D * sizeof(double));
    double *local_max_val = (double *)malloc(D * sizeof(double));

    //global arrays for the final results
    double *sum = (double *)calloc(D, sizeof(double)); //memory alloc and initialisation with 0
    double *sum_sq = (double *)calloc(D, sizeof(double));//memory alloc and initialisation with 0
    double *min_val = (double *)malloc(D * sizeof(double));//memory allocation
    double *max_val = (double *)malloc(D * sizeof(double));//memory allocation
    double *mean = (double *)malloc(D * sizeof(double));//memory allocation
    double *std_dev = (double *)malloc(D * sizeof(double));//memory allocation

    //Initializing the min_val and max_val array with min and max values
    for(long long j = 0; j<D; j++){ 
        local_min_val[j] = DBL_MAX;
        local_max_val[j] = -DBL_MAX;
    }

    //Opening the bin input file with MPI
    MPI_File fptr;
    if(MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fptr) != MPI_SUCCESS){
        if (rank == 0) printf("Error no file %s found", input_file);
        MPI_Finalize();
        return 1;
    }

    MPI_File output_file_ptr;
    if(MPI_File_open(MPI_COMM_WORLD, output_file, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file_ptr) != MPI_SUCCESS){
        if (rank == 0) printf("Error creating output file %s found\n", output_file);
        MPI_Finalize();
        return 1;
    }

    //Initializing buffer size in RAM
    double *buffer = (double *)malloc(block_rows * D *sizeof(double));
    long long rows_processed = 0;
    
    if (rank == 0) printf("Starting the statistics calculaions\n");

    
    MPI_Barrier(MPI_COMM_WORLD); //sync everyone before starting the timer
    double t1 = get_current_time();
    
    while(rows_processed < local_N){
        long long current_rows;
        long long remaining_rows = local_N - rows_processed;
        if(remaining_rows < block_rows){
            current_rows = remaining_rows;
        } else{
            current_rows = block_rows;
        }
        
        //Calculate where to read from
        MPI_Offset offset = (local_start_row + rows_processed) * D * sizeof(double);
        MPI_File_read_at(fptr, offset, buffer, current_rows * D, MPI_DOUBLE, MPI_STATUS_IGNORE);
        
        compute_block_statistics(buffer, current_rows, D, local_sum, local_sum_sq, local_min_val, local_max_val);
        rows_processed += current_rows;
    }
    
    //Combine all local stats into the global arrays
    MPI_Allreduce(local_sum, sum, D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_sum_sq, sum_sq, D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_min_val, min_val, D, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_max_val, max_val, D, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = get_current_time();

    if (rank == 0) printf("Took %f seconds to calculate statistics\n", t2 - t1);
    

    for (long long j = 0; j < D; j++) {
        mean[j] = sum[j] / N;

        double var = (sum_sq[j] / N) - (mean[j] * mean[j]);
        if(var < 0.0){
            var = 0.0;
        }
        std_dev[j] = sqrt(var);
    } 

    rows_processed = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t3 = get_current_time();
    
    //No need to rewind(fptr) here because MPI_File_read_at uses exact offsets!

    while(rows_processed < local_N){
        long long current_rows = (local_N - rows_processed < block_rows) ? (local_N - rows_processed) : block_rows;
        
        MPI_Offset offset = (local_start_row + rows_processed) * D * sizeof(double);
        
        MPI_File_read_at(fptr, offset, buffer, current_rows * D, MPI_DOUBLE, MPI_STATUS_IGNORE);
        scale_block(buffer, current_rows, D, mode, mean, std_dev, min_val, max_val); //using the global stats here

        MPI_File_write_at(output_file_ptr, offset, buffer, current_rows * D, MPI_DOUBLE, MPI_STATUS_IGNORE);
        rows_processed += current_rows;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t4 = get_current_time();
    
    if (rank == 0) printf("Writing to the output file took %f seconds\n", t4-t3);


    free(buffer);
    free(sum);
    free(sum_sq);
    free(min_val);
    free(max_val);
    free(mean);
    free(std_dev);
    free(local_sum);
    free(local_sum_sq);
    free(local_min_val);
    free(local_max_val);
    
    MPI_File_close(&fptr);
    MPI_File_close(&output_file_ptr);

    MPI_Finalize();
    return 0;
}