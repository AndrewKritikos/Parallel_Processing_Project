CC=gcc
MPICC=mpicc
CFLAGS=-Wall -O3 
LDFLAGS=-lm

all: serial_scaler manual_simd_scaler auto_simd_scaler mpi_scaler

serial_scaler: serial_scaler.c
	$(CC) -O0 -o serial_scaler serial_scaler.c $(LDFLAGS)

manual_simd_scaler: manual_simd_scaler.c
	$(CC) $(CFLAGS) -march=native -fopt-info-vec-optimized -fno-tree-vectorize -o manual_simd_scaler manual_simd_scaler.c $(LDFLAGS)

auto_simd_scaler: auto_simd_scaler.c
	$(CC) $(CFLAGS) -march=native -fopt-info-vec-optimized -o auto_simd_scaler auto_simd_scaler.c $(LDFLAGS)

mpi_scaler: mpi_scaler.c
	$(MPICC) $(CFLAGS) -o mpi_scaler mpi_scaler.c $(LDFLAGS)

clean:
	rm -f serial_scaler manual_simd_scaler auto_simd_scaler mpi_scaler