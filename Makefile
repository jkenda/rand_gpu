CC= gcc
CFLAGS= -O3 -g -Wall -Wpedantic
CPPFLAGS= --std=c++17 -O3 -g -Wall -Wpedantic
LINK_EXEC= -lstdc++ -lrand_gpu
LINK_LIB= -lstdc++ -lOpenCL

SLURM_ARGS= --reservation=fri -c2 -G2

# Arnes
# izmeri pri različnih deležih generiranja proti računanju
# kakšna je razlika, koliko naklj. števil naenkrat zahtevam
# naredi par grafov iz meritev

default: lib/librand_gpu.so

all: print pi pi_simple pi_parallel algorithms equality frequency fastest_multiplier speedup_measurement
lib: lib/librand_gpu.so

lib/librand_gpu.so: RNG.o bin/test_kernel
	@mkdir -p lib
	$(CC) $(CPPFLAGS) -shared -o lib/librand_gpu.so RNG.o $(LINK_LIB)

install:
	@mkdir -p ~/.local/lib/
	cp lib/librand_gpu.so ~/.local/lib/

run: pi_parallel
	LD_LIBRARY_PATH=lib bin/pi_parallel

run_slurm: pi_parallel
	LD_LIBRARY_PATH=lib srun $(SLURM_ARGS) bin/pi_parallel | tee output &


bin/test_kernel: tools/test_kernel.cpp kernel.hpp
	@mkdir -p bin/c++
	$(CC) $(CPPFLAGS) -o bin/test_kernel tools/test_kernel.cpp $(LINK_LIB)
	@LD_LIBRARY_PATH=lib bin/test_kernel

RNG.o: src/RNG.hpp src/RNG.cpp kernel.hpp
	$(CC) $(CPPFLAGS) -c src/RNG.cpp -fPIC

kernel.hpp: tools/convert_kernel.py src/kernels/*.cl
	tools/convert_kernel.py src/kernels/ kernel.hpp

print: lib/librand_gpu.so examples/print.c
	@mkdir -p bin/c++
	$(CC) $(CFLAGS) -Llib -o bin/print examples/print.c $(LINK_EXEC)
	$(CC) $(CPPFLAGS) -Llib -o bin/c++/print examples/print.cpp $(LINK_EXEC)

pi: lib/librand_gpu.so examples/pi.c
	@mkdir -p bin/c++
	$(CC) $(CFLAGS) -Llib -o bin/pi examples/pi.c $(LINK_EXEC)
	$(CC) $(CPPFLAGS) -Llib -o bin/c++/pi examples/pi.cpp $(LINK_EXEC)

pi_simple: lib/librand_gpu.so examples/pi_simple.c
	@mkdir -p bin/c++
	$(CC) $(CFLAGS) -Llib -o bin/pi_simple examples/pi_simple.c $(LINK_EXEC)
	$(CC) $(CPPFLAGS) -Llib -o bin/c++/pi_simple examples/pi_simple.cpp $(LINK_EXEC)

pi_parallel: lib/librand_gpu.so examples/pi_parallel.cpp
	@mkdir -p bin/c++
	$(CC) $(CFLAGS) -Llib -o bin/pi_parallel examples/pi_parallel.c -lstdc++ -lm -lrand_gpu -fopenmp
	$(CC) $(CPPFLAGS) -Llib -o bin/c++/pi_parallel examples/pi_parallel.cpp -lstdc++ -lm -lrand_gpu -fopenmp


algorithms: lib/librand_gpu.so test/algorithms.cpp
	@mkdir -p bin/c++
	$(CC) $(CPPFLAGS) -Llib -o bin/c++/algorithms test/algorithms.cpp $(LINK_EXEC)

equality: lib/librand_gpu.so test/equality.cpp
	@mkdir -p bin/c++
	$(CC) $(CPPFLAGS) -Llib -o bin/c++/equality test/equality.cpp $(LINK_EXEC)

frequency: lib/librand_gpu.so test/frequency.cpp
	@mkdir -p bin/c++
	$(CC) $(CPPFLAGS) -Llib -o bin/c++/frequency test/frequency.cpp $(LINK_EXEC) -fopenmp


fastest_multiplier: lib/librand_gpu.so test/fastest_multiplier.c
	@mkdir -p bin/c++
	$(CC) $(CFLAGS) -Llib -Wno-unused-value -o bin/fastest_multiplier test/fastest_multiplier.c $(LINK_EXEC)

speedup_measurement: lib/librand_gpu.so test/speedup_measurement.cpp
	@mkdir -p bin/c++
	$(CC) $(CPPFLAGS) -Llib -o bin/c++/speedup_measurement test/speedup_measurement.cpp $(LINK_EXEC)

clean:
	-rm -rf bin/*
	-rm *.o
	-rm lib/*.so
	-rm kernel.hpp
