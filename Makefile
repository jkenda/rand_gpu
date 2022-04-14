CC= gcc
CXXC= g++
CFLAGS= -O3 -g -Wall -Wpedantic
CXXFLAGS= --std=c++17 -O3 -g -Wall -Wpedantic
INSTALL_PATH= $(HOME)/.local/lib/

SLURM_ARGS= --reservation=fri -c2 -G2

default: lib/librand_gpu.so

all: print pi_simple pi fastest_multiplier equality
lib/librand_gpu.so: lib bin/test_kernel install

lib: RNG.o
	@mkdir -p lib
	$(CXXC) $(CXXFLAGS) -shared -o lib/librand_gpu.so RNG.o -lOpenCL -lpthread

install: lib
	@mkdir -p $(INSTALL_PATH)
	cp lib/librand_gpu.so $(INSTALL_PATH)

run: pi
	LD_LIBRARY_PATH=~/.local/lib bin/pi

run_slurm:
	srun $(SLURM_ARGS) bin/pi | tee output &


bin/test_kernel: tools/test_kernel.cpp kernel.hpp
	@mkdir -p bin
	$(CXXC) $(CXXFLAGS) -o bin/test_kernel tools/test_kernel.cpp -lOpenCL
	@LD_LIBRARY_PATH=lib bin/test_kernel

RNG.o: src/RNG.cpp src/RNG.hpp kernel.hpp
	$(CXXC) $(CXXFLAGS) -c src/RNG.cpp -fPIC

kernel.hpp: src/kernels/server.cl
	tools/convert_kernel.py src/kernels/server.cl

print: lib/librand_gpu.so examples/print.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -Llib -o bin/print examples/print.c -lrand_gpu

pi: lib/librand_gpu.so examples/pi.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -Llib -o bin/pi examples/pi.c -lrand_gpu

pi_simple: lib/librand_gpu.so examples/pi_simple.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -Llib -o bin/pi_simple examples/pi_simple.c -lrand_gpu

pi_parallel: lib/librand_gpu.so examples/pi_parallel.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -Llib -o bin/pi_parallel examples/pi_parallel.c -lm -lrand_gpu -fopenmp

equality: lib/librand_gpu.so test/equality.cpp
	@mkdir -p bin
	$(CXXC) $(CXXFLAGS) -Llib -o bin/equality test/equality.cpp -lrand_gpu

fastest_multiplier: lib/librand_gpu.so test/fastest_multiplier.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -Llib -Wno-unused-value -o bin/fastest_multiplier test/fastest_multiplier.c -lrand_gpu

clean:
	-rm *.o
	-rm lib/*.so
	-rm kernel.hpp
