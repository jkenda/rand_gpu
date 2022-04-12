CC= gcc
CPPC= g++
CFLAGS= -O3 -s -Wall -Wpedantic
CPPFLAGS= -O3 -g -Wall -Wpedantic
INSTALL_PATH= $(HOME)/.local/lib/

SLURM_ARGS= --reservation=fri -c2 -G2

default: lib/librand_gpu.so

all: print pi pi_compare fastest_multiplier equality
lib/librand_gpu.so: lib bin/test_kernel install

lib: RandGPU.o
	@mkdir -p lib
	$(CPPC) $(CPPFLAGS) -shared -o lib/librand_gpu.so RandGPU.o -lOpenCL -lpthread

install: lib
	@mkdir -p $(INSTALL_PATH)
	cp lib/librand_gpu.so $(INSTALL_PATH)

run: pi_compare
	LD_LIBRARY_PATH=~/.local/lib bin/pi_compare

run_slurm:
	slurm $(SLURM_ARGS) LD_LIBRARY_PATH=~/.local/lib bin/pi_compare | tee output &


bin/test_kernel: test/test_kernel.cpp kernel.hpp
	@mkdir -p bin
	$(CPPC) $(CPPFLAGS) -o bin/test_kernel test/test_kernel.cpp -lOpenCL
	@LD_LIBRARY_PATH=lib bin/test_kernel

RandGPU.o: src/RandGPU.cpp src/exceptions.hpp kernel.hpp
	$(CPPC) $(CPPFLAGS) -c src/RandGPU.cpp -fPIC

kernel.hpp: src/kernels/server.cl
	tools/convert_kernel.py src/kernels/server.cl

print: lib/librand_gpu.so examples/print.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -Llib -o bin/print examples/print.c -lrand_gpu

pi: lib/librand_gpu.so examples/pi.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -Llib -o bin/pi examples/pi.c -lrand_gpu

pi_compare: lib/librand_gpu.so examples/pi_compare.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -Llib -o bin/pi_compare examples/pi_compare.c -lrand_gpu

equality: lib/librand_gpu.so test/equality.cpp
	@mkdir -p bin
	$(CPPC) $(CPPFLAGS) -Llib -o bin/equality test/equality.cpp -lrand_gpu

fastest_multiplier: lib/librand_gpu.so test/fastest_multiplier.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -Llib -Wno-unused-value -o bin/fastest_multiplier test/fastest_multiplier.c -lrand_gpu

clean:
	-rm *.o
	-rm lib/*.so
	-rm kernel.hpp
