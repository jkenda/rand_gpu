CC= gcc
CPPC= g++
CFLAGS= -O3 -g -Wall -Wpedantic
CPPFLAGS= --std=c++17 -O3 -g -Wall -Wpedantic

SLURM_ARGS= --reservation=fri -c2 -G2

# Arnes
# izmeri pri različnih deležih generiranja proti računanju
# kakšna je razlika, koliko naklj. števil naenkrat zahtevam
# naredi par grafov iz meritev
# več algoritmov (predvsem Mersenne twister), podajn možnost izbire
# (navaden Tyche namesto Tyche_I)

default: lib/librand_gpu.so

all: print pi pi_simple pi_parallel equality frequency fastest_multiplier speedup_measurement
lib/librand_gpu.so: lib bin/test_kernel install

lib: RNG.o
	@mkdir -p lib
	$(CPPC) $(CPPFLAGS) -shared -o lib/librand_gpu.so RNG.o -lOpenCL -lpthread

install:
	@mkdir -p ~/.local/lib/
	cp lib/librand_gpu.so ~/.local/lib/

run: pi_parallel
	LD_LIBRARY_PATH=lib bin/pi_parallel

run_slurm: pi_parallel
	LD_LIBRARY_PATH=lib srun $(SLURM_ARGS) bin/pi_parallel | tee output &


bin/test_kernel: tools/test_kernel.cpp kernel.hpp
	@mkdir -p bin/c++
	$(CPPC) $(CPPFLAGS) -o bin/test_kernel tools/test_kernel.cpp -lOpenCL
	@LD_LIBRARY_PATH=lib bin/test_kernel

RNG.o: src/RNG.hpp src/RNG.cpp kernel.hpp
	$(CPPC) $(CPPFLAGS) -c src/RNG.cpp -fPIC

kernel.hpp: src/kernels/server.cl
	tools/convert_kernel.py src/kernels/server.cl kernel.hpp

print: lib/librand_gpu.so examples/print.c
	@mkdir -p bin/c++
	$(CC) $(CFLAGS) -Llib -o bin/print examples/print.c -lrand_gpu
	$(CPPC) $(CPPFLAGS) -Llib -o bin/c++/print examples/print.cpp -lrand_gpu

pi: lib/librand_gpu.so examples/pi.c
	@mkdir -p bin/c++
	$(CC) $(CFLAGS) -Llib -o bin/pi examples/pi.c -lrand_gpu
	$(CPPC) $(CPPFLAGS) -Llib -o bin/c++/pi examples/pi.cpp -lrand_gpu

pi_simple: lib/librand_gpu.so examples/pi_simple.c
	@mkdir -p bin/c++
	$(CC) $(CFLAGS) -Llib -o bin/pi_simple examples/pi_simple.c -lrand_gpu
	$(CPPC) $(CPPFLAGS) -Llib -o bin/c++/pi_simple examples/pi_simple.cpp -lrand_gpu

pi_parallel: lib/librand_gpu.so examples/pi_parallel.cpp
	@mkdir -p bin/c++
	$(CC) $(CFLAGS) -Llib -o bin/pi_parallel examples/pi_parallel.c -lm -lrand_gpu -fopenmp
	$(CPPC) $(CPPFLAGS) -Llib -o bin/c++/pi_parallel examples/pi_parallel.cpp -lm -lrand_gpu -fopenmp

equality: lib/librand_gpu.so test/equality.cpp
	@mkdir -p bin/c++
	$(CPPC) $(CPPFLAGS) -Llib -o bin/c++/equality test/equality.cpp -lrand_gpu

frequency: lib/librand_gpu.so test/frequency.cpp
	@mkdir -p bin/c++
	$(CPPC) $(CPPFLAGS) -Llib -o bin/c++/frequency test/frequency.cpp -lrand_gpu -fopenmp


fastest_multiplier: lib/librand_gpu.so test/fastest_multiplier.c
	@mkdir -p bin/c++
	$(CC) $(CFLAGS) -Llib -Wno-unused-value -o bin/fastest_multiplier test/fastest_multiplier.c -lrand_gpu

speedup_measurement: lib/librand_gpu.so test/speedup_measurement.cpp
	@mkdir -p bin/c++
	$(CPPC) $(CPPFLAGS) -Llib -o bin/c++/speedup_measurement test/speedup_measurement.cpp -lrand_gpu

clean:
	-rm -rf bin/*
	-rm *.o
	-rm lib/*.so
	-rm kernel.hpp
