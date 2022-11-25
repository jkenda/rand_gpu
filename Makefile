CC= gcc # clang
CPPC= g++ # clang++
CFLAGS= -O3 -ggdb -Wall -Wpedantic
CPPFLAGS= --std=c++17 -O3 -ggdb -Wall -Wpedantic
SLURM_ARGS= --reservation=fri -c2 -G2

# Arnes
# kakšna je razlika, koliko naklj. števil naenkrat zahtevam

default: lib

all: print pi pi_simple pi_urandom pi_parallel coin_flip myurandom algorithms equality frequency fastest_multiplier speedup_measurement
lib: lib/librand_gpu.so

lib/librand_gpu.so: RNG.o
	@mkdir -p lib
	$(CPPC) $(CPPFLAGS) -shared -o lib/librand_gpu.so RNG.o -lOpenCL -flto
	-bin/test_kernel

install: lib/librand_gpu.so
#	@mkdir -p ~/.local/lib/
	cp lib/*     /usr/local/lib/
	cp include/* /usr/local/include/

run: pi_parallel
	LD_LIBRARY_PATH=lib bin/pi_parallel

run_slurm: pi_parallel
	LD_LIBRARY_PATH=lib srun $(SLURM_ARGS) bin/pi_parallel | tee output &


bin/test_kernel: tools/test_kernel.cpp
	@mkdir -p bin/c++
	$(CPPC) $(CPPFLAGS) -o bin/test_kernel tools/test_kernel.cpp -lOpenCL

RNG.o: include/RNG.hpp src/RNG.cpp kernel.hpp bin/test_kernel
	$(CPPC) $(CPPFLAGS) -c src/RNG.cpp -fPIC

kernel.hpp: tools/convert_kernel.py src/kernels/*.cl
	tools/convert_kernel.py src/kernels/ kernel.hpp

print: lib/librand_gpu.so examples/print.c
	@mkdir -p bin/c++
	$(CC) $(CFLAGS) -Llib -o bin/print examples/print.c -lrand_gpu
	$(CPPC) $(CPPFLAGS) -Llib -o bin/c++/print examples/print.cpp -lrand_gpu

pi: lib/librand_gpu.so examples/pi.c RNG.o
	@mkdir -p bin/c++
	@mkdir -p bin/static/c++
	$(CC) $(CFLAGS) -Llib -o bin/pi examples/pi.c -lrand_gpu
	$(CC) $(CFLAGS) -o bin/static/pi examples/pi.c RNG.o -lstdc++ -lOpenCL
	$(CPPC) $(CPPFLAGS) -Llib -o bin/c++/pi examples/pi.cpp -lrand_gpu
	$(CPPC) $(CPPFLAGS) -o bin/static/c++/pi examples/pi.cpp RNG.o -lOpenCL

pi_simple: lib/librand_gpu.so examples/pi_simple.c RNG.o
	@mkdir -p bin/c++
	@mkdir -p bin/static/c++
	$(CC) $(CFLAGS) -Llib -o bin/pi_simple examples/pi_simple.c -lrand_gpu
	$(CC) $(CFLAGS) -o bin/static/pi_simple examples/pi_simple.c RNG.o -lOpenCL -lstdc++
	$(CPPC) $(CPPFLAGS) -Llib -o bin/c++/pi_simple examples/pi_simple.cpp -lrand_gpu
	$(CPPC) $(CPPFLAGS) -o bin/static/c++/pi_simple examples/pi_simple.cpp RNG.o -lOpenCL

pi_urandom: lib/librand_gpu.so examples/pi_urandom.c RNG.o
	@mkdir -p bin/static
	$(CC) $(CFLAGS) -Llib -o bin/pi_urandom examples/pi_urandom.c -lrand_gpu
	$(CC) $(CFLAGS) -o bin/static/pi_urandom examples/pi_urandom.c RNG.o -lOpenCL -lstdc++

pi_parallel: lib/librand_gpu.so examples/pi_parallel.cpp
	@mkdir -p bin/c++
	$(CC) $(CFLAGS) -Llib -o bin/pi_parallel examples/pi_parallel.c -lm -lrand_gpu -fopenmp
	$(CPPC) $(CPPFLAGS) -Llib -o bin/c++/pi_parallel examples/pi_parallel.cpp -lm -lrand_gpu -fopenmp

coin_flip: lib/librand_gpu.so examples/coin_flip.c RNG.o
	@mkdir -p bin/c++
	@mkdir -p bin/static/c++
	$(CC) $(CFLAGS) -Llib -o bin/coin_flip examples/coin_flip.c -lm -lrand_gpu
	$(CC) $(CFLAGS) -o bin/static/coin_flip examples/coin_flip.c RNG.o -lm -lOpenCL -lstdc++

myurandom: lib/librand_gpu.so examples/myurandom.c RNG.o
	@mkdir -p bin/c++
	@mkdir -p bin/static/c++
	$(CC) $(CFLAGS) -Llib -o bin/myurandom examples/myurandom.c -lm -lrand_gpu
	$(CC) $(CFLAGS) -o bin/static/myurandom examples/myurandom.c RNG.o -lm -lOpenCL -lstdc++


algorithms: lib/librand_gpu.so test/algorithms.cpp
	@mkdir -p bin/c++
	$(CPPC) $(CPPFLAGS) -Llib -o bin/c++/algorithms test/algorithms.cpp -lrand_gpu

equality: lib/librand_gpu.so test/equality.cpp
	@mkdir -p bin/c++
	$(CPPC) $(CPPFLAGS) -Llib -o bin/c++/equality test/equality.cpp -lrand_gpu

frequency: lib/librand_gpu.so test/frequency.cpp
	@mkdir -p bin/c++
	$(CPPC) $(CPPFLAGS) -Llib -o bin/c++/frequency test/frequency.cpp -lrand_gpu -fopenmp


fastest_multiplier: lib/librand_gpu.so test/fastest_multiplier.c RNG.o
	@mkdir -p bin
	@mkdir -p bin/static
	$(CC) $(CFLAGS) -Llib -Wno-unused-value -o bin/fastest_multiplier test/fastest_multiplier.c -lrand_gpu
	$(CC) $(CFLAGS) -Wno-unused-value -o bin/static/fastest_multiplier test/fastest_multiplier.c RNG.o -lOpenCL -lstdc++

speedup_measurement: lib/librand_gpu.so test/speedup_measurement.cpp RNG.o
	@mkdir -p bin/c++
	@mkdir -p bin/static/c++
	$(CPPC) $(CPPFLAGS) -Llib -o bin/c++/speedup_measurement test/speedup_measurement.cpp -lrand_gpu
	$(CPPC) $(CPPFLAGS) -o bin/static/c++/speedup_measurement test/speedup_measurement.cpp RNG.o -lOpenCL

speedup_measurement_parallel: lib/librand_gpu.so test/speedup_measurement_parallel.cpp
	@mkdir -p bin/c++
	@mkdir -p bin/static/c++
	$(CPPC) $(CPPFLAGS) -o bin/c++/speedup_measurement_parallel test/speedup_measurement_parallel.cpp -lrand_gpu -fopenmp
	$(CPPC) $(CPPFLAGS) -o bin/c++/speedup_measurement_parallel test/speedup_measurement_parallel.cpp -lrand_gpu -fopenmp

clean:
	-rm -rf bin/*
	-rm *.o
	-rm lib/*.so
	-rm kernel.hpp
