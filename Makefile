ifeq ($(OS),Windows_NT)
#	TODO
#	CC=gcc -Ofast -ggdb -Wall -Wpedantic
#	CXX=g++ --std=c++17 -Ofast -ggdb -Wall -Wpedantic
#	LOPENCL=-lOpenCL
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
		CC=gcc             -Ofast -ggdb -Wall -Wpedantic
		CXX=g++ --std=c++17 -Ofast -ggdb -Wall -Wpedantic
		CLCXX:=$(CXX)
		LOPENCL=-lOpenCL
    endif
    ifeq ($(UNAME_S),Darwin)
		CC=clang               -Ofast -ggdb -Wall -Wpedantic
		CXX=clang++ --std=c++17 -Ofast -ggdb -Wall -Wpedantic
		CLCXX=clang++ -framework OpenCL --std=c++17 -Ofast -ggdb -Wall -Wpedantic
		LOPENCL=
    endif
endif


default: lib

all: print pi pi_simple pi_urandom coin_flip myurandom algorithms equality frequency fastest_multiplier speedup_measurement pi_parallel
lib: lib/librand_gpu.so

lib/librand_gpu.so: RNG.o
	@mkdir -p lib
	$(CLCXX) -shared -o lib/librand_gpu.so RNG.o $(LOPENCL) -flto
	-bin/test_kernel

install: lib/librand_gpu.so
#	@mkdir -p ~/.local/lib/
	cp lib/*     /usr/local/lib/
	cp include/* /usr/local/include/


bin/test_kernel: tools/test_kernel.cpp
	@mkdir	-p bin/c++
	$(CLCXX)	-o bin/test_kernel tools/test_kernel.cpp $(LOPENCL)

RNG.o: include/RNG.hpp src/RNG.cpp kernel.hpp bin/test_kernel
	$(CXX) -c src/RNG.cpp -fPIC

kernel.hpp: tools/convert_kernel.py src/kernels/*.cl
	tools/convert_kernel.py src/kernels/ kernel.hpp

print: lib/librand_gpu.so examples/print.c
	@mkdir	-p bin/c++
	$(CC)	-Llib -o bin/print examples/print.c -lrand_gpu
	$(CXX)	-Llib -o bin/c++/print examples/print.cpp -lrand_gpu

pi: lib/librand_gpu.so examples/pi.c
	@mkdir	-p bin/c++
	@mkdir	-p bin/static/c++
	$(CC)	-Llib -o bin/pi examples/pi.c -lrand_gpu
	$(CC)	-o bin/static/pi examples/pi.c lib/librand_gpu.so
	$(CXX)	-Llib -o bin/c++/pi examples/pi.cpp -lrand_gpu
	$(CXX)	-o bin/static/c++/pi examples/pi.cpp lib/librand_gpu.so

pi_simple: lib/librand_gpu.so examples/pi_simple.c
	@mkdir	-p bin/c++
	@mkdir	-p bin/static/c++
	$(CC)	-Llib -o bin/pi_simple examples/pi_simple.c -lrand_gpu
	$(CC)	-o bin/static/pi_simple examples/pi_simple.c lib/librand_gpu.so
	$(CXX)	-Llib -o bin/c++/pi_simple examples/pi_simple.cpp -lrand_gpu
	$(CXX)	-o bin/static/c++/pi_simple examples/pi_simple.cpp lib/librand_gpu.so

pi_urandom: lib/librand_gpu.so examples/pi_urandom.c
	@mkdir	-p bin/static
	$(CC)	-Llib -o bin/pi_urandom examples/pi_urandom.c -lrand_gpu
	$(CC)	-o bin/static/pi_urandom examples/pi_urandom.c lib/librand_gpu.so

pi_parallel: lib/librand_gpu.so examples/pi_parallel.cpp
	@mkdir	-p bin/c++
	$(CC)	-Llib -o bin/pi_parallel examples/pi_parallel.c -lm -lrand_gpu -fopenmp
	$(CXX)	-Llib -o bin/c++/pi_parallel examples/pi_parallel.cpp -lm -lrand_gpu -fopenmp

coin_flip: lib/librand_gpu.so examples/coin_flip.c
	@mkdir	-p bin/c++
	@mkdir	-p bin/static/c++
	$(CC)	-Llib -o bin/coin_flip examples/coin_flip.c -lm -lrand_gpu
	$(CC)	-o bin/static/coin_flip examples/coin_flip.c lib/librand_gpu.so

myurandom: lib/librand_gpu.so examples/myurandom.c
	@mkdir	-p bin/c++
	@mkdir	-p bin/static/c++
	$(CC)	-Llib -o bin/myurandom examples/myurandom.c -lm -lrand_gpu
	$(CC)	-o bin/static/myurandom examples/myurandom.c lib/librand_gpu.so -lm
	

algorithms: lib/librand_gpu.so test/algorithms.cpp
	@mkdir	-p bin/c++
	$(CXX)	-Llib -o bin/c++/algorithms test/algorithms.cpp -lrand_gpu

equality: lib/librand_gpu.so test/equality.cpp
	@mkdir	-p bin/c++
	$(CXX)	-Llib -o bin/c++/equality test/equality.cpp -lrand_gpu

frequency: lib/librand_gpu.so test/frequency.cpp
	@mkdir	-p bin/c++
	$(CXX)	-Llib -o bin/c++/frequency test/frequency.cpp -lrand_gpu


fastest_multiplier: lib/librand_gpu.so test/fastest_multiplier.c
	@mkdir	-p bin
	@mkdir	-p bin/static
	$(CC)	-Llib -Wno-unused-value -o bin/fastest_multiplier test/fastest_multiplier.c -lrand_gpu
	$(CC)	-Wno-unused-value -o bin/static/fastest_multiplier test/fastest_multiplier.c lib/librand_gpu.so

speedup_measurement: lib/librand_gpu.so test/speedup_measurement.cpp
	@mkdir	-p bin/c++
	@mkdir	-p bin/static/c++
	$(CXX)	-Llib -o bin/c++/speedup_measurement test/speedup_measurement.cpp -lrand_gpu
	$(CXX)	-o bin/static/c++/speedup_measurement test/speedup_measurement.cpp lib/librand_gpu.so

speedup_measurement_parallel: lib/librand_gpu.so test/speedup_measurement_parallel.cpp
	@mkdir  -p bin/c++
	@mkdir  -p bin/static/c++
	$(CXX)	-o bin/c++/speedup_measurement_parallel test/speedup_measurement_parallel.cpp -lrand_gpu -fopenmp
	$(CXX)	-o bin/c++/speedup_measurement_parallel test/speedup_measurement_parallel.cpp -lrand_gpu -fopenmp

clean:
	-rm -rf bin/
	-rm -f  *.o
	-rm -rf lib/
	-rm -f  kernel.hpp
