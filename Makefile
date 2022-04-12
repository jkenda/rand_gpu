CC= gcc
CPPC= g++
CFLAGS= -O3 -s -Wall -Wpedantic
CPPFLAGS= --std=c++17 -O3 -g -Wall -Wpedantic
INSTALL_PATH= $(HOME)/.local/lib/

default: lib/librand_gpu.so

all: print pi pi_compare fastest_multiplier equality
lib/librand_gpu.so: lib install

lib: RandGPU.o
	@mkdir -p lib
	$(CPPC) $(CPPFLAGS) -shared -o lib/librand_gpu.so RandGPU.o -lOpenCL -lpthread

install: lib
	@mkdir -p $(INSTALL_PATH)
	cp lib/librand_gpu.so $(INSTALL_PATH)

RandGPU.o: src/RandGPU.cpp
	$(CPPC) $(CPPFLAGS) -c src/RandGPU.cpp -fPIC

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
