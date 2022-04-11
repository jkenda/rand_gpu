CC= gcc
CPPC= g++
CFLAGS= -O3 -s -Wall -Wpedantic
CPPFLAGS= --std=c++17 -O3 -g -Wall -Wpedantic
INSTALL_PATH=~/.local/lib/
SO= -Llib -l:rand_gpu.so

default: pi_compare

all: print pi pi_compare fastest_multiplier equality

install: lib
	cp lib/rand_gpu.so $(INSTALL_PATH)

print: lib/rand_gpu.so examples/print.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -o bin/print examples/print.c $(SO)

pi: lib/rand_gpu.so examples/pi.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -o bin/pi examples/pi.c $(SO)

pi_compare: lib/rand_gpu.so examples/pi_compare.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -o bin/pi_compare examples/pi_compare.c $(SO)

equality: lib/rand_gpu.so test/equality.cpp
	@mkdir -p bin
	$(CPPC) $(CPPFLAGS) -o bin/equality test/equality.cpp $(SO)

fastest_multiplier: lib/rand_gpu.so test/fastest_multiplier.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -Wno-unused-value -o bin/fastest_multiplier test/fastest_multiplier.c $(SO)

RandGPU.o: src/RandGPU.cpp
	$(CPPC) $(CPPFLAGS) -c src/RandGPU.cpp -fPIC

lib/rand_gpu.so: lib install

lib: RandGPU.o
	@mkdir -p lib
	$(CPPC) $(CPPFLAGS) -shared -o lib/rand_gpu.so RandGPU.o -lOpenCL -lpthread

clean:
	-rm *.o
	-rm lib/*.so
