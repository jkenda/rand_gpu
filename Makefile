CC= gcc
#CFLAGS= -Wall -Wpedantic -g
CFLAGS= -Wall -Wpedantic -O2 -s -DNDEBUG

equality: rand_gpu.o examples/equality.c
	-mkdir -p bin
	$(CC) $(CFLAGS) -o bin/equality examples/equality.c rand_gpu.o -lOpenCL -lpthread

print: rand_gpu.o examples/print.c
	-mkdir -p bin
	$(CC) $(CFLAGS) -o bin/print examples/print.c rand_gpu.o -lOpenCL -lpthread

pi: rand_gpu.o examples/pi.c
	-mkdir -p bin
	$(CC) $(CFLAGS) -o bin/pi examples/pi.c rand_gpu.o -lOpenCL -lpthread

pi_compare: rand_gpu.o examples/pi_compare.c
	-mkdir -p bin
	$(CC) $(CFLAGS) -o bin/pi_compare examples/pi_compare.c rand_gpu.o -lOpenCL -lpthread

fastest_multiplier: rand_gpu.o test/fastest_multiplier.c
	-mkdir -p bin
	$(CC) $(CFLAGS) -o bin/fastest_multiplier test/fastest_multiplier.c rand_gpu.o -lOpenCL -lpthread

rand_gpu.o: src/rand_gpu.c src/util.h
	$(CC) $(CFLAGS) -c src/rand_gpu.c

clean:
	-rm *.o
