CC= gcc
CFLAGS= -Wall -Wpedantic -O3

test: rand_gpu.o examples/test.c
	mkdir -p bin
	$(CC) $(CFLAGS) -o bin/test examples/test.c rand_gpu.o -lOpenCL

pi: rand_gpu.o examples/pi.c
	$(CC) $(CFLAGS) -o bin/pi examples/pi.c rand_gpu.o -lOpenCL

rand_gpu.o: src/rand_gpu.c src/util.h
	$(CC) $(CFLAGS) -c src/rand_gpu.c

clean:
	rm *.o
