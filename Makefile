CC= gcc
CFLAGS= -Wall -Wpedantic -O3

test: rand_gpu.o test/test.c
	mkdir -p bin
	$(CC) $(CFLAGS) -o bin/test test/test.c rand_gpu.o -lOpenCL

pi: rand_gpu.o test/pi.c
	$(CC) $(CFLAGS) -o bin/pi test/pi.c rand_gpu.o -lOpenCL

rand_gpu.o: src/rand_gpu.c
	$(CC) $(CFLAGS) -c src/rand_gpu.c

clean:
	rm *.o
