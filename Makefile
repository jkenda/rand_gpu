CC= gcc
CFLAGS= -O3

test: server.o src/test.c
	mkdir -p bin
	$(CC) $(CFLAGS) -o bin/test src/test.c server.o -lOpenCL

pi: server.o src/pi.c
	$(CC) $(CFLAGS) -o bin/pi src/pi.c server.o -lOpenCL

server.o: src/server.c
	$(CC) $(CFLAGS) -c src/server.c

clean:
	rm *.o
