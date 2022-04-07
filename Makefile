CC= gcc
CFLAGS= -g

test: test.o server.o
	mkdir -p bin
	$(CC) $(CFLAGS) -o bin/test test.o server.o -lOpenCL

test.o: src/test.c
	$(CC) $(CFLAGS) -c src/test.c

server.o: src/server.c
	$(CC) $(CFLAGS) -c src/server.c

clean:
	rm *.o
