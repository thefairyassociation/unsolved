# Build the C tools.  The Riesz optimiser needs OpenBLAS + OpenMP; the rest
# match the historical gcc -O2 -lm command line.
CC = gcc
OMPFLAGS ?= -O3 -fopenmp
LIBS_RIESZ ?= -lopenblas -lm
CFLAGS ?= -O2 -lm

.PHONY: all riesz-tools verify-tools clean

all: riesz-tools verify-tools

riesz-tools: riesz riesz2

riesz: riesz.c
	$(CC) $(OMPFLAGS) -o $@ $< $(LIBS_RIESZ)

riesz2: riesz
	cp -f riesz riesz2

verify-tools: mis8 clique gclique addable opt opt2 signmis

mis8: mis8.c
	$(CC) -O2 -o $@ $< -lm
clique: clique.c
	$(CC) -O2 -o $@ $< -lm
gclique: gclique.c
	$(CC) -O2 -o $@ $< -lm
addable: addable.c
	$(CC) -O2 -o $@ $< -lm
opt: opt.c
	$(CC) -O2 -o $@ $< -lm
opt2: opt
	cp -f opt opt2
signmis: signmis.c
	$(CC) -O2 -o $@ $< -lm

clean:
	rm -f riesz riesz2
