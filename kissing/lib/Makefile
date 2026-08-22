# Build the C tools.  The Riesz optimiser needs OpenBLAS + OpenMP; the rest
# match the historical gcc -O2 -lm command line.  Prefer a system OpenBLAS,
# then fall back to scipy-openblas32 (whose public symbols are prefixed).
CC = gcc
OMPFLAGS ?= -O3 -fopenmp
CFLAGS ?= -O2 -lm

OPENBLAS_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
OPENBLAS_LIBS := $(shell pkg-config --libs openblas 2>/dev/null)

ifeq ($(strip $(OPENBLAS_LIBS)),)
SCIPY_OPENBLAS_INC := $(shell python3 -c "import scipy_openblas32 as s; print(s.get_include_dir())" 2>/dev/null)
SCIPY_OPENBLAS_LIB := $(shell python3 -c "import scipy_openblas32 as s; print(s.get_lib_dir())" 2>/dev/null)
ifeq ($(strip $(SCIPY_OPENBLAS_LIB)),)
$(error OpenBLAS not found. Install a system OpenBLAS or run: python3 -m pip install scipy-openblas32)
endif
RIESZ_CFLAGS := -I$(SCIPY_OPENBLAS_INC) -include cblas.h \
	-Dcblas_dgemm=scipy_cblas_dgemm -Dcblas_dscal=scipy_cblas_dscal \
	-Dcblas_ddot=scipy_cblas_ddot -Dcblas_daxpy=scipy_cblas_daxpy \
	-Dcblas_dcopy=scipy_cblas_dcopy -Dcblas_dnrm2=scipy_cblas_dnrm2 \
	-Dopenblas_set_num_threads=scipy_openblas_set_num_threads
LIBS_RIESZ := -L$(SCIPY_OPENBLAS_LIB) -Wl,-rpath,$(SCIPY_OPENBLAS_LIB) -lscipy_openblas -lm
else
RIESZ_CFLAGS := $(OPENBLAS_CFLAGS)
LIBS_RIESZ := $(OPENBLAS_LIBS) -lm
endif

.PHONY: all riesz-tools verify-tools clean

all: riesz-tools verify-tools

riesz-tools: riesz riesz2

riesz: riesz.c
	$(CC) $(OMPFLAGS) $(RIESZ_CFLAGS) -o $@ $< $(LIBS_RIESZ)

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
