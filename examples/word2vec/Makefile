src_c := $(wildcard *.c)
src_s := $(patsubst %.c,%.s,$(src_c))
lib_so := $(patsubst %.c,lib%.so,$(src_c))
cflags := 
cflags += -Wall
cflags += -Werror
cflags += -DNDEBUG
cflags += -O3
cflags += -march=native
cflags += -fopenmp-simd
cflags += -ffast-math
cflags += -std=gnu99
cflags += -fopt-info-vec-optimized
#cflags += -fopt-info-vec-missed

all : $(lib_so) $(src_s)

$(lib_so) : lib%.so : %.c
	gcc -shared -fPIC $(cflags) $< -o $@

$(src_s) : %.s : %.c
	gcc -shared -fPIC $(cflags) $< -o $@ -S

