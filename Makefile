
CC ?= gcc

CFLAGS += -std=c99 -Wall -g
LDFLAG := -g

.PHONY: all clean

all: synth

synth: synth.o fft.o
	$(CC) $(LDFLAG) $^ -o $@

synth.o: synth.c fft.h
fft.o: fft.c fft.h

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@
