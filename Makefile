# TinyLPU Native C Hardware Driver Makefile
# Supports both native x86 testing and ARM GCC cross-compilation for DE1-SoC

CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -Iinclude -Isrc

TARGET = lpu_driver
SRC = src/lpu_driver.c

all: headers $(TARGET)

headers:
	python scripts/export_c_headers.py

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET) include/lpu_vliw.h include/lpu_weights.h

.PHONY: all clean headers
