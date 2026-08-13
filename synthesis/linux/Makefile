CC ?= gcc
PYTHON ?= python3
CFLAGS ?= -O2 -Wall -Wextra -std=c11 -D_DEFAULT_SOURCE
CPPFLAGS += -Iinclude
LDLIBS += -lm -lrt

TARGET := microgpt_hps_runtime
SRCS := src/microgpt_hps_runtime.c src/tinylpu_hps_mmio.c

.PHONY: all clean image runtime

all: runtime

runtime: $(TARGET)

image:
	$(PYTHON) ../scripts/export_microgpt_hps_headers.py

$(TARGET): $(SRCS) include/tinylpu_hps_mmio.h include/microgpt_hps_image.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $(SRCS) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)
