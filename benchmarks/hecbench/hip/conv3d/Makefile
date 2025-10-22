###############################################################################
# User Options
###############################################################################

CC        ?= ${PROTEUS_CC}
OPTIMIZE  = yes
DEBUG     = no
PROTEUS_PATH  ?= /path/to/proteus/install
ENABLE_PROTEUS ?= no

###############################################################################
# Program name & source code list
###############################################################################

ifeq ($(ENABLE_PROTEUS),yes)
  SUFFIX = -proteus
else
  SUFFIX = -aot
endif

program = conv3d$(SUFFIX).x
source = main.cu
obj = $(source:.cu=$(SUFFIX).o)

###############################################################################
# Flags
###############################################################################

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall
LDFLAGS =

ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(ENABLE_PROTEUS),yes)
  CFLAGS += -fpass-plugin=${PROTEUS_PATH}/lib64/libProteusPass.so -DENABLE_PROTEUS
  LDFLAGS += -Wl,-rpath,${PROTEUS_PATH}/lib64 -L${PROTEUS_PATH}/lib64/ -lproteus \
             -L${ROCM_PATH}/lib -L${ROCM_PATH}/llvm/lib \
             -Wl,--start-group \
             $(shell ls ${ROCM_PATH}/llvm/lib/libclang*.a) \
             $(shell ${ROCM_PATH}/llvm/bin/llvm-config --libs) \
             -Wl,--end-group \
             $(shell ${ROCM_PATH}/llvm/bin/llvm-config --system-libs) \
             -llldCommon -llldELF
endif

###############################################################################
# Targets
###############################################################################

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cu Makefile
	$(CC) $(CFLAGS) -x hip -c $< -o $@

.PHONY: clean run

clean:
	rm -rf *.x *.o *.ll *.bc .proteus

run: $(program)
	./$(program) 32 96 256 26 26 5 100
