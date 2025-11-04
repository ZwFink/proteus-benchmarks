#===============================================================================
# User Options
#===============================================================================

CC        ?= ${PROTEUS_CC}
OPTIMIZE  = yes
DEBUG     = no
ARCH      ?= sm_90
LAUNCHER  =
PROTEUS_PATH  ?= ${PROTEUS_INSTALL_PATH}
ENABLE_PROTEUS ?= no
CUDA_HOME ?= /usr/local/cuda

# Simple timing helper; override TIME env var to change format/tool.
TIME ?= /usr/bin/env time -f '%E real => $@'

#===============================================================================
# Program name & source code list
#===============================================================================

ifeq ($(ENABLE_PROTEUS),yes)
  SUFFIX = -proteus
else
  SUFFIX = -aot
endif

program = conv3d$(SUFFIX).x
source = main.cu
obj = $(source:.cu=$(SUFFIX).o)

#===============================================================================
# Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall --cuda-gpu-arch=$(ARCH) \
          --cuda-path=${CUDA_HOME} -DPROTEUS_ENABLE_CUDA

LDFLAGS = -Wl,-rpath,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64 \
          -L${CUDA_HOME}/lib64/stubs \
          -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl

ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(ENABLE_PROTEUS),yes)
  LIBLLVM_PATH=$(dir $(realpath $(CC)))../lib
  CFLAGS += -fpass-plugin=${PROTEUS_PATH}/lib64/libProteusPass.so \
            -DENABLE_PROTEUS -DPROTEUS_JIT
  LDFLAGS += -Wl,-rpath,${PROTEUS_PATH}/lib64 -L${PROTEUS_PATH}/lib64/ \
             -lproteus -lnvptxcompiler_static \
             -Wl,-rpath,$(LIBLLVM_PATH) -L$(LIBLLVM_PATH) -lclang-cpp -lLLVM
endif

#===============================================================================
# Targets
#===============================================================================

$(program): $(obj) Makefile
	@$(TIME) -f 'link %E real' $(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cu Makefile
	@$(TIME) -f 'compile %E real' $(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean run

clean:
	rm -rf *.x *.o *.ll *.bc .proteus

run: $(program)
	$(LAUNCHER) ./$(program) 32 96 256 26 26 5 100
