###############################################################################
# CUDA Build Configuration
###############################################################################

CC             ?= ${PROTEUS_CC}
OPTIMIZE        = yes
DEBUG           = no
ARCH            = sm_90
CUDA_HOME      ?= /usr/local/cuda
PROTEUS_PATH   ?= ${PROTEUS_INSTALL_PATH}
ENABLE_PROTEUS ?= no

ifeq ($(ENABLE_PROTEUS),yes)
  SUFFIX = -proteus
else
  SUFFIX = -aot
endif

program = 3mm$(SUFFIX).x
source  = main.cu
obj     = $(source:.cu=$(SUFFIX).o)

###############################################################################
# Compiler & Linker Flags
###############################################################################

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall \
          --cuda-gpu-arch=$(ARCH) --cuda-path=${CUDA_HOME} \
          -DPROTEUS_ENABLE_CUDA -I${PROTEUS_PATH}/include

LDFLAGS = -Wl,-rpath,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64 \
          -L${CUDA_HOME}/lib64/stubs \
          -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl

ifeq ($(DEBUG),yes)
  CFLAGS  += -g
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(ENABLE_PROTEUS),yes)
  LIBLLVM_PATH = $(dir $(realpath $(CC)))../lib
  CFLAGS  += -fpass-plugin=${PROTEUS_PATH}/lib64/libProteusPass.so \
             -DENABLE_PROTEUS -DPROTEUS_JIT
  LDFLAGS += -Wl,-rpath,${PROTEUS_PATH}/lib64 -L${PROTEUS_PATH}/lib64/ \
             -lproteus -lnvptxcompiler_static \
             -Wl,-rpath,$(LIBLLVM_PATH) -L$(LIBLLVM_PATH) \
             -lclang-cpp -lLLVM
endif

###############################################################################
# Build Targets
###############################################################################

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean run

clean:
	rm -rf *.x *.o .proteus

run: $(program)
	./$(program) --no-verify --N 8192 --kernel gpu_regtiled
