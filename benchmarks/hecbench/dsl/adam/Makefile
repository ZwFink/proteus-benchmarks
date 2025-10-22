#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = ${PROTEUS_CC}
OPTIMIZE  = yes
DEBUG     = no
PROTEUS_PATH  ?= /path/to/proteus/install

#===============================================================================
# Program name & source code list
#===============================================================================

SUFFIX = -proteus

program = adam$(SUFFIX).x

source = adam.cpp
obj = $(source:.cpp=$(SUFFIX).o)

#===============================================================================
# Backend Selection
#===============================================================================

# Select GPU backend: hip | cuda (default hip)
GPU_BACKEND ?= hip

# Optionally point to an LLVM install (preferred over ROCm's LLVM if set)
LLVM_INSTALL_DIR ?= ${ROCM_PATH}/llvm
LLVM_CONFIG ?= ${LLVM_INSTALL_DIR}/bin/llvm-config

#===============================================================================
# Flags per-backend
#===============================================================================

# HIP (ROCm) toolchain
OFFLOAD_ARCH ?= gfx942
HIPFLAGS := --offload-arch=${OFFLOAD_ARCH}
# Defer shell expansion; only evaluated when HIP backend is selected
HIP_CFLAGS = $(EXTRA_CFLAGS) -std=c++17 -Wall -DPROTEUS_ENABLE_HIP \
	-I${PROTEUS_PATH}/include -I${LLVM_INSTALL_DIR}/include/ -I${ROCM_PATH}/include \
	$(shell ${LLVM_CONFIG} --cxxflags | sed 's/-fno-exceptions//g') \
	-fexceptions $(HIPFLAGS)
HIP_LDFLAGS = -L${ROCM_PATH}/lib -L${LLVM_INSTALL_DIR}/lib \
	$(shell ${LLVM_CONFIG} --libs) \
	$(shell ${LLVM_CONFIG} --system-libs) \
	-llldCommon -llldELF -lamdhip64 -lhiprtc -lhiprtc-builtins -Wl,-rpath,${ROCM_PATH}/lib
HIP_LANG_FLAG := -x hip

# CUDA toolchain
CUDA_PATH ?= ${CUDA_HOME}
# Defer shell expansion; use LLVM_CONFIG when building for CUDA as well
CUDA_CFLAGS = $(EXTRA_CFLAGS) -std=c++17 -Wall -DPROTEUS_ENABLE_CUDA \
	-I${PROTEUS_PATH}/include -I${CUDA_PATH}/include -I${LLVM_INSTALL_DIR}/include \
	$(shell ${LLVM_CONFIG} --cxxflags | sed 's/-fno-exceptions//g') \
	-fexceptions
CUDA_LDFLAGS = -L${CUDA_PATH}/lib64 -Wl,-rpath,${CUDA_PATH}/lib64 \
	-L${LLVM_INSTALL_DIR}/lib -Wl,-rpath,${LLVM_INSTALL_DIR}/lib \
	$(shell ${LLVM_CONFIG} --libs) \
	$(shell ${LLVM_CONFIG} --system-libs) \
	-lcudart -lnvrtc
CUDA_LANG_FLAG :=

# Backend switch
ifeq ($(GPU_BACKEND),hip)
	CFLAGS := $(HIP_CFLAGS)
	LDFLAGS := $(HIP_LDFLAGS)
	LANG_FLAG := $(HIP_LANG_FLAG)
else ifeq ($(GPU_BACKEND),cuda)
	CFLAGS := $(CUDA_CFLAGS)
	LDFLAGS := $(CUDA_LDFLAGS)
	LANG_FLAG := $(CUDA_LANG_FLAG)
else
	$(error Unknown GPU_BACKEND '$(GPU_BACKEND)'; expected hip or cuda)
endif

# Debug Flags
ifeq ($(DEBUG),yes)
	CFLAGS += -g
	LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3
endif

# Always link against Proteus runtime for DSL builds
LDFLAGS += -Wl,-rpath,${PROTEUS_PATH}/lib64 -L${PROTEUS_PATH}/lib64/ -lproteus -pthread

# CUDA driver libs must appear after libproteus (static) on link line
ifeq ($(GPU_BACKEND),cuda)
	LDFLAGS += -L/usr/lib64 -Wl,-rpath,/usr/lib64 -lcuda -lnvptxcompiler_static
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cpp Makefile
	$(CC) $(CFLAGS) $(LANG_FLAG) -c $< -o $@

clean:
	rm -rf *.x *.o .proteus

run: $(program)
	./$(program) 10000 200 100
