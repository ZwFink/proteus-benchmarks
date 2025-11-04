#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        ?= ${PROTEUS_CC}
OPTIMIZE  = yes
DEBUG     = no
ARCH      ?= sm_90
LAUNCHER  =
PROTEUS_PATH    ?=/path/to/proteus/install
ENABLE_PROTEUS ?= no
CUDA_HOME ?= /usr/local/cuda

# Simple timing helper; override TIME env var to change format/tool.
TIME ?= /usr/bin/env time -f '%E real => $@'

#===============================================================================
# Program name & source code list
#===============================================================================

ifeq ($(ENABLE_PROTEUS),yes)
  SUFFIX = "-proteus"
else
  SUFFIX = "-aot"
endif

program = sw4ck$(SUFFIX).x

source = main.cu curvilinear4sg.cu

obj = $(source:.cu=$(SUFFIX).o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 --cuda-gpu-arch=$(ARCH) --cuda-path=${CUDA_HOME} \
          -DPROTEUS_ENABLE_CUDA

# Linker Flags
LDFLAGS = -Wl,-rpath,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64 \
          -L${CUDA_HOME}/lib64/stubs \
	  -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(ENABLE_PROTEUS),yes)
  LIBLLVM_PATH=$(dir $(realpath $(CC)))../lib
  CFLAGS += -fpass-plugin=${PROTEUS_PATH}/lib64/libProteusPass.so -DENABLE_PROTEUS
  LDFLAGS += -Wl,-rpath,${PROTEUS_PATH}/lib64 -L${PROTEUS_PATH}/lib64/ -lproteus \
             -lnvptxcompiler_static \
             -Wl,-rpath,$(LIBLLVM_PATH) -L$(LIBLLVM_PATH) -lclang-cpp -lLLVM
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	@$(TIME) -f 'link %E real' $(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main$(SUFFIX).o: main.cu utils.cpp utils.h
	@$(TIME) -f 'compile %E real' $(CC) $(CFLAGS) -c $< -o $@

curvilinear4sg$(SUFFIX).o : curvilinear4sg.cu \
       	           kernel1.cpp \
	           kernel2.cpp \
       	           kernel3.cpp \
	           kernel4.cpp \
	           kernel5.cpp \
	           utils.h \
	           curvilinear4sg.h
	@$(TIME) -f 'compile %E real' $(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.x *.o
	rm -rf .proteus

run: $(program)
	$(LAUNCHER) ./$(program) sw4ck.in 100
