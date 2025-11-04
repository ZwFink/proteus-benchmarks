#===============================================================================
# User Options
#===============================================================================

CC ?= ${PROTEUS_CC}
OPTIMIZE    = yes
DEBUG       = no
PROFILE     = no
PROTEUS_PATH    ?=/path/to/proteus/install
ENABLE_PROTEUS ?= no

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

program = rsbench$(SUFFIX).x

source = \
main.cpp \
simulation.cpp\
io.cpp \
init.cpp \
material.cpp \
utils.cpp

obj = $(source:.cpp=$(SUFFIX).o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS :=

CFLAGS += -std=c++14

# Linker Flags
LDFLAGS =

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -G
  LDFLAGS += -g -G
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS += -pg
  LDFLAGS  += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(ENABLE_PROTEUS),yes)
  CFLAGS += -DENABLE_PROTEUS -fpass-plugin=${PROTEUS_PATH}/lib64/libProteusPass.so
  LDFLAGS += -Wl,-rpath,${PROTEUS_PATH}/lib64 -L${PROTEUS_PATH}/lib64/ -lproteus \
          -L${ROCM_PATH}/lib -L${ROCM_PATH}/llvm/lib \
          -Wl,--start-group \
          $(shell ${ROCM_PATH}/llvm/bin/llvm-config --libs) \
          -Wl,--end-group \
          $(shell ${ROCM_PATH}/llvm/bin/llvm-config --system-libs) \
          -llldCommon -llldELF
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) rsbench.h Makefile
	@$(TIME) -f 'link %E real' $(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%$(SUFFIX).o: %.cpp rsbench.h Makefile
	@$(TIME) -f 'compile %E real' $(CC) $(CFLAGS) -x hip -c $< -o $@

clean:
	rm -rf *.x *.o *.ll *.bc
	rm -rf .proteus
edit:
	vim -p $(source) rsbench.h

run:
	./rsbench
