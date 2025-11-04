MPI_FLAGS =

CC?= ${PROTEUS_CC}
PROTEUS_PATH ?= /path/to/proteus/install
ENABLE_PROTEUS ?= no

# Simple timing helper; override TIME env var to change format/tool.
TIME ?= /usr/bin/env time -f '%E real => $@'
FLAGS	=
DFLAGS	= $(MPI_FLAGS) -lineinfo
RFLAGS 	= $(MPI_FLAGS) -O3 -DNDEBUG

ifeq ($(ENABLE_PROTEUS),yes)
  SUFFIX = "-proteus"
else
  SUFFIX = "-aot"
endif


#SILO_INCLUDES := /usr/local/silo-4.8/include
#SILO_LIBS := /usr/local/silo-4.8/lib

LINKFLAGS =
#LINKFLAGS += -L$(SILO_LIBS) -lsilo

#INC_SILO:= -I$(SILO_INCLUDES)

# release flags
FLAGS += $(RFLAGS)

ifeq ($(ENABLE_PROTEUS),yes)
  FLAGS += -DENABLE_PROTEUS -fpass-plugin=${PROTEUS_PATH}/lib64/libProteusPass.so
  LINKFLAGS += -Wl,-rpath,${PROTEUS_PATH}/lib64 -L${PROTEUS_PATH}/lib64/ -lproteus \
          -L${ROCM_PATH}/lib -L${ROCM_PATH}/llvm/lib \
          -Wl,--start-group \
          $(shell ${ROCM_PATH}/llvm/bin/llvm-config --libs) \
          -Wl,--end-group \
          $(shell ${ROCM_PATH}/llvm/bin/llvm-config --system-libs) \
          -llldCommon -llldELF
endif

lulesh$(SUFFIX).x: allocator$(SUFFIX).o lulesh$(SUFFIX).o lulesh-comms$(SUFFIX).o lulesh-comms-gpu$(SUFFIX).o
	@$(TIME) -f 'link %E real' $(CC) $(LINKFLAGS) allocator$(SUFFIX).o lulesh$(SUFFIX).o lulesh-comms$(SUFFIX).o lulesh-comms-gpu$(SUFFIX).o -o $@

allocator$(SUFFIX).o: allocator.cu vector.h
	@$(TIME) -f 'compile %E real' $(CC) $(FLAGS) allocator.cu -I ./ -c -o $@

lulesh$(SUFFIX).o: lulesh.cu util.h vector.h allocator.h
	@$(TIME) -f 'compile %E real' $(CC) $(FLAGS) lulesh.cu -I ./  $(INC_SILO) -c -o $@

lulesh-comms$(SUFFIX).o: lulesh-comms.cu
	@$(TIME) -f 'compile %E real' $(CC) $(FLAGS) lulesh-comms.cu -I ./ -c -o $@

lulesh-comms-gpu$(SUFFIX).o: lulesh-comms-gpu.cu
	@$(TIME) -f 'compile %E real' $(CC) $(FLAGS) lulesh-comms-gpu.cu -I ./ -c -o $@

clean:
	rm -rf *.x *.o xyz.asc regNumList.txt
	rm -rf .proteus
