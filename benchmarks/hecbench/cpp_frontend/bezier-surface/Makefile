###############################################################################
# User Options
###############################################################################

# Compiler can be set below, or via environment variable
CC              = ${PROTEUS_CC}
OPTIMIZE        = yes
DEBUG           = no
PROTEUS_PATH    ?= /path/to/proteus/install
GPU_BACKEND     ?= hip
OFFLOAD_ARCH    ?= gfx942
LLVM_INSTALL_DIR ?= ${ROCM_PATH}/llvm
LLVM_CONFIG     ?= ${LLVM_INSTALL_DIR}/bin/llvm-config
CUDA_PATH       ?= ${CUDA_HOME}
CUDA_ARCH       ?= sm_90

###############################################################################
# Program name & source code list
###############################################################################

program = bezier-surface-cpp-frontend-proteus.x

source = main.cpp
obj = $(source:.cpp=.o)

###############################################################################
# Flags
###############################################################################

HIP_CLANG_LIBS = -lclangAPINotes -lclangARCMigrate -lclangAST -lclangASTMatchers -lclangAnalysis \
-lclangAnalysisFlowSensitive -lclangAnalysisFlowSensitiveModels -lclangApplyReplacements -lclangBasic -lclangChangeNamespace \
-lclangCodeGen -lclangCrossTU -lclangDaemon -lclangDaemonTweaks -lclangDependencyScanning \
-lclangDirectoryWatcher -lclangDoc -lclangDriver -lclangDynamicASTMatchers -lclangEdit \
-lclangExtractAPI -lclangFormat -lclangFrontend -lclangFrontendTool -lclangHandleCXX \
-lclangHandleLLVM -lclangIncludeCleaner -lclangIncludeFixer -lclangIncludeFixerPlugin -lclangIndex \
-lclangIndexSerialization -lclangInstallAPI -lclangInterpreter -lclangLex -lclangMove \
-lclangParse -lclangPseudo -lclangPseudoCLI -lclangPseudoCXX -lclangPseudoGrammar \
-lclangQuery -lclangReorderFields -lclangRewrite -lclangRewriteFrontend -lclangSema \
-lclangSerialization -lclangStaticAnalyzerCheckers -lclangStaticAnalyzerCore -lclangStaticAnalyzerFrontend -lclangSupport \
-lclangTidy -lclangTidyAbseilModule -lclangTidyAlteraModule -lclangTidyAndroidModule -lclangTidyBoostModule \
-lclangTidyBugproneModule -lclangTidyCERTModule -lclangTidyConcurrencyModule -lclangTidyCppCoreGuidelinesModule -lclangTidyDarwinModule \
-lclangTidyFuchsiaModule -lclangTidyGoogleModule -lclangTidyHICPPModule -lclangTidyLLVMLibcModule -lclangTidyLLVMModule \
-lclangTidyLinuxKernelModule -lclangTidyMPIModule -lclangTidyMain -lclangTidyMiscModule -lclangTidyModernizeModule \
-lclangTidyObjCModule -lclangTidyOpenMPModule -lclangTidyPerformanceModule -lclangTidyPlugin -lclangTidyPortabilityModule \
-lclangTidyReadabilityModule -lclangTidyUtils -lclangTidyZirconModule -lclangTooling -lclangToolingASTDiff \
-lclangToolingCore -lclangToolingInclusions -lclangToolingInclusionsStdlib -lclangToolingRefactoring -lclangToolingSyntax \
-lclangTransformer -lclangdMain -lclangdRemoteIndex -lclangdSupport

CUDA_CLANG_LIBS = -lclang-cpp

COMMON_CFLAGS = $(EXTRA_CFLAGS) -std=c++17 -Wall \
	-I${PROTEUS_PATH}/include \
	-I${LLVM_INSTALL_DIR}/include \
	-I${PWD}/../lib/ \
	$(shell ${LLVM_CONFIG} --cxxflags | sed 's/-fno-exceptions//g') \
	-fexceptions

COMMON_LDFLAGS = -Wl,-rpath,${PROTEUS_PATH}/lib64 -L${PROTEUS_PATH}/lib64/ -lproteus -pthread \
	-L${LLVM_INSTALL_DIR}/lib -Wl,-rpath,${LLVM_INSTALL_DIR}/lib \
	$(shell ${LLVM_CONFIG} --libs) \
	$(shell ${LLVM_CONFIG} --system-libs)

HIPFLAGS := --offload-arch=${OFFLOAD_ARCH}
HIP_CFLAGS = $(COMMON_CFLAGS) -DPROTEUS_ENABLE_HIP \
	-I${ROCM_PATH}/include \
	$(HIPFLAGS)
HIP_LDFLAGS = $(COMMON_LDFLAGS) \
	-L${ROCM_PATH}/lib -Wl,-rpath,${ROCM_PATH}/lib \
	-llldCommon -llldELF -lamdhip64 -lhiprtc -lhiprtc-builtins \
	$(HIP_CLANG_LIBS)
HIP_LANG_FLAG := -x hip

CUDA_CFLAGS = $(COMMON_CFLAGS) -DPROTEUS_ENABLE_CUDA \
	-I${CUDA_PATH}/include
CUDA_LDFLAGS = $(COMMON_LDFLAGS) \
	-L${CUDA_PATH}/lib64 -Wl,-rpath,${CUDA_PATH}/lib64 \
	-L/usr/lib64 -Wl,-rpath,/usr/lib64 \
	-lcudart -lnvrtc -lcuda -lnvptxcompiler_static -lnvJitLink \
	$(CUDA_CLANG_LIBS)
CUDA_LANG_FLAG :=

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
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

###############################################################################
# Targets
###############################################################################

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) $(LANG_FLAG) -c $< -o $@

.PHONY: clean run

clean:
	rm -rf $(program) *.o .proteus

# Default run (matches defaults in main.cpp)
run: $(program)
	./$(program) -n 8192
