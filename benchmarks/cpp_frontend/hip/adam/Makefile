###############################################################################
# User Options
###############################################################################

# Compiler can be set via environment variable
CC              = ${PROTEUS_CC}
OPTIMIZE        = yes
DEBUG           = no
PROFILE         = no
PROTEUS_PATH    ?= /path/to/proteus/install
ENABLE_PROTEUS  ?= yes

###############################################################################
# Program name & source code list
###############################################################################

ifeq ($(ENABLE_PROTEUS),yes)
  SUFFIX = -proteus
else
  SUFFIX = -aot
endif

program = adam-cpp$(SUFFIX).x

source = \
  adam.cpp

obj = $(source:.cpp=.o)

###############################################################################
# Flags
###############################################################################

CLANG_LIBS = -lclangAPINotes -lclangARCMigrate -lclangAST -lclangASTMatchers -lclangAnalysis \
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

# Standard Flags
OFFLOAD_ARCH ?= gfx942
HIPFLAGS := --offload-arch=${OFFLOAD_ARCH}
CFLAGS := $(EXTRA_CFLAGS) -DMKX=4 -Wall -I${ROCM_PATH}/llvm/include/ -I${ROCM_PATH}/include \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --cxxflags) -fexceptions $(HIPFLAGS) \
  -I${PWD}/../../lib/

# Linker Flags
LDFLAGS = -L${ROCM_PATH}/lib -L${ROCM_PATH}/llvm/lib \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --libs) \
	$(shell ${ROCM_PATH}/llvm/bin/llvm-config --system-libs) \
	-llldCommon -llldELF -lamdhip64 -Wl,-rpath,${ROCM_PATH}/lib $(CLANG_LIBS)

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g
  LDFLAGS += -g
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS  += -pg
  LDFLAGS += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

# Proteus Annotations
ifeq ($(ENABLE_PROTEUS),yes)
  CFLAGS  += -I${PROTEUS_PATH}/include -DPROTEUS_ENABLE_HIP
  LDFLAGS += -Wl,-rpath,${PROTEUS_PATH}/lib64 -L${PROTEUS_PATH}/lib64/ -lproteus
endif

###############################################################################
# Targets
###############################################################################

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -x hip -c $< -o $@

.PHONY: clean run

clean:
	rm -rf $(program) *.o .proteus

# Default run
run: $(program)
	./$(program) 160000 1600 1000
