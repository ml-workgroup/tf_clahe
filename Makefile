SRCDIR := src
BINDIR := bin
INCDIR := ../include
TF_INC ?= $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') 
TF_CFLAGS= $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS= $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
INC := $(wildcard $(INCDIR)/*)

CXX := g++
STD := c++14
CXXFLAGS := -std=$(STD) -O3 -Wall -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC

CUDA_HOME=/usr/local/cuda
NVCC=$(CUDA_HOME)/bin/nvcc
NVCCGENCODE = -gencode arch=compute_60,code=sm_60 \
			  -gencode arch=compute_61,code=sm_61 \
              -gencode arch=compute_70,code=sm_70 \
			  --expt-relaxed-constexpr
NVCCFLAGS := -std=$(STD) $(NVCCGENCODE) -ccbin $(CXX) $(addprefix -Xcompiler ,$(CXXFLAGS))


all: info $(BINDIR) clahe clahe_op_kernel.so

info:
	@echo -e '==== INFO =========================================='
	@echo -e '  CUDA_HOME=$(CUDA_HOME)                            '
	@echo -e '  TF_INC=$(TF_INC)                                  '
	@echo -e '  TF_CFLAGS=$(TF_CFLAGS)                            '
	@echo -e '  TF_LFLAGS=$(TF_LFLAGS)                            '
	@echo -e '===================================================='

clahe: $(SRCDIR)/clahe.cu
	 $(NVCC) $(NVCCFLAGS) -I$(INCDIR) $(SRCDIR)/clahe.cu -o $(BINDIR)/clahe

clahe_op_kernel.cu.o: $(SRCDIR)/clahe_op_kernel.cu.cc
	$(NVCC) $(NVCCFLAGS) $(SRCDIR)/clahe_op_kernel.cu.cc  -c \
		-o $(BINDIR)/clahe_op_kernel.cu.o -I $(TF_INC) \
		-D GOOGLE_CUDA=1 -x cu $(addprefix -Xcompiler ,$(CXXFLAGS))

clahe_op_kernel.cc.o: $(SRCDIR)/clahe_op_kernel.cc
	$(CXX) $(CXXFLAGS) $(SRCDIR)/clahe_op_kernel.cc -c \
	-o $(BINDIR)/clahe_op_kernel.cc.o -I $(TF_INC) -I $(CUDA_HOME)/include -fPIC -Wall

clahe_op_kernel.so: clahe_op_kernel.cu.o clahe_op_kernel.cc.o
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed ${TF_CFLAGS} \
	${TF_LFLAGS} -o $(BINDIR)/clahe_op_kernel.so $(BINDIR)/clahe_op_kernel.cc.o \
	$(BINDIR)/clahe_op_kernel.cu.o -lcudart -L $(CUDA_HOME)/lib64 

debug: NVCCFLAGS += -g -O0 -Xptxas -v -DDEBUG
debug: all

profile: NVCCFLAGS += -lineinfo -g -Xptxas -v
profile: all

clean:
	$(RM) -r $(BINDIR)

$(BINDIR):
	mkdir -p $@

.PHONY: clean all $(BINDIR)