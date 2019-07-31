SRCDIR := src
BINDIR := bin
INCDIR := include

CC := g++
STD := c++14
NVCC := nvcc
CCFLAGS := -O3 -Wall -fopenmp
NVCCGENCODE = -gencode arch=compute_60,code=sm_60 \
			  -gencode arch=compute_61,code=sm_61 \
              -gencode arch=compute_70,code=sm_70

NVCCFLAGS := -std=$(STD) $(NVCCGENCODE) -ccbin $(CC) $(addprefix -Xcompiler ,$(CCFLAGS))

INC := $(wildcard $(INCDIR)/*)

all: | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCDIR) $(SRCDIR)/clahe.cu -o $(BINDIR)/clahe

debug: NVCCFLAGS += -g -O0 -Xptxas -v -DDEBUG
debug: all

profile: NVCCFLAGS += -lineinfo -g -Xptxas -v
profile: all

clean:
	$(RM) -r $(BINDIR)

$(BINDIR):
	mkdir -p $@

.PHONY: clean all $(BINDIR)
