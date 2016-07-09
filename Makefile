CXX = $(shell root-config --cxx)
LD = $(shell root-config --ld)

INC = $(shell pwd)

#CPPFLAGS :=  $(shell root-config --cflags) -I$(INC)/include -I$(ROOTINC)/math/mathmore/inc
#LDFLAGS :=  $(shell root-config --glibs) $(STDLIBDIR) -lMathMore
CUDAFLAGS :=  -L/usr/local/cuda/lib64 -lcudart -lcuda

# Debugging Flag
CPPFLAGS := -g

TARGET1 = razor
OBJ1 = jsoncpp.o
OBJ2 = razor_cuda.o
OBJ3 = razor.o

all : $(TARGET1)


$(TARGET1) : $(OBJ1) $(OBJ2) $(OBJ3)
	nvcc $(CPPFLAGS) -o $(TARGET1) $(OBJ1) $(OBJ2) $(OBJ3) $(OBJ4) $(LDFLAGS)
	@echo $@
	@echo $<
	@echo $^

%.o : %.cpp %.cc
	$(CXX) $(CPPFLAGS) -o $@ -c $<
	@echo $@
	@echo $<

%.o : %.cu
	nvcc -g -o $@ -c $<
	@echo $@
	@echo $<

clean :
	rm -f *.o src/*.o $(TARGET1) *~
