# Change the value of this macro to select your C compiler
#!!!!!!!!!!!!!!!!#
CC = icc
#!!!!!!!!!!!!!!!!#

# include directory
IDIR = include

# objects directory
ODIR = obj
TESTODIR = obj

# source directory
SDIR = src
TESTSDIR = tests

# compile dependecies
_DEPS = parameters.h field.h operators.h fftwplans.h output.h solver.h control.h dbg.h 

# object files
_OBJ =  parameters.o field.o operators.o fftwplans.o output.o solver.o control.o
_MAINOBJ = main.o

# test objects files
_TESTOBJ = tests.o

# substitute paths
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))
MAINOBJ = $(patsubst %,$(ODIR)/%,$(_MAINOBJ))
TESTOBJ = $(patsubst %,$(TESTODIR)/%,$(_TESTOBJ))

# libraries, compile flags and include directories
LIBS = -lfftw3 -liniparser -lm -L$(MKLROOT)/lib/intel64 -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -lpthread -lm

#-llapacke
CFLAGS = -Wall -std=gnu99 -O3 -openmp  -DMKL_ILP64 -openmp -I$(MKLROOT)/include
# -profile-functions -profile-loops=all
# -mkl=sequential
INCLUDE = -I$(IDIR)


rpcf: $(MAINOBJ) $(OBJ) $(DEPS) 
	$(CC) -o $@ $(OBJ) $(MAINOBJ) $(LIBS) $(CFLAGS) $(INCLUDE)

$(ODIR)/%.o: $(SDIR)/%.c $(DEPS) 
	$(CC) -c -o $@ $< $(CFLAGS) $(INCLUDE)

$(TESTODIR)/%.o: $(TESTSDIR)/%.c $(DEPS) 
	$(CC) -c -o $@ $< $(CFLAGS) $(INCLUDE)

debug: $(OBJ) $(DEPS) 
	$(CC) -o $@ $(OBJ) $(LIBS) $(CFLAGS) $(INCLUDE) -DDEBUG

test: $(TESTOBJ) $(OBJ) $(DEPS)
	$(CC) -o $@ $(OBJ) $(TESTOBJ) $(LIBS) $(CFLAGS) $(INCLUDE)

clean:
	rm -f $(ODIR)/*.o
