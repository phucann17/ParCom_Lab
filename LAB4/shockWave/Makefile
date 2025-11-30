CPP = mpicxx
TARGET = main
SRC = shockwave.cpp
OBJ = shockwave.o
OPENMPFLAG = -fopenmp
CFLAGS = -O3

.PHONY: all build clean run run_node_syn run_node_asyn

all: build

$(OBJ): $(SRC)
	$(CPP) $(OPENMPFLAG) $(CFLAGS) -c $(SRC) -o $(OBJ)

$(TARGET): $(OBJ)
	$(CPP) $(OPENMPFLAG) $(CFLAGS) -o $(TARGET) $(OBJ)

build: $(TARGET)

clean:
	rm -rf A.txt B.txt res.txt result_parallel_matrix.txt result_sequential_matrix.txt $(OBJ) $(TARGET)

run:
	mpirun -np 5 ./$(TARGET)

run_node_syn:
	mpirun -np 2 --hostfile hosts.txt ./main

run_node_asyn:
	mpirun -np 2 --hostfile hosts.txt ./main 1
.PHONY: all clean run