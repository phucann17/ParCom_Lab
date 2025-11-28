CPP = mpicxx

TARGET = main
SRC = radioactive.cpp
OBJ = radioactive.o
OPENMPFLAG = -fopenmp


$(OBJ): $(SRC)
        $(CPP) $(OPENMPFLAG) -c $(SRC) -o $(OBJ)

# $(TARGET): $(OBJ)
#       $(CPP) -o $(TARGET) $(OBJ)

$(TARGET): $(OBJ)
        $(CPP) $(OPENMPFLAG) -o $(TARGET) $(OBJ)

build: $(TARGET)

clean:
        rm -rf A.txt B.txt res.txt result_parallel_matrix.txt result_sequential_matrix.txt $(OBJ) $(TARGET)

run:
        mpirun -np 5 ./$(TARGET)

run_node:
        mpirun -np 2 --hostfile hosts.txt ./main

.PHONY: all clean run