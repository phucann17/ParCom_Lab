CPP = mpicxx

TARGET = ex
SRC = ex.cpp
OBJ = ex.o
OPENMPFLAG = -fopenmp


$(OBJ): $(SRC)
	$(CPP) -c $(SRC) -o $(OBJ)

# $(TARGET): $(OBJ)
# 	$(CPP) -o $(TARGET) $(OBJ) 
	
$(TARGET): $(OBJ)
	$(CPP) -o $(TARGET) $(OBJ) 

build: $(TARGET)

clean: 
	rm -rf A.txt B.txt res.txt result_parallel_matrix.txt result_sequential_matrix.txt $(OBJ) $(TARGET)

run:
	mpirun -np 7 ./$(TARGET)

.PHONY: all clean run
