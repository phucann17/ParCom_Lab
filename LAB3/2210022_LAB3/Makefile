CPP = g++

TARGET = ex
SRC = ex.cpp
OBJ = ex.o
OPENMPFLAG = -fopenmp


$(OBJ): $(SRC)
	$(CPP) $(OPENMPFLAG) -c $(SRC) -o $(OBJ)

# $(TARGET): $(OBJ)
# 	$(CPP) -o $(TARGET) $(OBJ) 
	
$(TARGET): $(OBJ)
	$(CPP) $(OPENMPFLAG) -o $(TARGET) $(OBJ) 

build: $(TARGET)

clean: 
	rm -rf A.txt B.txt res.txt result_parallel_matrix.txt result_sequential_matrix.txt $(OBJ) $(TARGET)

run:
	./$(TARGET)

.PHONY: all clean run
