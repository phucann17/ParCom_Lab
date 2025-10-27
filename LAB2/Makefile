CPP = mpicxx

TARGET = ex
SRC = ex.cpp
OBJ = ex.o
#OPENMPFLAG = -fopenmp

$(OBJ): $(SRC)
	$(CPP) -c $(SRC) -o $(OBJ)

$(TARGET): $(OBJ)
	$(CPP) -o $(TARGET) $(OBJ) 
	
# $(TARGET): $(SRC)
# 	$(CPP) -o $(TARGET) $(SRC)

build: $(TARGET)

clean: 
	rm -rf A.txt B.txt res.txt $(OBJ) $(TARGET)

run:
	mpirun -np 4 ./$(TARGET)

.PHONY: all clean run