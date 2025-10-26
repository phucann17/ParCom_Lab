CPP = g++

TARGET = ex
SRC = ex.cpp
OBJ = ex.o
OPENMPFLAG = -fopenmp

$(OBJ): $(SRC)
	$(CPP) $(OPENMPFLAG) -c $(SRC) -o $(OBJ)

$(TARGET): $(OBJ)
	$(CPP) $(OPENMPFLAG) -o $(TARGET) $(OBJ) 
	
# $(TARGET): $(SRC)
# 	$(CPP) -o $(TARGET) $(SRC)

build: $(TARGET)

clean: 
	rm -rf A.txt B.txt res.txt $(OBJ) $(TARGET)

run:
	./$(TARGET)

.PHONY: all clean run