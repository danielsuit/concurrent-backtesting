CXX = clang++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall
LDFLAGS = -pthread

SRC_DIR = src
TARGET = model_run

SRCS = $(SRC_DIR)/main.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -c -o $@ $<

export_models:
	python3 ../training/export_models.py

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean export_models
