CXX = clang++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -flto
LDFLAGS = -pthread -flto

SRC_DIR = src
TEST_DIR = tests
TARGET = model_run
TEST_TARGET = run_tests

SRCS = $(SRC_DIR)/main.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -c -o $@ $<

# ---------------------------------------------------------------------------
# Test target — builds and runs the doctest-based test suite
# ---------------------------------------------------------------------------
$(TEST_TARGET): $(TEST_DIR)/test_main.cpp $(wildcard $(SRC_DIR)/*.hpp)
	$(CXX) -std=c++17 -Wall -g -I$(SRC_DIR) -I$(TEST_DIR) -o $@ $(TEST_DIR)/test_main.cpp

test: $(TEST_TARGET)
	./$(TEST_TARGET)

export_models:
	python3 ../training/export_models.py

clean:
	rm -f $(OBJS) $(TARGET) $(TEST_TARGET)

.PHONY: all clean export_models test
