# Define variables
CXX = g++
CXXFLAGS = -std=c++11 -O3
LDFLAGS = -lfreeimage
OUTPUT_DIR = bin
SRC_DIR = src
RES_DIR = res
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp,$(OUTPUT_DIR)/%.o,$(SRC_FILES))

all: fresh_build

prepare:
	@echo "Preparing..."
	@mkdir -p $(SRC_DIR) $(OUTPUT_DIR) $(RES_DIR)
	@echo "Prepared!"

build: 
	@echo "Building Non-Accelerated..."
	@$(CXX) $(CXXFLAGS) $(SRC_FILES) $(LDFLAGS) -o $(OUTPUT_DIR)/mandelbrot_nonacc
	@echo "Built Non-Accelerated!"

build-gpu: 
	@echo "Building Accelerated..."
	@nvcc -arch=native -std=c++11 -O3 $(SRC_DIR)/save_image.cpp $(SRC_DIR)/utils.cpp $(SRC_DIR)/mandel.cu -lfreeimage -o $(OUTPUT_DIR)/mandelbrot_acc
	@echo "Built Accelerated!"
# Clean up build artifacts
clean:
	@echo "Cleaning build files..."
	@rm -rf $(OUTPUT_DIR)
	@echo "Cleaning Results..."
	@rm -rf $(RES_DIR)
	@echo "All files cleaned!"

run:
	@echo "Running CPU Version..."
	-@$(OUTPUT_DIR)/mandelbrot_nonacc || echo "Error in Non-Accelerated"
	@echo "Running GPU Version..."
	-@$(OUTPUT_DIR)/mandelbrot_acc || echo "Error in Accelerated"
	@echo "Finished!!!"

fresh_build: clean prepare build build-gpu run 
# Phony targets
.PHONY: all prepare build build-gpu clean run
