# Define variables
CXX = g++
CXXFLAGS = -std=c++11 -O3
LDFLAGS = -lfreeimage
OUTPUT_DIR = bin
SRC_DIR = src
RES_DIR = res
REP_DIR = reports
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp,$(OUTPUT_DIR)/%.o,$(SRC_FILES))
NCU = $(shell which ncu)


all: fresh_build

prepare:
	@echo "Preparing..."
	@mkdir -p $(SRC_DIR) $(OUTPUT_DIR) $(RES_DIR) $(REP_DIR)
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
	@echo "Cleaning Reports..."
	@rm -rf $(REP_DIR)
	@echo "All files cleaned!"
clean-res:
	@echo "Cleaning Results..."	
	@rm -rf $(RES_DIR)
	@mkdir -p $(RES_DIR)
	@echo "Cleaned!		"

run:
	@echo "Running CPU Version..."
	-@$(OUTPUT_DIR)/mandelbrot_nonacc || echo "Error in Non-Accelerated"
	@echo "Running GPU Version..."
	-@$(OUTPUT_DIR)/mandelbrot_acc || echo "Error in Accelerated"
	@echo "Finished!!!"

check:
	@python similarity.py

profiler:
	@echo "Generating Nsights-Systems Report..."
	-@nsys profile -o $(REP_DIR)/nsys_profile.nsys-rep --force-overwrite true $(OUTPUT_DIR)/mandelbrot_acc
	@echo "Opening NSYS Report!"
	@nsys-ui $(REP_DIR)/nsys_profile.nsys-rep
	@echo "Generating Nsights-Compute Report..."
	@sudo $(NCU) -o $(REP_DIR)/ncu-rep.ncu-rep -f --section ComputeWorkloadAnalysis --section InstructionStats --section SpeedOfLight $(OUTPUT_DIR)/mandelbrot_acc	
	@echo "Opening NCU Report!"
	@ncu-ui $(REP_DIR)/ncu-rep.ncu-rep	
	@echo "Generated!"

fresh_build: clean prepare build build-gpu run check
execution: clean-res run check 
# Phony targets
.PHONY: all prepare build build-gpu clean clean-res run check profiler
