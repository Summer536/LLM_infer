# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yqgao/LLM_inference/LLM_interview

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yqgao/LLM_inference/LLM_interview/build

# Include any dependencies generated for this target.
include src/kernels/CMakeFiles/sampling.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/kernels/CMakeFiles/sampling.dir/compiler_depend.make

# Include the progress variables for this target.
include src/kernels/CMakeFiles/sampling.dir/progress.make

# Include the compile flags for this target's objects.
include src/kernels/CMakeFiles/sampling.dir/flags.make

src/kernels/CMakeFiles/sampling.dir/sampling.cu.o: src/kernels/CMakeFiles/sampling.dir/flags.make
src/kernels/CMakeFiles/sampling.dir/sampling.cu.o: ../src/kernels/sampling.cu
src/kernels/CMakeFiles/sampling.dir/sampling.cu.o: src/kernels/CMakeFiles/sampling.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yqgao/LLM_inference/LLM_interview/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object src/kernels/CMakeFiles/sampling.dir/sampling.cu.o"
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/kernels && /usr/local/cuda-12.5/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT src/kernels/CMakeFiles/sampling.dir/sampling.cu.o -MF CMakeFiles/sampling.dir/sampling.cu.o.d -x cu -c /home/yqgao/LLM_inference/LLM_interview/src/kernels/sampling.cu -o CMakeFiles/sampling.dir/sampling.cu.o

src/kernels/CMakeFiles/sampling.dir/sampling.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/sampling.dir/sampling.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

src/kernels/CMakeFiles/sampling.dir/sampling.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/sampling.dir/sampling.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target sampling
sampling_OBJECTS = \
"CMakeFiles/sampling.dir/sampling.cu.o"

# External object files for target sampling
sampling_EXTERNAL_OBJECTS =

lib/libsampling.a: src/kernels/CMakeFiles/sampling.dir/sampling.cu.o
lib/libsampling.a: src/kernels/CMakeFiles/sampling.dir/build.make
lib/libsampling.a: src/kernels/CMakeFiles/sampling.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yqgao/LLM_inference/LLM_interview/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA static library ../../lib/libsampling.a"
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/kernels && $(CMAKE_COMMAND) -P CMakeFiles/sampling.dir/cmake_clean_target.cmake
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/kernels && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sampling.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/kernels/CMakeFiles/sampling.dir/build: lib/libsampling.a
.PHONY : src/kernels/CMakeFiles/sampling.dir/build

src/kernels/CMakeFiles/sampling.dir/clean:
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/kernels && $(CMAKE_COMMAND) -P CMakeFiles/sampling.dir/cmake_clean.cmake
.PHONY : src/kernels/CMakeFiles/sampling.dir/clean

src/kernels/CMakeFiles/sampling.dir/depend:
	cd /home/yqgao/LLM_inference/LLM_interview/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yqgao/LLM_inference/LLM_interview /home/yqgao/LLM_inference/LLM_interview/src/kernels /home/yqgao/LLM_inference/LLM_interview/build /home/yqgao/LLM_inference/LLM_interview/build/src/kernels /home/yqgao/LLM_inference/LLM_interview/build/src/kernels/CMakeFiles/sampling.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/kernels/CMakeFiles/sampling.dir/depend

