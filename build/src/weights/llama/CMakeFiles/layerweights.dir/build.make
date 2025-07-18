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
include src/weights/llama/CMakeFiles/layerweights.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/weights/llama/CMakeFiles/layerweights.dir/compiler_depend.make

# Include the progress variables for this target.
include src/weights/llama/CMakeFiles/layerweights.dir/progress.make

# Include the compile flags for this target's objects.
include src/weights/llama/CMakeFiles/layerweights.dir/flags.make

src/weights/llama/CMakeFiles/layerweights.dir/layer_weights.cc.o: src/weights/llama/CMakeFiles/layerweights.dir/flags.make
src/weights/llama/CMakeFiles/layerweights.dir/layer_weights.cc.o: ../src/weights/llama/layer_weights.cc
src/weights/llama/CMakeFiles/layerweights.dir/layer_weights.cc.o: src/weights/llama/CMakeFiles/layerweights.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yqgao/LLM_inference/LLM_interview/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/weights/llama/CMakeFiles/layerweights.dir/layer_weights.cc.o"
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/weights/llama && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/weights/llama/CMakeFiles/layerweights.dir/layer_weights.cc.o -MF CMakeFiles/layerweights.dir/layer_weights.cc.o.d -o CMakeFiles/layerweights.dir/layer_weights.cc.o -c /home/yqgao/LLM_inference/LLM_interview/src/weights/llama/layer_weights.cc

src/weights/llama/CMakeFiles/layerweights.dir/layer_weights.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/layerweights.dir/layer_weights.cc.i"
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/weights/llama && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yqgao/LLM_inference/LLM_interview/src/weights/llama/layer_weights.cc > CMakeFiles/layerweights.dir/layer_weights.cc.i

src/weights/llama/CMakeFiles/layerweights.dir/layer_weights.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/layerweights.dir/layer_weights.cc.s"
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/weights/llama && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yqgao/LLM_inference/LLM_interview/src/weights/llama/layer_weights.cc -o CMakeFiles/layerweights.dir/layer_weights.cc.s

# Object files for target layerweights
layerweights_OBJECTS = \
"CMakeFiles/layerweights.dir/layer_weights.cc.o"

# External object files for target layerweights
layerweights_EXTERNAL_OBJECTS =

lib/liblayerweights.a: src/weights/llama/CMakeFiles/layerweights.dir/layer_weights.cc.o
lib/liblayerweights.a: src/weights/llama/CMakeFiles/layerweights.dir/build.make
lib/liblayerweights.a: src/weights/llama/CMakeFiles/layerweights.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yqgao/LLM_inference/LLM_interview/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ../../../lib/liblayerweights.a"
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/weights/llama && $(CMAKE_COMMAND) -P CMakeFiles/layerweights.dir/cmake_clean_target.cmake
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/weights/llama && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/layerweights.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/weights/llama/CMakeFiles/layerweights.dir/build: lib/liblayerweights.a
.PHONY : src/weights/llama/CMakeFiles/layerweights.dir/build

src/weights/llama/CMakeFiles/layerweights.dir/clean:
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/weights/llama && $(CMAKE_COMMAND) -P CMakeFiles/layerweights.dir/cmake_clean.cmake
.PHONY : src/weights/llama/CMakeFiles/layerweights.dir/clean

src/weights/llama/CMakeFiles/layerweights.dir/depend:
	cd /home/yqgao/LLM_inference/LLM_interview/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yqgao/LLM_inference/LLM_interview /home/yqgao/LLM_inference/LLM_interview/src/weights/llama /home/yqgao/LLM_inference/LLM_interview/build /home/yqgao/LLM_inference/LLM_interview/build/src/weights/llama /home/yqgao/LLM_inference/LLM_interview/build/src/weights/llama/CMakeFiles/layerweights.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/weights/llama/CMakeFiles/layerweights.dir/depend

