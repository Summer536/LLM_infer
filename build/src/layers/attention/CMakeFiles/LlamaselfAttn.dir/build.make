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
include src/layers/attention/CMakeFiles/LlamaselfAttn.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/layers/attention/CMakeFiles/LlamaselfAttn.dir/compiler_depend.make

# Include the progress variables for this target.
include src/layers/attention/CMakeFiles/LlamaselfAttn.dir/progress.make

# Include the compile flags for this target's objects.
include src/layers/attention/CMakeFiles/LlamaselfAttn.dir/flags.make

src/layers/attention/CMakeFiles/LlamaselfAttn.dir/masked_self_attention.cpp.o: src/layers/attention/CMakeFiles/LlamaselfAttn.dir/flags.make
src/layers/attention/CMakeFiles/LlamaselfAttn.dir/masked_self_attention.cpp.o: ../src/layers/attention/masked_self_attention.cpp
src/layers/attention/CMakeFiles/LlamaselfAttn.dir/masked_self_attention.cpp.o: src/layers/attention/CMakeFiles/LlamaselfAttn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yqgao/LLM_inference/LLM_interview/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/layers/attention/CMakeFiles/LlamaselfAttn.dir/masked_self_attention.cpp.o"
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/layers/attention && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/layers/attention/CMakeFiles/LlamaselfAttn.dir/masked_self_attention.cpp.o -MF CMakeFiles/LlamaselfAttn.dir/masked_self_attention.cpp.o.d -o CMakeFiles/LlamaselfAttn.dir/masked_self_attention.cpp.o -c /home/yqgao/LLM_inference/LLM_interview/src/layers/attention/masked_self_attention.cpp

src/layers/attention/CMakeFiles/LlamaselfAttn.dir/masked_self_attention.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LlamaselfAttn.dir/masked_self_attention.cpp.i"
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/layers/attention && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yqgao/LLM_inference/LLM_interview/src/layers/attention/masked_self_attention.cpp > CMakeFiles/LlamaselfAttn.dir/masked_self_attention.cpp.i

src/layers/attention/CMakeFiles/LlamaselfAttn.dir/masked_self_attention.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LlamaselfAttn.dir/masked_self_attention.cpp.s"
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/layers/attention && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yqgao/LLM_inference/LLM_interview/src/layers/attention/masked_self_attention.cpp -o CMakeFiles/LlamaselfAttn.dir/masked_self_attention.cpp.s

# Object files for target LlamaselfAttn
LlamaselfAttn_OBJECTS = \
"CMakeFiles/LlamaselfAttn.dir/masked_self_attention.cpp.o"

# External object files for target LlamaselfAttn
LlamaselfAttn_EXTERNAL_OBJECTS =

lib/libLlamaselfAttn.a: src/layers/attention/CMakeFiles/LlamaselfAttn.dir/masked_self_attention.cpp.o
lib/libLlamaselfAttn.a: src/layers/attention/CMakeFiles/LlamaselfAttn.dir/build.make
lib/libLlamaselfAttn.a: src/layers/attention/CMakeFiles/LlamaselfAttn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yqgao/LLM_inference/LLM_interview/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ../../../lib/libLlamaselfAttn.a"
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/layers/attention && $(CMAKE_COMMAND) -P CMakeFiles/LlamaselfAttn.dir/cmake_clean_target.cmake
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/layers/attention && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LlamaselfAttn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/layers/attention/CMakeFiles/LlamaselfAttn.dir/build: lib/libLlamaselfAttn.a
.PHONY : src/layers/attention/CMakeFiles/LlamaselfAttn.dir/build

src/layers/attention/CMakeFiles/LlamaselfAttn.dir/clean:
	cd /home/yqgao/LLM_inference/LLM_interview/build/src/layers/attention && $(CMAKE_COMMAND) -P CMakeFiles/LlamaselfAttn.dir/cmake_clean.cmake
.PHONY : src/layers/attention/CMakeFiles/LlamaselfAttn.dir/clean

src/layers/attention/CMakeFiles/LlamaselfAttn.dir/depend:
	cd /home/yqgao/LLM_inference/LLM_interview/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yqgao/LLM_inference/LLM_interview /home/yqgao/LLM_inference/LLM_interview/src/layers/attention /home/yqgao/LLM_inference/LLM_interview/build /home/yqgao/LLM_inference/LLM_interview/build/src/layers/attention /home/yqgao/LLM_inference/LLM_interview/build/src/layers/attention/CMakeFiles/LlamaselfAttn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/layers/attention/CMakeFiles/LlamaselfAttn.dir/depend

