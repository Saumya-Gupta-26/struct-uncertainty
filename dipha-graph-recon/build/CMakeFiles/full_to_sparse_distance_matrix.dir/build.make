# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

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
CMAKE_COMMAND = /home/saumgupta/.local/lib/python3.9/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/saumgupta/.local/lib/python3.9/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/saumgupta/dmt-crf-public-code/dipha-graph-recon

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/saumgupta/dmt-crf-public-code/dipha-graph-recon/build

# Include any dependencies generated for this target.
include CMakeFiles/full_to_sparse_distance_matrix.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/full_to_sparse_distance_matrix.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/full_to_sparse_distance_matrix.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/full_to_sparse_distance_matrix.dir/flags.make

CMakeFiles/full_to_sparse_distance_matrix.dir/src/full_to_sparse_distance_matrix.cpp.o: CMakeFiles/full_to_sparse_distance_matrix.dir/flags.make
CMakeFiles/full_to_sparse_distance_matrix.dir/src/full_to_sparse_distance_matrix.cpp.o: /home/saumgupta/dmt-crf-public-code/dipha-graph-recon/src/full_to_sparse_distance_matrix.cpp
CMakeFiles/full_to_sparse_distance_matrix.dir/src/full_to_sparse_distance_matrix.cpp.o: CMakeFiles/full_to_sparse_distance_matrix.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/saumgupta/dmt-crf-public-code/dipha-graph-recon/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/full_to_sparse_distance_matrix.dir/src/full_to_sparse_distance_matrix.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/full_to_sparse_distance_matrix.dir/src/full_to_sparse_distance_matrix.cpp.o -MF CMakeFiles/full_to_sparse_distance_matrix.dir/src/full_to_sparse_distance_matrix.cpp.o.d -o CMakeFiles/full_to_sparse_distance_matrix.dir/src/full_to_sparse_distance_matrix.cpp.o -c /home/saumgupta/dmt-crf-public-code/dipha-graph-recon/src/full_to_sparse_distance_matrix.cpp

CMakeFiles/full_to_sparse_distance_matrix.dir/src/full_to_sparse_distance_matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/full_to_sparse_distance_matrix.dir/src/full_to_sparse_distance_matrix.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/saumgupta/dmt-crf-public-code/dipha-graph-recon/src/full_to_sparse_distance_matrix.cpp > CMakeFiles/full_to_sparse_distance_matrix.dir/src/full_to_sparse_distance_matrix.cpp.i

CMakeFiles/full_to_sparse_distance_matrix.dir/src/full_to_sparse_distance_matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/full_to_sparse_distance_matrix.dir/src/full_to_sparse_distance_matrix.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/saumgupta/dmt-crf-public-code/dipha-graph-recon/src/full_to_sparse_distance_matrix.cpp -o CMakeFiles/full_to_sparse_distance_matrix.dir/src/full_to_sparse_distance_matrix.cpp.s

# Object files for target full_to_sparse_distance_matrix
full_to_sparse_distance_matrix_OBJECTS = \
"CMakeFiles/full_to_sparse_distance_matrix.dir/src/full_to_sparse_distance_matrix.cpp.o"

# External object files for target full_to_sparse_distance_matrix
full_to_sparse_distance_matrix_EXTERNAL_OBJECTS =

full_to_sparse_distance_matrix: CMakeFiles/full_to_sparse_distance_matrix.dir/src/full_to_sparse_distance_matrix.cpp.o
full_to_sparse_distance_matrix: CMakeFiles/full_to_sparse_distance_matrix.dir/build.make
full_to_sparse_distance_matrix: /usr/local/anaconda3/lib/libmpicxx.so
full_to_sparse_distance_matrix: /usr/local/anaconda3/lib/libmpi.so
full_to_sparse_distance_matrix: CMakeFiles/full_to_sparse_distance_matrix.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/saumgupta/dmt-crf-public-code/dipha-graph-recon/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable full_to_sparse_distance_matrix"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/full_to_sparse_distance_matrix.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/full_to_sparse_distance_matrix.dir/build: full_to_sparse_distance_matrix
.PHONY : CMakeFiles/full_to_sparse_distance_matrix.dir/build

CMakeFiles/full_to_sparse_distance_matrix.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/full_to_sparse_distance_matrix.dir/cmake_clean.cmake
.PHONY : CMakeFiles/full_to_sparse_distance_matrix.dir/clean

CMakeFiles/full_to_sparse_distance_matrix.dir/depend:
	cd /home/saumgupta/dmt-crf-public-code/dipha-graph-recon/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/saumgupta/dmt-crf-public-code/dipha-graph-recon /home/saumgupta/dmt-crf-public-code/dipha-graph-recon /home/saumgupta/dmt-crf-public-code/dipha-graph-recon/build /home/saumgupta/dmt-crf-public-code/dipha-graph-recon/build /home/saumgupta/dmt-crf-public-code/dipha-graph-recon/build/CMakeFiles/full_to_sparse_distance_matrix.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/full_to_sparse_distance_matrix.dir/depend
