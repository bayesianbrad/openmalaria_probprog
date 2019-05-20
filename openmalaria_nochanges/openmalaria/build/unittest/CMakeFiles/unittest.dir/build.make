# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /code/openmalaria

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /code/openmalaria/build

# Include any dependencies generated for this target.
include unittest/CMakeFiles/unittest.dir/depend.make

# Include the progress variables for this target.
include unittest/CMakeFiles/unittest.dir/progress.make

# Include the compile flags for this target's objects.
include unittest/CMakeFiles/unittest.dir/flags.make

unittest/tests.cpp: ../unittest/ExtraAsserts.h
unittest/tests.cpp: ../unittest/LSTMPkPdSuite.h
unittest/tests.cpp: ../unittest/CheckpointSuite.h
unittest/tests.cpp: ../unittest/DummyInfectionSuite.h
unittest/tests.cpp: ../unittest/EmpiricalInfectionSuite.h
unittest/tests.cpp: ../unittest/InfectionImmunitySuite.h
unittest/tests.cpp: ../unittest/CMDecisionTreeSuite.h
unittest/tests.cpp: ../unittest/AgeGroupInterpolationSuite.h
unittest/tests.cpp: ../unittest/DecayFunctionSuite.h
unittest/tests.cpp: ../unittest/PennyInfectionSuite.h
unittest/tests.cpp: ../unittest/MolineauxInfectionSuite.h
unittest/tests.cpp: ../unittest/UtilVectorsSuite.h
unittest/tests.cpp: ../unittest/PkPdComplianceSuite.h
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating unittest code with cxxtestgen"
	cd /code/openmalaria/unittest && /usr/bin/perl /code/openmalaria/contrib/cxxtest/cxxtestgen.pl --have-std --have-eh --abort-on-fail --runner=ParenPrinter -o /code/openmalaria/build/unittest/tests.cpp ExtraAsserts.h LSTMPkPdSuite.h CheckpointSuite.h DummyInfectionSuite.h EmpiricalInfectionSuite.h InfectionImmunitySuite.h CMDecisionTreeSuite.h AgeGroupInterpolationSuite.h DecayFunctionSuite.h PennyInfectionSuite.h MolineauxInfectionSuite.h UtilVectorsSuite.h PkPdComplianceSuite.h

unittest/CMakeFiles/unittest.dir/WHMock.cpp.o: unittest/CMakeFiles/unittest.dir/flags.make
unittest/CMakeFiles/unittest.dir/WHMock.cpp.o: ../unittest/WHMock.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object unittest/CMakeFiles/unittest.dir/WHMock.cpp.o"
	cd /code/openmalaria/build/unittest && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/unittest.dir/WHMock.cpp.o -c /code/openmalaria/unittest/WHMock.cpp

unittest/CMakeFiles/unittest.dir/WHMock.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/unittest.dir/WHMock.cpp.i"
	cd /code/openmalaria/build/unittest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /code/openmalaria/unittest/WHMock.cpp > CMakeFiles/unittest.dir/WHMock.cpp.i

unittest/CMakeFiles/unittest.dir/WHMock.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/unittest.dir/WHMock.cpp.s"
	cd /code/openmalaria/build/unittest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /code/openmalaria/unittest/WHMock.cpp -o CMakeFiles/unittest.dir/WHMock.cpp.s

unittest/CMakeFiles/unittest.dir/WHMock.cpp.o.requires:

.PHONY : unittest/CMakeFiles/unittest.dir/WHMock.cpp.o.requires

unittest/CMakeFiles/unittest.dir/WHMock.cpp.o.provides: unittest/CMakeFiles/unittest.dir/WHMock.cpp.o.requires
	$(MAKE) -f unittest/CMakeFiles/unittest.dir/build.make unittest/CMakeFiles/unittest.dir/WHMock.cpp.o.provides.build
.PHONY : unittest/CMakeFiles/unittest.dir/WHMock.cpp.o.provides

unittest/CMakeFiles/unittest.dir/WHMock.cpp.o.provides.build: unittest/CMakeFiles/unittest.dir/WHMock.cpp.o


unittest/CMakeFiles/unittest.dir/tests.cpp.o: unittest/CMakeFiles/unittest.dir/flags.make
unittest/CMakeFiles/unittest.dir/tests.cpp.o: unittest/tests.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object unittest/CMakeFiles/unittest.dir/tests.cpp.o"
	cd /code/openmalaria/build/unittest && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/unittest.dir/tests.cpp.o -c /code/openmalaria/build/unittest/tests.cpp

unittest/CMakeFiles/unittest.dir/tests.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/unittest.dir/tests.cpp.i"
	cd /code/openmalaria/build/unittest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /code/openmalaria/build/unittest/tests.cpp > CMakeFiles/unittest.dir/tests.cpp.i

unittest/CMakeFiles/unittest.dir/tests.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/unittest.dir/tests.cpp.s"
	cd /code/openmalaria/build/unittest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /code/openmalaria/build/unittest/tests.cpp -o CMakeFiles/unittest.dir/tests.cpp.s

unittest/CMakeFiles/unittest.dir/tests.cpp.o.requires:

.PHONY : unittest/CMakeFiles/unittest.dir/tests.cpp.o.requires

unittest/CMakeFiles/unittest.dir/tests.cpp.o.provides: unittest/CMakeFiles/unittest.dir/tests.cpp.o.requires
	$(MAKE) -f unittest/CMakeFiles/unittest.dir/build.make unittest/CMakeFiles/unittest.dir/tests.cpp.o.provides.build
.PHONY : unittest/CMakeFiles/unittest.dir/tests.cpp.o.provides

unittest/CMakeFiles/unittest.dir/tests.cpp.o.provides.build: unittest/CMakeFiles/unittest.dir/tests.cpp.o


# Object files for target unittest
unittest_OBJECTS = \
"CMakeFiles/unittest.dir/WHMock.cpp.o" \
"CMakeFiles/unittest.dir/tests.cpp.o"

# External object files for target unittest
unittest_EXTERNAL_OBJECTS =

unittest/unittest: unittest/CMakeFiles/unittest.dir/WHMock.cpp.o
unittest/unittest: unittest/CMakeFiles/unittest.dir/tests.cpp.o
unittest/unittest: unittest/CMakeFiles/unittest.dir/build.make
unittest/unittest: model/libmodel.a
unittest/unittest: schema/libschema.a
unittest/unittest: contrib/libcontrib.a
unittest/unittest: /usr/lib/x86_64-linux-gnu/libgsl.so
unittest/unittest: /usr/lib/x86_64-linux-gnu/libgslcblas.so
unittest/unittest: /usr/lib/x86_64-linux-gnu/libxerces-c.so
unittest/unittest: /usr/lib/x86_64-linux-gnu/libz.so
unittest/unittest: /usr/lib/x86_64-linux-gnu/libpthread.so
unittest/unittest: unittest/CMakeFiles/unittest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable unittest"
	cd /code/openmalaria/build/unittest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/unittest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
unittest/CMakeFiles/unittest.dir/build: unittest/unittest

.PHONY : unittest/CMakeFiles/unittest.dir/build

unittest/CMakeFiles/unittest.dir/requires: unittest/CMakeFiles/unittest.dir/WHMock.cpp.o.requires
unittest/CMakeFiles/unittest.dir/requires: unittest/CMakeFiles/unittest.dir/tests.cpp.o.requires

.PHONY : unittest/CMakeFiles/unittest.dir/requires

unittest/CMakeFiles/unittest.dir/clean:
	cd /code/openmalaria/build/unittest && $(CMAKE_COMMAND) -P CMakeFiles/unittest.dir/cmake_clean.cmake
.PHONY : unittest/CMakeFiles/unittest.dir/clean

unittest/CMakeFiles/unittest.dir/depend: unittest/tests.cpp
	cd /code/openmalaria/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /code/openmalaria /code/openmalaria/unittest /code/openmalaria/build /code/openmalaria/build/unittest /code/openmalaria/build/unittest/CMakeFiles/unittest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : unittest/CMakeFiles/unittest.dir/depend
