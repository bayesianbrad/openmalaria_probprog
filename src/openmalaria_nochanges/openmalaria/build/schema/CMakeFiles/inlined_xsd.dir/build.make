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

# Utility rule file for inlined_xsd.

# Include the progress variables for this target.
include schema/CMakeFiles/inlined_xsd.dir/progress.make

schema/CMakeFiles/inlined_xsd: schema/scenario_current.xsd


schema/scenario_current.xsd: ../schema/scenario.xsd
schema/scenario_current.xsd: ../schema/demography.xsd
schema/scenario_current.xsd: ../schema/monitoring.xsd
schema/scenario_current.xsd: ../schema/interventions.xsd
schema/scenario_current.xsd: ../schema/healthSystem.xsd
schema/scenario_current.xsd: ../schema/entomology.xsd
schema/scenario_current.xsd: ../schema/pharmacology.xsd
schema/scenario_current.xsd: ../schema/util.xsd
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Inlining scenario.xsd and dependencies into /code/openmalaria/build/schema/scenario_current.xsd"
	cd /code/openmalaria/schema && /usr/bin/python /code/openmalaria/util/inlineSchema.py scenario.xsd /code/openmalaria/build/schema/scenario_current.xsd

inlined_xsd: schema/CMakeFiles/inlined_xsd
inlined_xsd: schema/scenario_current.xsd
inlined_xsd: schema/CMakeFiles/inlined_xsd.dir/build.make

.PHONY : inlined_xsd

# Rule to build all files generated by this target.
schema/CMakeFiles/inlined_xsd.dir/build: inlined_xsd

.PHONY : schema/CMakeFiles/inlined_xsd.dir/build

schema/CMakeFiles/inlined_xsd.dir/clean:
	cd /code/openmalaria/build/schema && $(CMAKE_COMMAND) -P CMakeFiles/inlined_xsd.dir/cmake_clean.cmake
.PHONY : schema/CMakeFiles/inlined_xsd.dir/clean

schema/CMakeFiles/inlined_xsd.dir/depend:
	cd /code/openmalaria/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /code/openmalaria /code/openmalaria/schema /code/openmalaria/build /code/openmalaria/build/schema /code/openmalaria/build/schema/CMakeFiles/inlined_xsd.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : schema/CMakeFiles/inlined_xsd.dir/depend
