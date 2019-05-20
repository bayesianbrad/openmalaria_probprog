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
include schema/CMakeFiles/schema.dir/depend.make

# Include the progress variables for this target.
include schema/CMakeFiles/schema.dir/progress.make

# Include the compile flags for this target's objects.
include schema/CMakeFiles/schema.dir/flags.make

schema/scenario.cpp: ../schema/scenario.xsd
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Compiling /code/openmalaria/schema/scenario.xsd"
	cd /code/openmalaria/build/schema && /usr/bin/xsdcxx cxx-tree --std c++11 --type-naming ucc --function-naming java --namespace-map http://openmalaria.org/schema/scenario_39=scnXml --generate-doxygen --generate-intellisense --hxx-suffix .h --cxx-suffix .cpp /code/openmalaria/schema/scenario.xsd

schema/scenario.h: schema/scenario.cpp
	@$(CMAKE_COMMAND) -E touch_nocreate schema/scenario.h

schema/demography.cpp: ../schema/demography.xsd
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Compiling /code/openmalaria/schema/demography.xsd"
	cd /code/openmalaria/build/schema && /usr/bin/xsdcxx cxx-tree --std c++11 --type-naming ucc --function-naming java --namespace-map http://openmalaria.org/schema/scenario_39=scnXml --generate-doxygen --generate-intellisense --hxx-suffix .h --cxx-suffix .cpp /code/openmalaria/schema/demography.xsd

schema/demography.h: schema/demography.cpp
	@$(CMAKE_COMMAND) -E touch_nocreate schema/demography.h

schema/monitoring.cpp: ../schema/monitoring.xsd
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Compiling /code/openmalaria/schema/monitoring.xsd"
	cd /code/openmalaria/build/schema && /usr/bin/xsdcxx cxx-tree --std c++11 --type-naming ucc --function-naming java --namespace-map http://openmalaria.org/schema/scenario_39=scnXml --generate-doxygen --generate-intellisense --hxx-suffix .h --cxx-suffix .cpp /code/openmalaria/schema/monitoring.xsd

schema/monitoring.h: schema/monitoring.cpp
	@$(CMAKE_COMMAND) -E touch_nocreate schema/monitoring.h

schema/interventions.cpp: ../schema/interventions.xsd
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Compiling /code/openmalaria/schema/interventions.xsd"
	cd /code/openmalaria/build/schema && /usr/bin/xsdcxx cxx-tree --std c++11 --type-naming ucc --function-naming java --namespace-map http://openmalaria.org/schema/scenario_39=scnXml --generate-doxygen --generate-intellisense --hxx-suffix .h --cxx-suffix .cpp /code/openmalaria/schema/interventions.xsd

schema/interventions.h: schema/interventions.cpp
	@$(CMAKE_COMMAND) -E touch_nocreate schema/interventions.h

schema/healthSystem.cpp: ../schema/healthSystem.xsd
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Compiling /code/openmalaria/schema/healthSystem.xsd"
	cd /code/openmalaria/build/schema && /usr/bin/xsdcxx cxx-tree --std c++11 --type-naming ucc --function-naming java --namespace-map http://openmalaria.org/schema/scenario_39=scnXml --generate-doxygen --generate-intellisense --hxx-suffix .h --cxx-suffix .cpp /code/openmalaria/schema/healthSystem.xsd

schema/healthSystem.h: schema/healthSystem.cpp
	@$(CMAKE_COMMAND) -E touch_nocreate schema/healthSystem.h

schema/entomology.cpp: ../schema/entomology.xsd
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Compiling /code/openmalaria/schema/entomology.xsd"
	cd /code/openmalaria/build/schema && /usr/bin/xsdcxx cxx-tree --std c++11 --type-naming ucc --function-naming java --namespace-map http://openmalaria.org/schema/scenario_39=scnXml --generate-doxygen --generate-intellisense --hxx-suffix .h --cxx-suffix .cpp /code/openmalaria/schema/entomology.xsd

schema/entomology.h: schema/entomology.cpp
	@$(CMAKE_COMMAND) -E touch_nocreate schema/entomology.h

schema/pharmacology.cpp: ../schema/pharmacology.xsd
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Compiling /code/openmalaria/schema/pharmacology.xsd"
	cd /code/openmalaria/build/schema && /usr/bin/xsdcxx cxx-tree --std c++11 --type-naming ucc --function-naming java --namespace-map http://openmalaria.org/schema/scenario_39=scnXml --generate-doxygen --generate-intellisense --hxx-suffix .h --cxx-suffix .cpp /code/openmalaria/schema/pharmacology.xsd

schema/pharmacology.h: schema/pharmacology.cpp
	@$(CMAKE_COMMAND) -E touch_nocreate schema/pharmacology.h

schema/util.cpp: ../schema/util.xsd
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Compiling /code/openmalaria/schema/util.xsd"
	cd /code/openmalaria/build/schema && /usr/bin/xsdcxx cxx-tree --std c++11 --type-naming ucc --function-naming java --namespace-map http://openmalaria.org/schema/scenario_39=scnXml --generate-doxygen --generate-intellisense --hxx-suffix .h --cxx-suffix .cpp /code/openmalaria/schema/util.xsd

schema/util.h: schema/util.cpp
	@$(CMAKE_COMMAND) -E touch_nocreate schema/util.h

schema/CMakeFiles/schema.dir/scenario.cpp.o: schema/CMakeFiles/schema.dir/flags.make
schema/CMakeFiles/schema.dir/scenario.cpp.o: schema/scenario.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object schema/CMakeFiles/schema.dir/scenario.cpp.o"
	cd /code/openmalaria/build/schema && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/schema.dir/scenario.cpp.o -c /code/openmalaria/build/schema/scenario.cpp

schema/CMakeFiles/schema.dir/scenario.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/schema.dir/scenario.cpp.i"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /code/openmalaria/build/schema/scenario.cpp > CMakeFiles/schema.dir/scenario.cpp.i

schema/CMakeFiles/schema.dir/scenario.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/schema.dir/scenario.cpp.s"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /code/openmalaria/build/schema/scenario.cpp -o CMakeFiles/schema.dir/scenario.cpp.s

schema/CMakeFiles/schema.dir/scenario.cpp.o.requires:

.PHONY : schema/CMakeFiles/schema.dir/scenario.cpp.o.requires

schema/CMakeFiles/schema.dir/scenario.cpp.o.provides: schema/CMakeFiles/schema.dir/scenario.cpp.o.requires
	$(MAKE) -f schema/CMakeFiles/schema.dir/build.make schema/CMakeFiles/schema.dir/scenario.cpp.o.provides.build
.PHONY : schema/CMakeFiles/schema.dir/scenario.cpp.o.provides

schema/CMakeFiles/schema.dir/scenario.cpp.o.provides.build: schema/CMakeFiles/schema.dir/scenario.cpp.o


schema/CMakeFiles/schema.dir/demography.cpp.o: schema/CMakeFiles/schema.dir/flags.make
schema/CMakeFiles/schema.dir/demography.cpp.o: schema/demography.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object schema/CMakeFiles/schema.dir/demography.cpp.o"
	cd /code/openmalaria/build/schema && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/schema.dir/demography.cpp.o -c /code/openmalaria/build/schema/demography.cpp

schema/CMakeFiles/schema.dir/demography.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/schema.dir/demography.cpp.i"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /code/openmalaria/build/schema/demography.cpp > CMakeFiles/schema.dir/demography.cpp.i

schema/CMakeFiles/schema.dir/demography.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/schema.dir/demography.cpp.s"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /code/openmalaria/build/schema/demography.cpp -o CMakeFiles/schema.dir/demography.cpp.s

schema/CMakeFiles/schema.dir/demography.cpp.o.requires:

.PHONY : schema/CMakeFiles/schema.dir/demography.cpp.o.requires

schema/CMakeFiles/schema.dir/demography.cpp.o.provides: schema/CMakeFiles/schema.dir/demography.cpp.o.requires
	$(MAKE) -f schema/CMakeFiles/schema.dir/build.make schema/CMakeFiles/schema.dir/demography.cpp.o.provides.build
.PHONY : schema/CMakeFiles/schema.dir/demography.cpp.o.provides

schema/CMakeFiles/schema.dir/demography.cpp.o.provides.build: schema/CMakeFiles/schema.dir/demography.cpp.o


schema/CMakeFiles/schema.dir/monitoring.cpp.o: schema/CMakeFiles/schema.dir/flags.make
schema/CMakeFiles/schema.dir/monitoring.cpp.o: schema/monitoring.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object schema/CMakeFiles/schema.dir/monitoring.cpp.o"
	cd /code/openmalaria/build/schema && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/schema.dir/monitoring.cpp.o -c /code/openmalaria/build/schema/monitoring.cpp

schema/CMakeFiles/schema.dir/monitoring.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/schema.dir/monitoring.cpp.i"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /code/openmalaria/build/schema/monitoring.cpp > CMakeFiles/schema.dir/monitoring.cpp.i

schema/CMakeFiles/schema.dir/monitoring.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/schema.dir/monitoring.cpp.s"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /code/openmalaria/build/schema/monitoring.cpp -o CMakeFiles/schema.dir/monitoring.cpp.s

schema/CMakeFiles/schema.dir/monitoring.cpp.o.requires:

.PHONY : schema/CMakeFiles/schema.dir/monitoring.cpp.o.requires

schema/CMakeFiles/schema.dir/monitoring.cpp.o.provides: schema/CMakeFiles/schema.dir/monitoring.cpp.o.requires
	$(MAKE) -f schema/CMakeFiles/schema.dir/build.make schema/CMakeFiles/schema.dir/monitoring.cpp.o.provides.build
.PHONY : schema/CMakeFiles/schema.dir/monitoring.cpp.o.provides

schema/CMakeFiles/schema.dir/monitoring.cpp.o.provides.build: schema/CMakeFiles/schema.dir/monitoring.cpp.o


schema/CMakeFiles/schema.dir/interventions.cpp.o: schema/CMakeFiles/schema.dir/flags.make
schema/CMakeFiles/schema.dir/interventions.cpp.o: schema/interventions.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object schema/CMakeFiles/schema.dir/interventions.cpp.o"
	cd /code/openmalaria/build/schema && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/schema.dir/interventions.cpp.o -c /code/openmalaria/build/schema/interventions.cpp

schema/CMakeFiles/schema.dir/interventions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/schema.dir/interventions.cpp.i"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /code/openmalaria/build/schema/interventions.cpp > CMakeFiles/schema.dir/interventions.cpp.i

schema/CMakeFiles/schema.dir/interventions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/schema.dir/interventions.cpp.s"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /code/openmalaria/build/schema/interventions.cpp -o CMakeFiles/schema.dir/interventions.cpp.s

schema/CMakeFiles/schema.dir/interventions.cpp.o.requires:

.PHONY : schema/CMakeFiles/schema.dir/interventions.cpp.o.requires

schema/CMakeFiles/schema.dir/interventions.cpp.o.provides: schema/CMakeFiles/schema.dir/interventions.cpp.o.requires
	$(MAKE) -f schema/CMakeFiles/schema.dir/build.make schema/CMakeFiles/schema.dir/interventions.cpp.o.provides.build
.PHONY : schema/CMakeFiles/schema.dir/interventions.cpp.o.provides

schema/CMakeFiles/schema.dir/interventions.cpp.o.provides.build: schema/CMakeFiles/schema.dir/interventions.cpp.o


schema/CMakeFiles/schema.dir/healthSystem.cpp.o: schema/CMakeFiles/schema.dir/flags.make
schema/CMakeFiles/schema.dir/healthSystem.cpp.o: schema/healthSystem.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object schema/CMakeFiles/schema.dir/healthSystem.cpp.o"
	cd /code/openmalaria/build/schema && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/schema.dir/healthSystem.cpp.o -c /code/openmalaria/build/schema/healthSystem.cpp

schema/CMakeFiles/schema.dir/healthSystem.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/schema.dir/healthSystem.cpp.i"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /code/openmalaria/build/schema/healthSystem.cpp > CMakeFiles/schema.dir/healthSystem.cpp.i

schema/CMakeFiles/schema.dir/healthSystem.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/schema.dir/healthSystem.cpp.s"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /code/openmalaria/build/schema/healthSystem.cpp -o CMakeFiles/schema.dir/healthSystem.cpp.s

schema/CMakeFiles/schema.dir/healthSystem.cpp.o.requires:

.PHONY : schema/CMakeFiles/schema.dir/healthSystem.cpp.o.requires

schema/CMakeFiles/schema.dir/healthSystem.cpp.o.provides: schema/CMakeFiles/schema.dir/healthSystem.cpp.o.requires
	$(MAKE) -f schema/CMakeFiles/schema.dir/build.make schema/CMakeFiles/schema.dir/healthSystem.cpp.o.provides.build
.PHONY : schema/CMakeFiles/schema.dir/healthSystem.cpp.o.provides

schema/CMakeFiles/schema.dir/healthSystem.cpp.o.provides.build: schema/CMakeFiles/schema.dir/healthSystem.cpp.o


schema/CMakeFiles/schema.dir/entomology.cpp.o: schema/CMakeFiles/schema.dir/flags.make
schema/CMakeFiles/schema.dir/entomology.cpp.o: schema/entomology.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object schema/CMakeFiles/schema.dir/entomology.cpp.o"
	cd /code/openmalaria/build/schema && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/schema.dir/entomology.cpp.o -c /code/openmalaria/build/schema/entomology.cpp

schema/CMakeFiles/schema.dir/entomology.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/schema.dir/entomology.cpp.i"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /code/openmalaria/build/schema/entomology.cpp > CMakeFiles/schema.dir/entomology.cpp.i

schema/CMakeFiles/schema.dir/entomology.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/schema.dir/entomology.cpp.s"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /code/openmalaria/build/schema/entomology.cpp -o CMakeFiles/schema.dir/entomology.cpp.s

schema/CMakeFiles/schema.dir/entomology.cpp.o.requires:

.PHONY : schema/CMakeFiles/schema.dir/entomology.cpp.o.requires

schema/CMakeFiles/schema.dir/entomology.cpp.o.provides: schema/CMakeFiles/schema.dir/entomology.cpp.o.requires
	$(MAKE) -f schema/CMakeFiles/schema.dir/build.make schema/CMakeFiles/schema.dir/entomology.cpp.o.provides.build
.PHONY : schema/CMakeFiles/schema.dir/entomology.cpp.o.provides

schema/CMakeFiles/schema.dir/entomology.cpp.o.provides.build: schema/CMakeFiles/schema.dir/entomology.cpp.o


schema/CMakeFiles/schema.dir/pharmacology.cpp.o: schema/CMakeFiles/schema.dir/flags.make
schema/CMakeFiles/schema.dir/pharmacology.cpp.o: schema/pharmacology.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object schema/CMakeFiles/schema.dir/pharmacology.cpp.o"
	cd /code/openmalaria/build/schema && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/schema.dir/pharmacology.cpp.o -c /code/openmalaria/build/schema/pharmacology.cpp

schema/CMakeFiles/schema.dir/pharmacology.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/schema.dir/pharmacology.cpp.i"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /code/openmalaria/build/schema/pharmacology.cpp > CMakeFiles/schema.dir/pharmacology.cpp.i

schema/CMakeFiles/schema.dir/pharmacology.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/schema.dir/pharmacology.cpp.s"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /code/openmalaria/build/schema/pharmacology.cpp -o CMakeFiles/schema.dir/pharmacology.cpp.s

schema/CMakeFiles/schema.dir/pharmacology.cpp.o.requires:

.PHONY : schema/CMakeFiles/schema.dir/pharmacology.cpp.o.requires

schema/CMakeFiles/schema.dir/pharmacology.cpp.o.provides: schema/CMakeFiles/schema.dir/pharmacology.cpp.o.requires
	$(MAKE) -f schema/CMakeFiles/schema.dir/build.make schema/CMakeFiles/schema.dir/pharmacology.cpp.o.provides.build
.PHONY : schema/CMakeFiles/schema.dir/pharmacology.cpp.o.provides

schema/CMakeFiles/schema.dir/pharmacology.cpp.o.provides.build: schema/CMakeFiles/schema.dir/pharmacology.cpp.o


schema/CMakeFiles/schema.dir/util.cpp.o: schema/CMakeFiles/schema.dir/flags.make
schema/CMakeFiles/schema.dir/util.cpp.o: schema/util.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building CXX object schema/CMakeFiles/schema.dir/util.cpp.o"
	cd /code/openmalaria/build/schema && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/schema.dir/util.cpp.o -c /code/openmalaria/build/schema/util.cpp

schema/CMakeFiles/schema.dir/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/schema.dir/util.cpp.i"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /code/openmalaria/build/schema/util.cpp > CMakeFiles/schema.dir/util.cpp.i

schema/CMakeFiles/schema.dir/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/schema.dir/util.cpp.s"
	cd /code/openmalaria/build/schema && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /code/openmalaria/build/schema/util.cpp -o CMakeFiles/schema.dir/util.cpp.s

schema/CMakeFiles/schema.dir/util.cpp.o.requires:

.PHONY : schema/CMakeFiles/schema.dir/util.cpp.o.requires

schema/CMakeFiles/schema.dir/util.cpp.o.provides: schema/CMakeFiles/schema.dir/util.cpp.o.requires
	$(MAKE) -f schema/CMakeFiles/schema.dir/build.make schema/CMakeFiles/schema.dir/util.cpp.o.provides.build
.PHONY : schema/CMakeFiles/schema.dir/util.cpp.o.provides

schema/CMakeFiles/schema.dir/util.cpp.o.provides.build: schema/CMakeFiles/schema.dir/util.cpp.o


# Object files for target schema
schema_OBJECTS = \
"CMakeFiles/schema.dir/scenario.cpp.o" \
"CMakeFiles/schema.dir/demography.cpp.o" \
"CMakeFiles/schema.dir/monitoring.cpp.o" \
"CMakeFiles/schema.dir/interventions.cpp.o" \
"CMakeFiles/schema.dir/healthSystem.cpp.o" \
"CMakeFiles/schema.dir/entomology.cpp.o" \
"CMakeFiles/schema.dir/pharmacology.cpp.o" \
"CMakeFiles/schema.dir/util.cpp.o"

# External object files for target schema
schema_EXTERNAL_OBJECTS =

schema/libschema.a: schema/CMakeFiles/schema.dir/scenario.cpp.o
schema/libschema.a: schema/CMakeFiles/schema.dir/demography.cpp.o
schema/libschema.a: schema/CMakeFiles/schema.dir/monitoring.cpp.o
schema/libschema.a: schema/CMakeFiles/schema.dir/interventions.cpp.o
schema/libschema.a: schema/CMakeFiles/schema.dir/healthSystem.cpp.o
schema/libschema.a: schema/CMakeFiles/schema.dir/entomology.cpp.o
schema/libschema.a: schema/CMakeFiles/schema.dir/pharmacology.cpp.o
schema/libschema.a: schema/CMakeFiles/schema.dir/util.cpp.o
schema/libschema.a: schema/CMakeFiles/schema.dir/build.make
schema/libschema.a: schema/CMakeFiles/schema.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/code/openmalaria/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Linking CXX static library libschema.a"
	cd /code/openmalaria/build/schema && $(CMAKE_COMMAND) -P CMakeFiles/schema.dir/cmake_clean_target.cmake
	cd /code/openmalaria/build/schema && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/schema.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
schema/CMakeFiles/schema.dir/build: schema/libschema.a

.PHONY : schema/CMakeFiles/schema.dir/build

schema/CMakeFiles/schema.dir/requires: schema/CMakeFiles/schema.dir/scenario.cpp.o.requires
schema/CMakeFiles/schema.dir/requires: schema/CMakeFiles/schema.dir/demography.cpp.o.requires
schema/CMakeFiles/schema.dir/requires: schema/CMakeFiles/schema.dir/monitoring.cpp.o.requires
schema/CMakeFiles/schema.dir/requires: schema/CMakeFiles/schema.dir/interventions.cpp.o.requires
schema/CMakeFiles/schema.dir/requires: schema/CMakeFiles/schema.dir/healthSystem.cpp.o.requires
schema/CMakeFiles/schema.dir/requires: schema/CMakeFiles/schema.dir/entomology.cpp.o.requires
schema/CMakeFiles/schema.dir/requires: schema/CMakeFiles/schema.dir/pharmacology.cpp.o.requires
schema/CMakeFiles/schema.dir/requires: schema/CMakeFiles/schema.dir/util.cpp.o.requires

.PHONY : schema/CMakeFiles/schema.dir/requires

schema/CMakeFiles/schema.dir/clean:
	cd /code/openmalaria/build/schema && $(CMAKE_COMMAND) -P CMakeFiles/schema.dir/cmake_clean.cmake
.PHONY : schema/CMakeFiles/schema.dir/clean

schema/CMakeFiles/schema.dir/depend: schema/scenario.cpp
schema/CMakeFiles/schema.dir/depend: schema/scenario.h
schema/CMakeFiles/schema.dir/depend: schema/demography.cpp
schema/CMakeFiles/schema.dir/depend: schema/demography.h
schema/CMakeFiles/schema.dir/depend: schema/monitoring.cpp
schema/CMakeFiles/schema.dir/depend: schema/monitoring.h
schema/CMakeFiles/schema.dir/depend: schema/interventions.cpp
schema/CMakeFiles/schema.dir/depend: schema/interventions.h
schema/CMakeFiles/schema.dir/depend: schema/healthSystem.cpp
schema/CMakeFiles/schema.dir/depend: schema/healthSystem.h
schema/CMakeFiles/schema.dir/depend: schema/entomology.cpp
schema/CMakeFiles/schema.dir/depend: schema/entomology.h
schema/CMakeFiles/schema.dir/depend: schema/pharmacology.cpp
schema/CMakeFiles/schema.dir/depend: schema/pharmacology.h
schema/CMakeFiles/schema.dir/depend: schema/util.cpp
schema/CMakeFiles/schema.dir/depend: schema/util.h
	cd /code/openmalaria/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /code/openmalaria /code/openmalaria/schema /code/openmalaria/build /code/openmalaria/build/schema /code/openmalaria/build/schema/CMakeFiles/schema.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : schema/CMakeFiles/schema.dir/depend

