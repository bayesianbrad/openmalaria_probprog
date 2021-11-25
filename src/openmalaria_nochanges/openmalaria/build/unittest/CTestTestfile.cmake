# CMake generated Testfile for 
# Source directory: /code/openmalaria/unittest
# Build directory: /code/openmalaria/build/unittest
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(unittest "unittest" "-keep")
set_tests_properties(unittest PROPERTIES  PASS_REGULAR_EXPRESSION "^Running [0-9]+ tests.*OK![ 
]*\$")
