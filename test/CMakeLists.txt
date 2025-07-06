message(STATUS "Build tinyndarray tests")

# Executable file for tests (one c++ file)
add_executable(tinyndarray_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/test_basic.cpp
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utest.h)
target_link_libraries(tinyndarray_tests tinyndarray)
setup_target(tinyndarray_tests "" "${CMAKE_THREAD_LIBS_INIT}")
