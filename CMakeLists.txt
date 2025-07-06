cmake_minimum_required(VERSION 3.10)
project(tinyndarray)

set(CMAKE_CXX_STANDARD 17)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(LINK_TYPE STATIC)
else()
    set(LINK_TYPE SHARED)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Print make commands for debug
# set(CMAKE_VERBOSE_MAKEFILE 1)

# Set default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
set(PROJECT_VERSION 1.0.0)
# Output `compile_commands.json`
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_FLAGS "-fsanitize=address")
set(CMAKE_CXX_FLAGS "-fsanitize=address")
# cmake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sanitizers)
find_package(Sanitizers) # Address sanitizer (-DSANITIZE_ADDRESS=ON)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Warning options
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(warning_options "-Wall -Wextra -Wconversion")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(warning_options "-Wall -Wextra -Wcast-align -Wcast-qual \
                         -Wctor-dtor-privacy -Wdisabled-optimization \
                         -Wformat=2 -Winit-self \
                         -Wmissing-declarations -Wmissing-include-dirs \
                         -Wold-style-cast -Woverloaded-virtual \
                         -Wredundant-decls -Wshadow -Wsign-conversion \
                         -Wsign-promo  -Wno-old-style-cast\
                         -Wstrict-overflow=5 -Wundef -Wno-unknown-pragmas \
                         -Wreturn-std-move")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(warning_options "--pedantic -Wall -Wextra -Wcast-align -Wcast-qual \
                         -Wctor-dtor-privacy -Wdisabled-optimization \
                         -Wformat=2 -Winit-self -Wlogical-op \
                         -Wmissing-declarations -Wmissing-include-dirs \
                         -Wnoexcept -Wold-style-cast -Woverloaded-virtual \
                         -Wredundant-decls -Wshadow -Wsign-conversion \
                         -Wsign-promo -Wstrict-null-sentinel \
                         -Wstrict-overflow=5 -Wundef -Wno-unknown-pragmas")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warning_options "/W4")
else()
    message(WARNING "Unsupported compiler for warning options")
    message("CMAKE_CXX_COMPILER_ID is ${CMAKE_CXX_COMPILER_ID}")
endif()

# Utility function to setup a target (include, link, warning, sanitizer)
function(setup_target target includes libs)
    target_include_directories(${target} PUBLIC ${includes})
    target_link_libraries(${target} ${libs})
    set_target_properties(${target} PROPERTIES COMPILE_FLAGS ${warning_options})
    add_sanitizers(${target})
endfunction(setup_target)

# Utility function to setup a target simply (include, link)
function(setup_target_simple target includes libs)
    target_include_directories(${target} PUBLIC ${includes})
    target_link_libraries(${target} ${libs})
endfunction(setup_target_simple)


# ------------------------------------------------------------------------------
# ----------------------------- Internal Libraries -----------------------------
# ------------------------------------------------------------------------------

find_package(Threads)

add_subdirectory(third_party/cnpy)
# 主库
add_library(tinyndarray STATIC
    src/core.cpp
    src/arithmetic.cpp
    src/shape_manipulation.cpp
    src/reduction.cpp
    src/random.cpp
    src/utils.cpp
    src/io.cpp
    src/tensor_ops.cpp
)

# 添加头文件目录
target_include_directories(tinyndarray
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# 链接 cnpy
target_link_libraries(tinyndarray PRIVATE cnpy)

# 定义实现宏（仅在库内部使用）
target_compile_definitions(tinyndarray PRIVATE
    TINYNDARRAY_IMPLEMENTATION
)

# 设置安装规则
install(
    TARGETS tinyndarray cnpy
    EXPORT tinyndarrayTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(
    DIRECTORY include/tinyndarray
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# 安装 cnpy 头文件
install(
    DIRECTORY third-party/cnpy/
    DESTINATION include/cnpy
    FILES_MATCHING 
        PATTERN "*.h"
        PATTERN "*.hpp"
)

install(
    EXPORT tinyndarrayTargets
    FILE tinyndarrayTargets.cmake
    NAMESPACE tinyndarray::
    DESTINATION lib/cmake/tinyndarray
)

# 创建配置文件
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/tinyndarrayConfig.cmake
    INSTALL_DESTINATION lib/cmake/tinyndarray
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/tinyndarrayConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/tinyndarrayConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/tinyndarrayConfigVersion.cmake
    DESTINATION lib/cmake/tinyndarray
)

# 测试
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
