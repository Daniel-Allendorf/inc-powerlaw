cmake_minimum_required(VERSION 3.10)

# we first give our project a name
project(uniform-pld-sampling)

# prohibit in-source builds
if("../.." STREQUAL "${PROJECT_BINARY_DIR}")
    message(SEND_ERROR "In-source builds are not allowed.")
endif()

if (APPLE)
    FIND_PACKAGE(Boost REQUIRED)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

find_package(OpenMP REQUIRED)
option(ACCEPT_REJECT_LOG "Add benchmark which also outputs the accept/reject log" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Woverloaded-virtual -Wunused-variable -Wunused-parameter") # enable warnings (always good)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17") # enable C++17
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -msse4")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcx16")

# debug
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

# release
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mtune=native -msse4")

# profiling
set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_PROFILE} -O2 -g -fno-omit-frame-pointer -DNDEBUG -fno-inline-functions -fno-inline-functions-called-once -fno-optimize-sibling-calls")

# add sources to library
set(LIB_UNIFORM_SAMPLING_SRC
        include/incpwl/IncPowerlawGraphSampler.cpp
        include/incpwl/ConfigurationModel.cpp
        include/incpwl/DegreeSequenceHelper.cpp
        include/incpwl/AdjacencyVector.cpp
        include/incpwl/PowerlawDegreeSequence.cpp
)

add_library(libuniformpldsampling STATIC ${LIB_UNIFORM_SAMPLING_SRC})
target_include_directories(libuniformpldsampling PUBLIC include/ PRIVATE extlibs/shuffle/include extlibs/ska_sort/include extlibs/ips4o/include)
target_link_libraries(libuniformpldsampling PUBLIC tlx range-v3 OpenMP::OpenMP_CXX PRIVATE ips4o)

if (ACCEPT_REJECT_LOG)
    add_library(libuniformpldsampling-arl STATIC ${LIB_UNIFORM_SAMPLING_SRC})
    target_include_directories(libuniformpldsampling-arl PUBLIC include/ PRIVATE extlibs/shuffle/include extlibs/ska_sort/include extlibs/ips4o/include)
    target_link_libraries(libuniformpldsampling-arl PUBLIC tlx range-v3 OpenMP::OpenMP_CXX PRIVATE ips4o)
    target_compile_definitions(libuniformpldsampling-arl PRIVATE UNIFORM_PLD_SAMPLING_LOG_ACCEPT_REJECT)
endif()

# add third-party dependencies
add_subdirectory(extlibs/googletest)
add_subdirectory(extlibs/range-v3)
add_subdirectory(extlibs/tlx)
add_subdirectory(extlibs/ips4o)

# benchmarks and tests
enable_testing()
add_subdirectory(benchmark)
add_subdirectory(test)

add_executable(generator src/generator_main.cpp)
target_link_libraries(generator libuniformpldsampling tlx)

# apply CXXFLAGS to our configuration
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${STXXL_CXX_FLAGS}")
