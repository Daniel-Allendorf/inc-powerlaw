# add executable test file
add_executable(uniform_pld_sampling_tests TestAdjacencyVector.cpp TestIncPowerlawGraphSampler.cpp TestRegression.cpp)

# include googletest
include_directories(${gtest_SOURCE_DIR} ${gtest_SOURCE_DIR}/include)

# link googletest to executable
target_link_libraries(uniform_pld_sampling_tests
                      libuniformpldsampling
                      tlx
                      gtest gtest_main)

add_test(UniformPLDSampling uniform_pld_sampling_tests)
