add_executable(BenchmarkIncPowerlawGraphSampler BenchmarkIncPowerlawGraphSampler.cpp)
target_link_libraries(BenchmarkIncPowerlawGraphSampler libuniformpldsampling tlx)

if (ACCEPT_REJECT_LOG)
    add_executable(BenchmarkIncPowerlawGraphSamplerARL BenchmarkIncPowerlawGraphSampler.cpp)
    target_link_libraries(BenchmarkIncPowerlawGraphSamplerARL libuniformpldsampling-arl tlx)
endif()
