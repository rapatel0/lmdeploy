# Copyright (c) OpenMMLab. All rights reserved.

add_subdirectory(kernel)

add_library(attention STATIC
            attention.cu
            decoding.cu
            dflash_tilelang_combine.cu
            dflash_tilelang_partial.cu
            kv_cache_utils_v2.cu
            cp_utils.cu
            registry.cu
            )
set_property(TARGET attention PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(attention PRIVATE -O3
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math --expt-relaxed-constexpr -Xptxas=-v --threads 16>)
target_link_libraries(attention PUBLIC $<LINK_LIBRARY:WHOLE_ARCHIVE,attention_kernels>)
target_link_libraries(attention PRIVATE nvidia::cutlass::cutlass)

if (BUILD_TEST)
    target_compile_options(attention PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v --generate-line-info>)

    # test_utils.cu is deliberately absent. gemm2 compiles
    # gemm/test/test_utils.cu, which defines the same turbomind symbols with
    # identical declarations, and whole-archiving models force-includes it.
    # Compiling the attention copy as well is a duplicate definition of
    # curand_bytes, RNG::GenerateUInt and LoadBinary.
    add_executable(test_attention
        test_attention.cu
        reference.cu)
    target_compile_options(test_attention PRIVATE
        --generate-line-info -O3 -use_fast_math --expt-relaxed-constexpr)
    target_link_libraries(test_attention PRIVATE
        attention
        # flash_attention
        nvidia::cutlass::cutlass
        # models PUBLIC-links gemm2, and gemm2 whole-archives the
        # per-architecture kernel libraries. Those force-included objects carry
        # KernelImpl vtables that reference the gemm::Kernel base class, whose
        # definition lives in kernel.cu inside libgemm2.a. A plain static link
        # is resolved left to right without rescanning, so libgemm2.a is
        # already consumed by the time those references appear.
        #
        # Whole-archive models, matching how the shipped turbomind target links
        # it in src/turbomind/CMakeLists.txt.
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,models>"
        # Force-including every models object also pulls in translation units
        # the fixture never calls, such as LlamaFfnLayer, so the whole set of
        # dependencies the shipped turbomind target carries must be present.
        device_comm
        host_comm
        memory
        copy
        memory_utils
        nvtx_utils
        unfused_attention_kernels
        core
        cuda_utils
        fmt::fmt
        cublas
        CUDA::cublasLt
        CUDA::cudart)

    add_executable(test_quant test_quant.cu test_utils.cu)
    target_compile_options(test_quant PRIVATE
        --generate-line-info -O3 -use_fast_math --expt-relaxed-constexpr)
    target_link_libraries(test_quant PRIVATE
        nvidia::cutlass::cutlass
    )
endif ()
