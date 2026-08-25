// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/data_format.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/logger.h"

namespace turbomind {

bool DataFormat::is_quantized() const noexcept
{
    if (scales.present() || zeros.present()) {
        return true;
    }
    for (int bs : block_sizes) {
        if (bs > 1) {
            return true;
        }
    }
    return false;
}

DataFormat ResolveLinearWeightFormat(DataType data_type, DataType weight_dtype, int block_in, int block_out)
{
    DataFormat fmt;
    fmt.dtype = weight_dtype;

    if (IsTrivialFloatType(weight_dtype)) {
        TM_CHECK(block_in == 1 && block_out == 1)
            << "Trivial float weight requires block_in==1 and block_out==1, got " << block_in << ", " << block_out;
        fmt.block_sizes = {1, 1};
        return fmt;
    }

    if (weight_dtype == kFloat8_e4m3) {
        // Two accepted forms.
        //
        // {128, 128} is the native two-dimensional block scale that the
        // checkpoint ships. MakeQuantDesc maps it to QuantType::kB, which only
        // SM90 implements.
        //
        // {128, 1} is the K-grouped form the loader produces when it expands
        // the block scale along N. MakeQuantDesc maps it to QuantType::kK,
        // which the SM70 Config_E4M3 tiles implement. The expansion is exact,
        // because every weight element still receives the scale of its own
        // block. Scales stay 16-bit in that form, which matches the
        // kFloat8_e4m3 case in the conv_s branch of LinearWeight::prepare.
        TM_CHECK(block_in == 128 && (block_out == 128 || block_out == 1))
            << "FP8 weight format requires block_in==128 and block_out in {128, 1}, got " << block_in << ", "
            << block_out;
        fmt.block_sizes  = {128, block_out};
        fmt.scales.dtype = block_out == 128 ? kFloat : data_type;
        return fmt;
    }

    if (weight_dtype == kFloat4_e2m1) {
        TM_CHECK(block_in > 0 && block_out == 1)
            << "FP4 weight format requires block_in>0 and block_out==1, got " << block_in << ", " << block_out;
        fmt.block_sizes  = {block_in, 1};
        fmt.scales.dtype = kUint8;
        return fmt;
    }

    const bool is_qweight = weight_dtype == kUint4 || weight_dtype == kUint8;
    if (is_qweight) {
        TM_CHECK(block_in > 0 && block_in <= 256 && block_out == 1)
            << "Quantized integer weight requires 0 < block_in <= 256 and block_out==1, got " << block_in << ", "
            << block_out;
        fmt.block_sizes  = {block_in, 1};
        fmt.scales.dtype = data_type;
        fmt.zeros.dtype  = data_type;
        return fmt;
    }

    TM_LOG_FATAL("Unsupported weight format: {}", to_string(weight_dtype));
    return fmt;
}

}  // namespace turbomind
