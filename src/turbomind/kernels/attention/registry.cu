// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/registry.h"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/attention/arch.h"
#include "src/turbomind/kernels/attention/registrar.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::attention {

namespace {

constexpr float kMaxWasteRatio = 1.f;

}  // namespace

Registry::Registry(std::shared_ptr<cudaDeviceProp> device_prop):
    device_prop_{std::move(device_prop)}, arch_{device_prop_->major * 100 + device_prop_->minor * 10}
{
    for (auto& register_fn : gKernelFactories()) {
        Collector collector;
        register_fn(collector);
        for (auto& k : collector.release()) {
            Add(std::move(k));
        }
    }
}

bool Registry::Add(std::unique_ptr<Kernel> kernel)
{
    bool is_valid = true;

    if (!arch::is_arch_compatible(kernel->arch(), arch_)) {
        is_valid = false;
    }

    if ((int)device_prop_->sharedMemPerBlockOptin < kernel->smem_size()) {
        is_valid = false;
    }

    if (is_valid) {
        ptrs_.push_back(kernels_.emplace_back(std::move(kernel)).get());
    }

    return is_valid;
}

const Kernel* Registry::Find(const AttnDesc& desc) const
{
    const int threshold = static_cast<int>(kMaxWasteRatio * desc.query_group_sz);

    const Kernel*             best = nullptr;
    const Kernel*             prefill_fallback = nullptr;
    std::tuple<int, int, int> cost{};

    static const int sm70_128_q_tile = [] {
        const char* value = std::getenv("TM_ATTENTION_SM70_128_CTA_Q");
        if (!value) {
            return 64;
        }
        const int tile = std::atoi(value);
        return tile == 16 || tile == 32 || tile == 64 ? tile : 64;
    }();
    static const int sm70_128_kv_tile = [] {
        const char* value = std::getenv("TM_ATTENTION_SM70_128_CTA_KV");
        return value && std::atoi(value) == 32 ? 32 : 64;
    }();
    static const int sm70_256_q_tile = [] {
        const char* value = std::getenv("TM_ATTENTION_SM70_256_CTA_Q");
        if (!value) {
            return 64;
        }
        const int tile = std::atoi(value);
        return tile == 16 || tile == 32 || tile == 64 ? tile : 64;
    }();
    static const int sm70_256_kv_tile = [] {
        const char* value = std::getenv("TM_ATTENTION_SM70_256_CTA_KV");
        return value && std::atoi(value) == 32 ? 32 : 64;
    }();

    for (const auto* k : ptrs_) {
        const auto& d = k->desc();
        if (d.mode != desc.mode || d.head_dim != desc.head_dim  //
            || d.data_type != desc.data_type || d.kv_quant != desc.kv_quant || d.causal != desc.causal) {
            continue;
        }
        if (desc.mode == AttnDesc::kDecoding) {
            const int ctas  = cdiv(desc.query_group_sz, d.qh);
            const int waste = d.qh * ctas - desc.query_group_sz;

            const auto v = std::make_tuple(waste > threshold, ctas, waste);
            if (!best || v < cost) {
                best = k;
                cost = v;
            }
        }
        else {
            if (!prefill_fallback) {
                prefill_fallback = k;
            }
            if (arch_ == 700) {
                if (desc.head_dim == 128 && d.q_tile == sm70_128_q_tile && d.kv_tile == sm70_128_kv_tile) {
                    return k;
                }
                if (desc.head_dim == 256 && d.q_tile == sm70_256_q_tile && d.kv_tile == sm70_256_kv_tile) {
                    return k;
                }
            }
            if (arch_ != 700 || (desc.head_dim != 128 && desc.head_dim != 256)) {
                return k;
            }
        }
    }
    return desc.mode == AttnDesc::kPrefill ? prefill_fallback : best;
}

Registry& Registry::instance()
{
    struct DeviceState {
        std::unique_ptr<Registry> registry;
        std::once_flag            flag;
    };

    static std::vector<std::unique_ptr<DeviceState>> states = [] {
        int count{};
        TM_CUDA_CHECK(cudaGetDeviceCount(&count));
        std::vector<std::unique_ptr<DeviceState>> vec(count);
        for (auto& s : vec) {
            s = std::make_unique<DeviceState>();
        }
        return vec;
    }();

    int device_id{};
    TM_CUDA_CHECK(cudaGetDevice(&device_id));

    auto& state = *states.at(device_id);

    std::call_once(state.flag, [&]() {
        auto prop = std::make_shared<cudaDeviceProp>();
        TM_CUDA_CHECK(cudaGetDeviceProperties(prop.get(), device_id));
        state.registry = std::make_unique<Registry>(std::move(prop));
    });

    return *TM_CHECK_NOTNULL(state.registry);
}

}  // namespace turbomind::attention
