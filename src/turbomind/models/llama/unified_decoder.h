#pragma once

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/models/llama/GatedDeltaNetLayer.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"

namespace turbomind {

class ModelWeight;
class DecoderLayerWeight;
class CacheRegistry;

class UnifiedDecoder {
public:
    using WeightType = DecoderLayerWeight;

    UnifiedDecoder(CacheRegistry&     registry,
                   const EngineParam& engine,
                   const Context&     ctx,
                   int                phases,
                   const ModelWeight& model_weight);

    void Run(BatchOp op, int phase, TensorMap& env);

    void Forward(int phase, TensorMap& env, const std::vector<WeightType*>& weights);

    /// Attention-list index of the MTP draft layer, or -1 when absent.
    int mtp_attn_index() const noexcept
    {
        return mtp_attn_index_;
    }

    /// The attention layer instance, shared with MTPPredictor.
    ///
    /// The draft must use this instance rather than its own: the KV slots were
    /// registered here as one block, and only this object knows the resulting
    /// offsets. Returns nullptr for a model with no attention layers.
    UnifiedAttentionLayer* attn_layer() const noexcept
    {
        return attn_layer_.get();
    }

private:
    const size_t layer_num_;
    const size_t hidden_units_;
    const bool   output_norm_zero_centered_;

    const int attn_tp_size_;
    const int attn_dp_size_;
    const int attn_dp_rank_;
    const int mlp_tp_size_;

    const int attn_tp_group_;

    comm::DeviceCommImpl* const d_comm_;

    const int tune_layer_num_;

    /// Index of the Multi-Token Prediction draft layer inside the attention
    /// weight list, or -1 when the checkpoint carries no MTP layer.
    ///
    /// The draft layer is appended after the target's layers, so it receives
    /// its own `cache_block_offset` and therefore its own KV space.
    /// MTPPredictor needs this index to address that slot.
    int mtp_attn_index_{-1};

    int& is_warm_up_;

    std::unique_ptr<UnifiedAttentionLayer> attn_layer_;
    std::unique_ptr<GatedDeltaNetLayer>    linear_attn_layer_;
    std::unique_ptr<LlamaFfnLayer>         ffn_layer_;
    std::unique_ptr<MoeFfnLayer>           moe_ffn_layer_;

    void AllreduceResidualRMSnorm(Tensor&       hidden_states,
                                  Tensor&       residual,
                                  const Tensor& bias,
                                  const Tensor& weight,
                                  float         eps,
                                  bool          zero_centered,
                                  int           token_num,
                                  int           t0,
                                  int           t1,
                                  const int*    local_token_nums);
};

}  // namespace turbomind
