#pragma once

#include <memory>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class ModelWeight;
struct Sequence;
class CacheRegistry;

class LanguageModel {
public:
    ~LanguageModel();

    LanguageModel() = default;

    LanguageModel(LanguageModel&&) noexcept;

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    LanguageModel(CacheRegistry&     registry,
                  const EngineParam& engine,
                  const Context&     context,
                  const ModelWeight& weights,
                  int                phases);

    void Run(BatchOp op, int phase, TensorMap& env);

    /// Does this phase's batch carry drafts that the next forward must verify?
    /// Answered here because BatchData holds no sequence pointers; the state is
    /// captured during Setup.
    bool HasDraftsToVerify(int phase) const;

    /// Is this phase's batch a real decode step, so drafting from it is valid?
    /// A final prefill chunk is `generating` but is not a decode step.
    bool CanDraft(int phase) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
