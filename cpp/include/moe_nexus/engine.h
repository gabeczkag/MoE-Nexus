#pragma once

#include <vector>
#include <string>
#include <chrono>
#include "moe_nexus/tokenizer.h"
#include "moe_nexus/model.h"
#include "moe_nexus/router.h"
#include "moe_nexus/load_balancer.h"

namespace moe_nexus {

struct GenerationConfig {
    int max_new_tokens = 256;
    float temperature = 1.0f;
    float top_p = 0.9f;
    int top_k = 50;
    float repetition_penalty = 1.0f;
    bool do_sample = true;
    int pad_token_id = -1;
    int eos_token_id = -1;
    int max_steps = 1000;
};

struct BenchmarkResult {
    float total_time_s = 0.0f;
    float tokens_per_second = 0.0f;
    float latency_per_token_ms = 0.0f;
    int total_tokens = 0;
};

class InferenceEngine {
public:
    InferenceEngine(
        MoEModelPtr model,
        std::shared_ptr<NumberTokenizer> tokenizer,
        std::shared_ptr<LoadBalancer> load_balancer = nullptr
    );
    
    // Generate tokens autoregressively
    std::vector<int> generate(const std::vector<int>& input_ids, const GenerationConfig& config);
    
    // Generate text (decode after generation)
    std::string generate_text(const std::string& prompt, const GenerationConfig& config);
    
    // Benchmark
    BenchmarkResult benchmark(
        const std::vector<int>& input_ids,
        const GenerationConfig& config,
        int warmup_steps = 3,
        int measure_steps = 10
    );
    
    MoEModel& get_model() { return *model_; }
    const MoEModel& get_model() const { return *model_; }
    NumberTokenizer& get_tokenizer() { return *tokenizer_; }
    const NumberTokenizer& get_tokenizer() const { return *tokenizer_; }

private:
    MoEModelPtr model_;
    std::shared_ptr<NumberTokenizer> tokenizer_;
    std::shared_ptr<LoadBalancer> load_balancer_;
    
    // Apply top-p filtering
    std::vector<float> apply_top_p_filtering(const std::vector<float>& logits, float top_p, int top_k) const;
    
    // Sample from logits
    int sample_token(const std::vector<float>& logits, float temperature) const;
};

} // namespace moe_nexus
