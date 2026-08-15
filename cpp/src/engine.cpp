#include "moe_nexus/engine.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cstdio>

namespace moe_nexus {

InferenceEngine::InferenceEngine(
    MoEModelPtr model,
    std::shared_ptr<NumberTokenizer> tokenizer,
    std::shared_ptr<LoadBalancer> load_balancer
) : model_(std::move(model)),
    tokenizer_(std::move(tokenizer)),
    load_balancer_(std::move(load_balancer)) {
    if (!model_ || !tokenizer_) {
        throw std::invalid_argument("Model and tokenizer must not be null");
    }
}

std::vector<int> InferenceEngine::generate(const std::vector<int>& input_ids, const GenerationConfig& config) {
    if (input_ids.empty()) {
        return {};
    }
    
    std::vector<int> generated = input_ids;
    generated.reserve(generated.size() + config.max_new_tokens);
    
    if (load_balancer_) {
        load_balancer_->reset();
    }
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    volatile float sink = 0.0f;
    
    for (int step = 0; step < config.max_new_tokens; ++step) {
        std::vector<int> current_input = generated;
        
        auto [logits, aux_loss] = model_->forward(current_input.data(), 1, static_cast<int>(current_input.size()), false);
        
        for (size_t i = 0; i < logits.size(); ++i) {
            sink += logits[i];
        }
        
        if (config.temperature != 1.0f) {
            for (float& logit : logits) {
                logit /= config.temperature;
            }
        }
        
        std::vector<float> filtered_logits = apply_top_p_filtering(logits, config.top_p, config.top_k);
        
        int next_token = sample_token(filtered_logits, config.temperature);
        generated.push_back(next_token);
        
        if (step >= config.max_steps) {
            break;
        }
        
        if (config.eos_token_id >= 0 && next_token == config.eos_token_id) {
            break;
        }
    }
    
    return generated;
}

std::string InferenceEngine::generate_text(const std::string& prompt, const GenerationConfig& config) {
    std::vector<int> input_ids = tokenizer_->encode(prompt, true, false);
    std::vector<int> output_ids = generate(input_ids, config);
    return tokenizer_->decode(output_ids.data(), output_ids.size());
}

BenchmarkResult InferenceEngine::benchmark(
    const std::vector<int>& input_ids,
    const GenerationConfig& config,
    int warmup_steps,
    int measure_steps
) {
    BenchmarkResult result;
    if (input_ids.empty() || config.max_new_tokens <= 0) {
        return result;
    }
    
    for (int i = 0; i < warmup_steps; ++i) {
        generate(input_ids, config);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    int total_tokens_generated = 0;
    volatile float global_sink = 0.0f;
    
    for (int i = 0; i < measure_steps; ++i) {
        std::vector<int> output = generate(input_ids, config);
        total_tokens_generated += static_cast<int>(output.size() - input_ids.size());
        
        for (int token : output) {
            global_sink += static_cast<float>(token);
        }
    }
    std::cout << "[BENCHMARK] global_sink=" << global_sink << std::endl;
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    
    result.total_time_s = elapsed.count();
    result.total_tokens = total_tokens_generated;
    
    if (result.total_time_s > 0.0f) {
        result.tokens_per_second = total_tokens_generated / result.total_time_s;
        result.latency_per_token_ms = (result.total_time_s / total_tokens_generated) * 1000.0f;
    }
    
    return result;
}

std::vector<float> InferenceEngine::apply_top_p_filtering(
    const std::vector<float>& logits,
    float top_p,
    int top_k
) const {
    if (top_p <= 0.0f || top_p >= 1.0f) {
        return logits;
    }
    
    std::vector<std::pair<float, int>> indexed;
    indexed.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        indexed.emplace_back(logits[i], static_cast<int>(i));
    }
    
    std::sort(indexed.begin(), indexed.end(), std::greater<std::pair<float, int>>());
    
    std::vector<float> sorted_logits;
    sorted_logits.reserve(indexed.size());
    for (const auto& [logit, _] : indexed) {
        sorted_logits.push_back(logit);
    }
    
    float max_logit = sorted_logits[0];
    for (float& logit : sorted_logits) {
        if (logit > max_logit) max_logit = logit;
    }
    
    float sum = 0.0f;
    for (float& logit : sorted_logits) {
        logit = std::exp(logit - max_logit);
        sum += logit;
    }
    
    if (sum > 0.0f) {
        for (float& logit : sorted_logits) {
            logit /= sum;
        }
    }
    
    float cumulative = 0.0f;
    int cutoff = static_cast<int>(sorted_logits.size());
    for (size_t i = 0; i < sorted_logits.size(); ++i) {
        cumulative += sorted_logits[i];
        if (cumulative >= top_p) {
            cutoff = static_cast<int>(i) + 1;
            break;
        }
    }
    
    std::vector<float> filtered(logits.size(), -1e9f);
    for (int i = 0; i < cutoff; ++i) {
        filtered[indexed[i].second] = logits[indexed[i].second];
    }
    
    return filtered;
}

int InferenceEngine::sample_token(const std::vector<float>& logits, float temperature) const {
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    
    if (sum > 0.0f) {
        for (float& p : probs) {
            p /= sum;
        }
    }
    
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    
    volatile float sink = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        sink += probs[i];
        if (r <= sink) {
            return static_cast<int>(i);
        }
    }
    
    return static_cast<int>(logits.size()) - 1;
}

} // namespace moe_nexus
