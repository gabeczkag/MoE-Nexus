#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <functional>
#include "moe_nexus/tokenizer.h"

namespace moe_nexus {

struct RouterConfig {
    int num_experts = 8;
    int top_k = 2;
    float noise_std = 0.1f;
    bool use_aux_loss = true;
    float aux_loss_weight = 0.01f;
    int hidden_dim = 64;
};

struct RouterOutput {
    std::vector<float> scores;      // [batch * seq_len * top_k]
    std::vector<int> indices;       // [batch * seq_len * top_k]
    float aux_loss = 0.0f;
    int batch_size = 0;
    int seq_len = 0;
};

class Router {
public:
    explicit Router(const RouterConfig& config) : config_(config) {}
    virtual ~Router() = default;
    
    virtual RouterOutput forward(const float* hidden_states, int batch_size, int seq_len) const = 0;
    virtual size_t get_num_experts() const { return config_.num_experts; }
    virtual int get_top_k() const { return config_.top_k; }

protected:
    RouterConfig config_;
    
    static void softmax(float* logits, int n);
    static void select_top_k(const float* scores, int n, int k, float* out_scores, int* out_indices);
    static float compute_z_loss(const float* logits, int batch_seq_len, int num_experts);
};

class TopKRouter : public Router {
public:
    explicit TopKRouter(const RouterConfig& config);
    RouterOutput forward(const float* hidden_states, int batch_size, int seq_len) const override;
    
    void set_weights(const float* gate_weights); // [hidden_dim * num_experts]
    const std::vector<float>& get_weights() const { return gate_weights_; }

private:
    std::vector<float> gate_weights_; // [hidden_dim * num_experts]
};

class ExpertChoiceRouter : public Router {
public:
    explicit ExpertChoiceRouter(const RouterConfig& config);
    RouterOutput forward(const float* hidden_states, int batch_size, int seq_len) const override;
    
    void set_weights(const float* gate_weights);
    const std::vector<float>& get_weights() const { return gate_weights_; }

private:
    std::vector<float> gate_weights_;
};

} // namespace moe_nexus
