#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include "moe_nexus/tokenizer.h"
#include "moe_nexus/router.h"

namespace moe_nexus {

struct ModelConfig {
    int vocab_size = 260;
    int hidden_dim = 64;
    int num_experts = 8;
    int top_k = 2;
    int num_layers = 2;
    float noise_std = 0.1f;
    int max_seq_len = 64;
};

struct ModelOutput {
    std::vector<float> logits;
    float aux_loss = 0.0f;
};

class MoEModel {
public:
    explicit MoEModel(const ModelConfig& config);
    ~MoEModel() = default;
    
    // Forward pass: [batch, seq] -> [batch, seq, vocab_size]
    ModelOutput forward(
        const int* input_ids,
        int batch_size,
        int seq_len,
        bool training = false
    ) const;
    
    // Load/save weights
    bool save_weights(const std::string& path) const;
    bool load_weights(const std::string& path);
    
    const ModelConfig& get_config() const { return config_; }

private:
    ModelConfig config_;
    
    // Weight matrices stored as [out_dim x in_dim] (row-major)
    std::vector<float> embedding_weights_;   // [vocab_size x hidden_dim]
    std::vector<float> gate_weights_;        // [hidden_dim x num_experts]
    std::vector<float> head_weights_;        // [vocab_size x hidden_dim]
    
    // Expert weights: each expert is [hidden_dim x hidden_dim]
    std::vector<std::vector<float>> expert_weights_;
    
    // Helper: matrix-vector multiply: y = A * x (A is [out x in], x is [in])
    static void matmul_vec(const float* A, const float* x, float* y, int out_dim, int in_dim);
    
    // Helper: softmax
    static void softmax(float* logits, int n);
};

using MoEModelPtr = std::unique_ptr<MoEModel>;

} // namespace moe_nexus

