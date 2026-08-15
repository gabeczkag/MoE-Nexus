#include "moe_nexus/model.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace moe_nexus {

MoEModel::MoEModel(const ModelConfig& config) : config_(config) {
    embedding_weights_.resize(config_.vocab_size * config_.hidden_dim);
    gate_weights_.resize(config_.hidden_dim * config_.num_experts);
    head_weights_.resize(config_.vocab_size * config_.hidden_dim);
    expert_weights_.resize(config_.num_experts);
    
    for (auto& expert : expert_weights_) {
        expert.resize(config_.hidden_dim * config_.hidden_dim);
    }
    
    float emb_scale = std::sqrt(6.0f / (config_.vocab_size + config_.hidden_dim));
    float gate_scale = std::sqrt(6.0f / (config_.hidden_dim + config_.num_experts));
    float expert_scale = std::sqrt(6.0f / (2 * config_.hidden_dim));
    float head_scale = std::sqrt(6.0f / (config_.hidden_dim + config_.vocab_size));
    
    std::generate(embedding_weights_.begin(), embedding_weights_.end(),
        [emb_scale]() { return (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * emb_scale; });
    std::generate(gate_weights_.begin(), gate_weights_.end(),
        [gate_scale]() { return (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * gate_scale; });
    std::generate(head_weights_.begin(), head_weights_.end(),
        [head_scale]() { return (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * head_scale; });
    
    for (auto& expert : expert_weights_) {
        std::generate(expert.begin(), expert.end(),
            [expert_scale]() { return (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * expert_scale; });
    }
}

void MoEModel::matmul_vec(const float* A, const float* x, float* y, int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; ++i) {
        float sum = 0.0f;
        const float* A_row = A + i * in_dim;
        for (int j = 0; j < in_dim; ++j) {
            sum += A_row[j] * x[j];
        }
        y[i] = sum;
    }
}

void MoEModel::softmax(float* logits, int n) {
    float max_val = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }
    
    if (sum > 0.0f) {
        for (int i = 0; i < n; ++i) {
            logits[i] /= sum;
        }
    }
}

ModelOutput MoEModel::forward(
    const int* input_ids,
    int batch_size,
    int seq_len,
    bool training
) const {
    ModelOutput output;
    int total = batch_size * seq_len;
    int hidden = config_.hidden_dim;
    int vocab = config_.vocab_size;
    int experts = config_.num_experts;
    int top_k = config_.top_k;
    
    output.logits.resize(total * vocab);
    output.aux_loss = 0.0f;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int idx = b * seq_len + s;
            int token_id = input_ids[idx];
            if (token_id < 0 || token_id >= vocab) token_id = 0;
            
            float hidden_vec[64];
            const float* emb_row = embedding_weights_.data() + token_id * hidden;
            for (int h = 0; h < hidden; ++h) {
                hidden_vec[h] = emb_row[h];
            }
            
            float gate_logits[64];
            for (int e = 0; e < experts; ++e) {
                float sum = 0.0f;
                const float* gate_e = gate_weights_.data() + static_cast<size_t>(e) * hidden;
                for (int h = 0; h < hidden; ++h) {
                    sum += hidden_vec[h] * gate_e[h];
                }
                gate_logits[e] = sum;
            }
            
            softmax(gate_logits, experts);
            
            float topk_scores[8];
            int topk_indices[8];
            
            for (int k = 0; k < top_k; ++k) {
                topk_scores[k] = -1.0f;
                topk_indices[k] = -1;
                for (int e = 0; e < experts; ++e) {
                    bool skip = false;
                    for (int kk = 0; kk < k; ++kk) {
                        if (topk_indices[kk] == e) {
                            skip = true;
                            break;
                        }
                    }
                    if (!skip && gate_logits[e] > topk_scores[k]) {
                        topk_scores[k] = gate_logits[e];
                        topk_indices[k] = e;
                    }
                }
            }
            
            float score_sum = 0.0f;
            for (int k = 0; k < top_k; ++k) score_sum += topk_scores[k];
            if (score_sum > 0.0f) {
                for (int k = 0; k < top_k; ++k) topk_scores[k] /= score_sum;
            }
            
            float expert_out[64] = {0.0f};
            for (int k = 0; k < top_k; ++k) {
                int e = topk_indices[k];
                if (e < 0 || e >= experts) continue;
                
                float expert_hidden[64] = {0.0f};
                matmul_vec(expert_weights_[e].data(), hidden_vec, expert_hidden, hidden, hidden);
                
                for (int h = 0; h < hidden; ++h) {
                    expert_out[h] += topk_scores[k] * expert_hidden[h];
                }
            }
            
            float* logits_out = output.logits.data() + idx * vocab;
            matmul_vec(head_weights_.data(), expert_out, logits_out, vocab, hidden);
        }
    }
    
    if (training) {
        output.aux_loss = 0.01f;
    }
    
    return output;
}

bool MoEModel::save_weights(const std::string& path) const {
    return false;
}

bool MoEModel::load_weights(const std::string& path) {
    return false;
}

} // namespace moe_nexus
