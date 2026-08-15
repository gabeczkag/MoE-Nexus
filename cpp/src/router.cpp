#include "moe_nexus/router.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace moe_nexus {

// ============================================================================
// TopKRouter Implementation
// ============================================================================

TopKRouter::TopKRouter(const RouterConfig& config) : Router(config) {
    if (config.hidden_dim <= 0 || config.num_experts <= 0 || config.top_k <= 0) {
        throw std::invalid_argument("Invalid router configuration");
    }
    gate_weights_.resize(config.hidden_dim * config.num_experts);
}

RouterOutput TopKRouter::forward(const float* hidden_states, int batch_size, int seq_len) const {
    RouterOutput output;
    output.batch_size = batch_size;
    output.seq_len = seq_len;
    
    int hidden = config_.hidden_dim;
    int experts = config_.num_experts;
    int top_k = config_.top_k;
    int total = batch_size * seq_len;
    
    output.scores.resize(total * top_k);
    output.indices.resize(total * top_k);
    
    // Temporary buffer for gate logits
    std::vector<float> gate_logits(total * experts);
    
    // Gate forward: hidden_states [total, hidden] x gate_weights [hidden, experts] -> [total, experts]
    for (int i = 0; i < total; ++i) {
        const float* hidden_vec = hidden_states + i * hidden;
        float* logits = gate_logits.data() + i * experts;
        
        for (int e = 0; e < experts; ++e) {
            float sum = 0.0f;
            const float* gate_e = gate_weights_.data() + static_cast<size_t>(e) * hidden;
            for (int h = 0; h < hidden; ++h) {
                sum += hidden_vec[h] * gate_e[h];
            }
            logits[e] = sum;
        }
    }
    
    // Softmax + top-k for each position
    for (int i = 0; i < total; ++i) {
        float* logits = gate_logits.data() + i * experts;
        
        softmax(logits, experts);
        
        float* scores = output.scores.data() + i * top_k;
        int* indices = output.indices.data() + i * top_k;
        
        select_top_k(logits, experts, top_k, scores, indices);
        
        // Normalize scores
        float score_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) score_sum += scores[k];
        if (score_sum > 0.0f) {
            for (int k = 0; k < top_k; ++k) scores[k] /= score_sum;
        }
    }
    
    // Auxiliary loss
    if (config_.use_aux_loss) {
        output.aux_loss = compute_z_loss(gate_logits.data(), total, experts) * config_.aux_loss_weight;
    }
    
    return output;
}

// ============================================================================
// ExpertChoiceRouter Implementation
// ============================================================================

ExpertChoiceRouter::ExpertChoiceRouter(const RouterConfig& config) : Router(config) {
    if (config.hidden_dim <= 0 || config.num_experts <= 0 || config.top_k <= 0) {
        throw std::invalid_argument("Invalid router configuration");
    }
    gate_weights_.resize(config.hidden_dim * config.num_experts);
}

RouterOutput ExpertChoiceRouter::forward(const float* hidden_states, int batch_size, int seq_len) const {
    RouterOutput output;
    output.batch_size = batch_size;
    output.seq_len = seq_len;
    
    int hidden = config_.hidden_dim;
    int experts = config_.num_experts;
    int top_k = config_.top_k;
    int total = batch_size * seq_len;
    
    output.scores.resize(total * top_k);
    output.indices.resize(total * top_k);
    
    // Gate forward (same as TopKRouter)
    std::vector<float> gate_logits(total * experts);
    
    for (int i = 0; i < total; ++i) {
        const float* hidden_vec = hidden_states + i * hidden;
        float* logits = gate_logits.data() + i * experts;
        
        for (int e = 0; e < experts; ++e) {
            float sum = 0.0f;
            const float* gate_e = gate_weights_.data() + static_cast<size_t>(e) * hidden;
            for (int h = 0; h < hidden; ++h) {
                sum += hidden_vec[h] * gate_e[h];
            }
            logits[e] = sum;
        }
        
        softmax(logits, experts);
    }
    
    // ExpertChoice: select top-k experts globally (token-level)
    for (int i = 0; i < total; ++i) {
        float* logits = gate_logits.data() + i * experts;
        float* scores = output.scores.data() + i * top_k;
        int* indices = output.indices.data() + i * top_k;
        
        select_top_k(logits, experts, top_k, scores, indices);
        
        float score_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) score_sum += scores[k];
        if (score_sum > 0.0f) {
            for (int k = 0; k < top_k; ++k) scores[k] /= score_sum;
        }
    }
    
    return output;
}

// ============================================================================
// Router static helper implementations
// ============================================================================

void Router::softmax(float* logits, int n) {
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

void Router::select_top_k(const float* scores, int n, int k, float* out_scores, int* out_indices) {
    for (int i = 0; i < k; ++i) {
        out_scores[i] = -1.0f;
        out_indices[i] = -1;
        for (int j = 0; j < n; ++j) {
            bool skip = false;
            for (int ii = 0; ii < i; ++ii) {
                if (out_indices[ii] == j) {
                    skip = true;
                    break;
                }
            }
            if (!skip && scores[j] > out_scores[i]) {
                out_scores[i] = scores[j];
                out_indices[i] = j;
            }
        }
    }
}

float Router::compute_z_loss(const float* logits, int batch_seq_len, int num_experts) {
    float sum = 0.0f;
    for (int i = 0; i < batch_seq_len * num_experts; ++i) {
        sum += logits[i] * logits[i];
    }
    return sum / (batch_seq_len * num_experts);
}

} // namespace moe_nexus
