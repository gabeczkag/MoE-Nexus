#include "moe_nexus/load_balancer.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace moe_nexus {

LoadBalancer::LoadBalancer(int num_experts, float capacity_factor)
    : num_experts_(num_experts), capacity_factor_(capacity_factor) {
    stats_.resize(num_experts);
    for (int i = 0; i < num_experts; ++i) {
        stats_[i].expert_id = i;
    }
}

void LoadBalancer::record_routing(const RouterOutput& routing) {
    const int* indices = routing.indices.data();
    const float* scores = routing.scores.data();
    int total = static_cast<int>(routing.indices.size());
    
    for (int i = 0; i < total; ++i) {
        int expert_id = indices[i];
        if (expert_id >= 0 && expert_id < num_experts_) {
            ExpertStats& stat = stats_[expert_id];
            stat.total_tokens++;
            stat.total_weight += scores[i];
        }
    }
}

BalanceResult LoadBalancer::analyze() const {
    BalanceResult result;
    
    if (stats_.empty()) {
        return result;
    }
    
    std::vector<int> utils;
    utils.reserve(stats_.size());
    for (const auto& stat : stats_) {
        utils.push_back(stat.total_tokens);
    }
    
    float sum = 0.0f;
    for (int u : utils) sum += u;
    float mean = sum / utils.size();
    
    if (mean > 0.0f) {
        float variance = 0.0f;
        for (int u : utils) {
            float diff = u - mean;
            variance += diff * diff;
        }
        variance /= utils.size();
        float std = std::sqrt(variance);
        result.coefficient_of_variation = std / mean;
    }
    
    auto [min_it, max_it] = std::minmax_element(utils.begin(), utils.end());
    result.min_utilization = static_cast<float>(*min_it);
    result.max_utilization = static_cast<float>(*max_it);
    result.imbalance = result.max_utilization - result.min_utilization;
    
    // Generate suggestions
    if (result.coefficient_of_variation > 0.5f) {
        result.suggestions.push_back("High expert imbalance detected. Consider increasing noise during training.");
    }
    if (mean > 0.0f && result.max_utilization > mean * capacity_factor_) {
        result.suggestions.push_back("Capacity violations detected. Review expert capacity limits.");
    }
    if (mean > 0.0f && result.min_utilization < mean * 0.1f) {
        result.suggestions.push_back("Some experts are underutilized. Consider expert pruning or merging.");
    }
    
    return result;
}

void LoadBalancer::reset() {
    for (auto& stat : stats_) {
        stat.total_tokens = 0;
        stat.total_weight = 0.0f;
    }
}

} // namespace moe_nexus
