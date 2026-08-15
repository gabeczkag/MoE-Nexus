#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "moe_nexus/router.h"

namespace moe_nexus {

struct ExpertStats {
    int expert_id = 0;
    int total_tokens = 0;
    float total_weight = 0.0f;
    
    float avg_weight() const { return total_tokens > 0 ? total_weight / total_tokens : 0.0f; }
};

struct BalanceResult {
    float imbalance = 0.0f;
    float coefficient_of_variation = 0.0f;
    float max_utilization = 0.0f;
    float min_utilization = 0.0f;
    std::vector<std::string> suggestions;
};

class LoadBalancer {
public:
    explicit LoadBalancer(int num_experts, float capacity_factor = 1.25f);
    
    void record_routing(const RouterOutput& routing);
    BalanceResult analyze() const;
    void reset();
    
    int get_num_experts() const { return num_experts_; }
    const std::vector<ExpertStats>& get_stats() const { return stats_; }

private:
    int num_experts_ = 0;
    float capacity_factor_ = 1.25f;
    std::vector<ExpertStats> stats_;
};

} // namespace moe_nexus
