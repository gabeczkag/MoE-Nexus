#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include "moe_nexus/tokenizer.h"
#include "moe_nexus/router.h"
#include "moe_nexus/load_balancer.h"
#include "moe_nexus/model.h"
#include "moe_nexus/engine.h"

using namespace moe_nexus;

void test_tokenizer() {
    std::cout << "[Tokenizer] Testing..." << std::endl;
    
    NumberTokenizer tokenizer;
    
    // Test encode/decode roundtrip
    std::string text = "hello";
    auto ids = tokenizer.encode(text, true, true);
    assert(ids.front() == tokenizer.get_bos_token_id());
    assert(ids.back() == tokenizer.get_eos_token_id());
    
    std::string decoded = tokenizer.decode(ids);
    assert(decoded == text);
    
    // Test empty string
    auto empty_ids = tokenizer.encode("");
    assert(empty_ids.empty());
    
    std::cout << "[Tokenizer] OK" << std::endl;
}

void test_router() {
    std::cout << "[Router] Testing..." << std::endl;
    
    RouterConfig config;
    config.num_experts = 8;
    config.top_k = 2;
    config.hidden_dim = 64;
    
    TopKRouter router(config);
    
    // Create dummy hidden states [batch=2, seq=4, hidden=64]
    std::vector<float> hidden(2 * 4 * 64, 0.1f);
    
    auto output = router.forward(hidden.data(), 2, 4);
    
    assert(output.batch_size == 2);
    assert(output.seq_len == 4);
    assert(output.scores.size() == 2 * 4 * 2);
    assert(output.indices.size() == 2 * 4 * 2);
    
    // Check scores sum to 1
    for (int i = 0; i < 2 * 4; ++i) {
        float sum = output.scores[i * 2] + output.scores[i * 2 + 1];
        assert(std::abs(sum - 1.0f) < 1e-5f);
    }
    
    std::cout << "[Router] OK" << std::endl;
}

void test_load_balancer() {
    std::cout << "[LoadBalancer] Testing..." << std::endl;
    
    LoadBalancer lb(4);
    
    RouterOutput routing;
    routing.indices = {0, 1, 0, 1, 2, 3};
    routing.scores = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    
    lb.record_routing(routing);
    auto result = lb.analyze();
    
    assert(result.max_utilization >= result.min_utilization);
    
    lb.reset();
    auto result_after_reset = lb.analyze();
    assert(result_after_reset.max_utilization == 0.0f);
    
    std::cout << "[LoadBalancer] OK" << std::endl;
}

void test_model() {
    std::cout << "[Model] Testing..." << std::endl;
    
    ModelConfig config;
    config.vocab_size = 260;
    config.hidden_dim = 64;
    config.num_experts = 8;
    config.top_k = 2;
    
    MoEModel model(config);
    
    // Dummy input [batch=1, seq=4]
    std::vector<int> input_ids = {1, 2, 3, 4};
    
    auto [logits, aux_loss] = model.forward(input_ids.data(), 1, 4, false);
    
    assert(logits.size() == 1 * 4 * config.vocab_size);
    assert(aux_loss >= 0.0f);
    
    std::cout << "[Model] OK" << std::endl;
}

void test_engine() {
    std::cout << "[Engine] Testing..." << std::endl;
    
    TokenizerConfig tokenizer_config;
    NumberTokenizer tokenizer(tokenizer_config);
    
    ModelConfig model_config;
    model_config.vocab_size = 260;
    model_config.hidden_dim = 64;
    model_config.num_experts = 8;
    model_config.top_k = 2;
    
    auto model = std::make_unique<MoEModel>(model_config);
    auto lb = std::make_shared<LoadBalancer>(8);
    
    InferenceEngine engine(
        std::move(model),
        std::make_shared<NumberTokenizer>(tokenizer),
        lb
    );
    
    GenerationConfig gen_config;
    gen_config.max_new_tokens = 8;
    gen_config.eos_token_id = tokenizer.get_eos_token_id();
    
    std::vector<int> input = tokenizer.encode("hello", true, false);
    auto output = engine.generate(input, gen_config);
    
    assert(output.size() >= input.size());
    assert(output.size() <= input.size() + gen_config.max_new_tokens);
    
    std::cout << "[Engine] OK" << std::endl;
}

int main() {
    std::cout << "\nMoE-Nexus C++ Tests" << std::endl;
    std::cout << "==================" << std::endl;
    
    try {
        test_tokenizer();
        test_router();
        test_load_balancer();
        test_model();
        test_engine();
        
        std::cout << "\n==================" << std::endl;
        std::cout << "All tests passed!" << std::endl;
        std::cout << "==================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
