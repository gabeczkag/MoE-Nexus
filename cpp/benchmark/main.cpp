#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include "moe_nexus/tokenizer.h"
#include "moe_nexus/model.h"
#include "moe_nexus/engine.h"
#include "moe_nexus/load_balancer.h"

using namespace moe_nexus;

std::vector<std::string> tokenize_and_prepare(const std::string& text, const NumberTokenizer& tokenizer) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = text.find(' ');
    while (end != std::string::npos) {
        tokens.push_back(text.substr(start, end - start));
        start = end + 1;
        end = text.find(' ', start);
    }
    if (start < text.size()) {
        tokens.push_back(text.substr(start));
    }
    return tokens;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "MoE-Nexus C++ Benchmark" << std::endl;
    std::cout << "==========================================================" << std::endl;
    
    const int vocab_size = 260;
    const int hidden_dim = 64;
    const int num_experts = 8;
    const int top_k = 2;
    const int max_new_tokens = 256;
    const int max_steps = 1000;
    const int batch_size = 4;
    
    TokenizerConfig tokenizer_config;
    NumberTokenizer tokenizer(tokenizer_config);
    
    ModelConfig model_config;
    model_config.vocab_size = vocab_size;
    model_config.hidden_dim = hidden_dim;
    model_config.num_experts = num_experts;
    model_config.top_k = top_k;
    model_config.max_seq_len = 64;
    
    std::vector<std::string> prompts = {"hello", "mixture", "cpu", "token"};
    std::vector<std::vector<int>> input_batch;
    input_batch.reserve(prompts.size());
    
    for (const auto& prompt : prompts) {
        auto ids = tokenizer.encode(prompt, true, false);
        input_batch.push_back(ids);
    }
    
    size_t max_len = 0;
    for (const auto& ids : input_batch) {
        max_len = std::max(max_len, ids.size());
    }
    
    std::vector<int> flat_inputs;
    flat_inputs.reserve(prompts.size() * max_len);
    for (const auto& ids : input_batch) {
        for (size_t i = 0; i < max_len; ++i) {
            flat_inputs.push_back(i < ids.size() ? ids[i] : tokenizer.get_pad_token_id());
        }
    }
    
    GenerationConfig gen_config;
    gen_config.max_new_tokens = max_new_tokens;
    gen_config.temperature = 1.0f;
    gen_config.top_p = 0.9f;
    gen_config.top_k = 50;
    gen_config.do_sample = false;
    gen_config.eos_token_id = tokenizer.get_eos_token_id();
    gen_config.max_steps = max_steps;
    
    std::cout << "\nRunning C++ inference benchmark..." << std::endl;
    std::cout << "Prompts: " << prompts.size() << std::endl;
    std::cout << "Max new tokens per prompt: " << max_new_tokens << std::endl;
    std::cout << "Max steps per prompt: " << max_steps << std::endl;
    std::cout << "Total target tokens: " << prompts.size() * max_new_tokens << std::endl;
    std::cout << std::endl;
    
    MoEModel model(model_config);
    LoadBalancer load_balancer(num_experts);
    InferenceEngine engine(
        std::make_unique<MoEModel>(model),
        std::make_shared<NumberTokenizer>(tokenizer),
        std::make_shared<LoadBalancer>(load_balancer)
    );
    
    auto warmup_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 3; ++i) {
        engine.generate(flat_inputs, gen_config);
    }
    auto warmup_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> warmup_elapsed = warmup_end - warmup_start;
    std::cout << "Warmup: " << warmup_elapsed.count() << "s" << std::endl;
    
    auto bench_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> all_outputs;
    for (int i = 0; i < 10; ++i) {
        all_outputs.push_back(engine.generate(flat_inputs, gen_config));
    }
    auto bench_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> bench_elapsed = bench_end - bench_start;
    
    int total_generated = 0;
    for (const auto& output : all_outputs) {
        total_generated += static_cast<int>(output.size() - flat_inputs.size());
    }
    
    float tps = total_generated / bench_elapsed.count();
    float latency = total_generated > 0 ? (bench_elapsed.count() / total_generated) * 1000.0f : 0.0f;
    
    std::cout << "\n==========================================================" << std::endl;
    std::cout << "RESULTS" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "Total generated tokens: " << total_generated << std::endl;
    std::cout << "Total time: " << bench_elapsed.count() << "s" << std::endl;
    std::cout << "Tokens/second: " << static_cast<int>(tps) << " tok/s" << std::endl;
    std::cout << "Latency/token: " << latency << " ms" << std::endl;
    std::cout << "==========================================================" << std::endl;
    
    if (!all_outputs.empty()) {
        const auto& first_output = all_outputs[0];
        std::string decoded = tokenizer.decode(first_output.data(), first_output.size());
        std::cout << "\nSample output (first prompt):" << std::endl;
        std::cout << decoded.substr(0, 100) << "..." << std::endl;
    }
    
    return 0;
}
