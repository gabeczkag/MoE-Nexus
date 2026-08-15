#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include <string>
#include <vector>

#include "moe_nexus/tokenizer.h"
#include "moe_nexus/router.h"
#include "moe_nexus/load_balancer.h"
#include "moe_nexus/model.h"
#include "moe_nexus/engine.h"

namespace py = pybind11;

using namespace moe_nexus;

PYBIND11_MODULE(moe_nexus_core, m) {
    m.doc() = "MoE-Nexus C++ Core Library";
    
    // Tokenizer
    py::class_<TokenizerConfig>(m, "TokenizerConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &TokenizerConfig::vocab_size)
        .def_readwrite("pad_token", &TokenizerConfig::pad_token)
        .def_readwrite("unk_token", &TokenizerConfig::unk_token)
        .def_readwrite("bos_token", &TokenizerConfig::bos_token)
        .def_readwrite("eos_token", &TokenizerConfig::eos_token);
    
    py::class_<NumberTokenizer>(m, "NumberTokenizer")
        .def(py::init<const TokenizerConfig&>(), py::arg("config") = TokenizerConfig{})
        .def("encode", &NumberTokenizer::encode)
        .def("decode", py::overload_cast<const std::vector<int>&>(&NumberTokenizer::decode))
        .def("decode", py::overload_cast<const int*, size_t>(&NumberTokenizer::decode))
        .def("get_vocab_size", &NumberTokenizer::get_vocab_size)
        .def("get_pad_token_id", &NumberTokenizer::get_pad_token_id)
        .def("get_unk_token_id", &NumberTokenizer::get_unk_token_id)
        .def("get_bos_token_id", &NumberTokenizer::get_bos_token_id)
        .def("get_eos_token_id", &NumberTokenizer::get_eos_token_id)
        .def("get_lookup_array", &NumberTokenizer::get_lookup_array);
    
    // Router
    py::class_<RouterConfig>(m, "RouterConfig")
        .def(py::init<>())
        .def_readwrite("num_experts", &RouterConfig::num_experts)
        .def_readwrite("top_k", &RouterConfig::top_k)
        .def_readwrite("noise_std", &RouterConfig::noise_std)
        .def_readwrite("use_aux_loss", &RouterConfig::use_aux_loss)
        .def_readwrite("aux_loss_weight", &RouterConfig::aux_loss_weight)
        .def_readwrite("hidden_dim", &RouterConfig::hidden_dim);
    
    py::class_<RouterOutput>(m, "RouterOutput")
        .def_readwrite("scores", &RouterOutput::scores)
        .def_readwrite("indices", &RouterOutput::indices)
        .def_readwrite("aux_loss", &RouterOutput::aux_loss)
        .def_readwrite("batch_size", &RouterOutput::batch_size)
        .def_readwrite("seq_len", &RouterOutput::seq_len);
    
    py::class_<TopKRouter, std::shared_ptr<TopKRouter>>(m, "TopKRouter")
        .def(py::init<const RouterConfig&>())
        .def("forward", &TopKRouter::forward)
        .def("set_weights", &TopKRouter::set_weights)
        .def("get_weights", &TopKRouter::get_weights)
        .def("get_num_experts", &TopKRouter::get_num_experts)
        .def("get_top_k", &TopKRouter::get_top_k);
    
    // LoadBalancer
    py::class_<ExpertStats>(m, "ExpertStats")
        .def_readwrite("expert_id", &ExpertStats::expert_id)
        .def_readwrite("total_tokens", &ExpertStats::total_tokens)
        .def_readwrite("total_weight", &ExpertStats::total_weight)
        .def_property_readonly("avg_weight", &ExpertStats::avg_weight);
    
    py::class_<BalanceResult>(m, "BalanceResult")
        .def_readwrite("imbalance", &BalanceResult::imbalance)
        .def_readwrite("coefficient_of_variation", &BalanceResult::coefficient_of_variation)
        .def_readwrite("max_utilization", &BalanceResult::max_utilization)
        .def_readwrite("min_utilization", &BalanceResult::min_utilization)
        .def_readwrite("suggestions", &BalanceResult::suggestions);
    
    py::class_<LoadBalancer, std::shared_ptr<LoadBalancer>>(m, "LoadBalancer")
        .def(py::init<int, float>(), py::arg("num_experts"), py::arg("capacity_factor") = 1.25f)
        .def("record_routing", &LoadBalancer::record_routing)
        .def("analyze", &LoadBalancer::analyze)
        .def("reset", &LoadBalancer::reset)
        .def("get_stats", &LoadBalancer::get_stats);
    
    // Model
    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &ModelConfig::vocab_size)
        .def_readwrite("hidden_dim", &ModelConfig::hidden_dim)
        .def_readwrite("num_experts", &ModelConfig::num_experts)
        .def_readwrite("top_k", &ModelConfig::top_k)
        .def_readwrite("num_layers", &ModelConfig::num_layers)
        .def_readwrite("noise_std", &ModelConfig::noise_std)
        .def_readwrite("max_seq_len", &ModelConfig::max_seq_len);
    
    py::class_<MoEModel, std::unique_ptr<MoEModel>>(m, "MoEModel")
        .def(py::init<const ModelConfig&>())
        .def("forward", &MoEModel::forward)
        .def("save_weights", &MoEModel::save_weights)
        .def("load_weights", &MoEModel::load_weights)
        .def("get_config", &MoEModel::get_config);
    
    // Engine
    py::class_<GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def_readwrite("max_new_tokens", &GenerationConfig::max_new_tokens)
        .def_readwrite("temperature", &GenerationConfig::temperature)
        .def_readwrite("top_p", &GenerationConfig::top_p)
        .def_readwrite("top_k", &GenerationConfig::top_k)
        .def_readwrite("repetition_penalty", &GenerationConfig::repetition_penalty)
        .def_readwrite("do_sample", &GenerationConfig::do_sample)
        .def_readwrite("pad_token_id", &GenerationConfig::pad_token_id)
        .def_readwrite("eos_token_id", &GenerationConfig::eos_token_id);
    
    py::class_<BenchmarkResult>(m, "BenchmarkResult")
        .def_readwrite("total_time_s", &BenchmarkResult::total_time_s)
        .def_readwrite("tokens_per_second", &BenchmarkResult::tokens_per_second)
        .def_readwrite("latency_per_token_ms", &BenchmarkResult::latency_per_token_ms)
        .def_readwrite("total_tokens", &BenchmarkResult::total_tokens);
    
    py::class_<InferenceEngine, std::shared_ptr<InferenceEngine>>(m, "InferenceEngine")
        .def(py::init<MoEModelPtr, std::shared_ptr<NumberTokenizer>, std::shared_ptr<LoadBalancer>>())
        .def("generate", &InferenceEngine::generate)
        .def("generate_text", &InferenceEngine::generate_text)
        .def("benchmark", &InferenceEngine::benchmark);
}
