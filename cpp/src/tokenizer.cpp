#include "moe_nexus/tokenizer.h"
#include <stdexcept>

namespace moe_nexus {

NumberTokenizer::NumberTokenizer(const TokenizerConfig& config) : config_(config) {
    build_vocab();
    build_lookup_array();
}

void NumberTokenizer::build_vocab() {
    char_to_int_.clear();
    int_to_char_.clear();
    
    // Special tokens first
    std::vector<std::pair<std::string, int>> special_tokens = {
        {config_.pad_token, 0},
        {config_.unk_token, 1},
        {config_.bos_token, 2},
        {config_.eos_token, 3}
    };
    
    for (const auto& [token, id] : special_tokens) {
        char_to_int_[token] = id;
        int_to_char_[id] = token;
    }
    
    // ASCII characters
    for (int i = 0; i < 256; ++i) {
        char c = static_cast<char>(i);
        std::string s(1, c);
        if (char_to_int_.find(s) == char_to_int_.end()) {
            int id = static_cast<int>(char_to_int_.size());
            char_to_int_[s] = id;
            int_to_char_[id] = s;
        }
    }
    
    vocab_size_ = static_cast<int>(char_to_int_.size());
    pad_token_id_ = char_to_int_.at(config_.pad_token);
    unk_token_id_ = char_to_int_.at(config_.unk_token);
    bos_token_id_ = char_to_int_.at(config_.bos_token);
    eos_token_id_ = char_to_int_.at(config_.eos_token);
}

void NumberTokenizer::build_lookup_array() {
    size_t max_id = 0;
    for (const auto& [id, str] : int_to_char_) {
        if (static_cast<size_t>(id) > max_id) {
            max_id = static_cast<size_t>(id);
        }
    }
    
    lookup_array_.resize(max_id + 1, 0);
    for (const auto& [id, str] : int_to_char_) {
        if (str.size() == 1) {
            lookup_array_[static_cast<size_t>(id)] = static_cast<uint32_t>(str[0]);
        }
    }
}

std::vector<int> NumberTokenizer::encode(const std::string& text, bool add_bos, bool add_eos) const {
    std::vector<int> tokens;
    tokens.reserve(text.size() + 2);
    
    if (add_bos) {
        tokens.push_back(bos_token_id_);
    }
    
    for (char c : text) {
        std::string s(1, c);
        auto it = char_to_int_.find(s);
        if (it != char_to_int_.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back(unk_token_id_);
        }
    }
    
    if (add_eos) {
        tokens.push_back(eos_token_id_);
    }
    
    return tokens;
}

std::string NumberTokenizer::decode(const std::vector<int>& tokens) const {
    return decode(tokens.data(), tokens.size());
}

std::string NumberTokenizer::decode(const int* tokens, size_t length) const {
    std::string result;
    result.reserve(length);
    
    for (size_t i = 0; i < length; ++i) {
        int id = tokens[i];
        if (id < 0 || static_cast<size_t>(id) >= lookup_array_.size()) {
            continue;
        }
        
        uint32_t cp = lookup_array_[static_cast<size_t>(id)];
        if (cp == 0) continue; // Skip special tokens
        
        result.push_back(static_cast<char>(cp));
    }
    
    return result;
}

} // namespace moe_nexus
