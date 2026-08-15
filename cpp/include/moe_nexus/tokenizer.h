#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

namespace moe_nexus {

struct TokenizerConfig {
    size_t vocab_size = 256;
    std::string pad_token = "<pad>";
    std::string unk_token = "<unk>";
    std::string bos_token = "<bos>";
    std::string eos_token = "<eos>";
};

class NumberTokenizer {
public:
    explicit NumberTokenizer(const TokenizerConfig& config = TokenizerConfig{});
    
    std::vector<int> encode(const std::string& text, bool add_bos = false, bool add_eos = false) const;
    std::string decode(const std::vector<int>& tokens) const;
    std::string decode(const int* tokens, size_t length) const;
    
    int get_vocab_size() const { return vocab_size_; }
    int get_pad_token_id() const { return pad_token_id_; }
    int get_unk_token_id() const { return unk_token_id_; }
    int get_bos_token_id() const { return bos_token_id_; }
    int get_eos_token_id() const { return eos_token_id_; }
    
    const std::vector<uint32_t>& get_lookup_array() const { return lookup_array_; }

private:
    void build_vocab();
    void build_lookup_array();
    
    TokenizerConfig config_;
    std::unordered_map<std::string, int> char_to_int_;
    std::unordered_map<int, std::string> int_to_char_;
    std::vector<uint32_t> lookup_array_;
    int vocab_size_ = 0;
    int pad_token_id_ = 0;
    int unk_token_id_ = 0;
    int bos_token_id_ = 0;
    int eos_token_id_ = 0;
};

} // namespace moe_nexus
