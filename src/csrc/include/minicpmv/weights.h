#pragma once

#include "minicpmv/tensor.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace minicpmv {

struct TensorInfo {
    DType dtype;
    std::vector<int64_t> shape;
    std::string path;
    uint64_t data_begin;
    uint64_t data_end;
};

class WeightsIndex {
public:
    explicit WeightsIndex(const std::string& safetensors_path);

    const TensorInfo& at(const std::string& name) const;
    bool contains(const std::string& name) const;
    size_t size() const { return tensors_.size(); }
    std::vector<std::string> names() const;

    Tensor load_to_device(const std::string& name) const;
    Tensor load_to_device_as(const std::string& name, DType target_dtype) const;

private:
    std::vector<std::string> paths_;
    std::unordered_map<std::string, TensorInfo> tensors_;

    void parse_file(const std::string& path);
};

// Resolve the path to a MiniCPM5 safetensors file or snapshot directory used by tools.
std::string default_safetensors_path();
std::string resolve_safetensors_path(const std::string& path);

}  // namespace minicpmv
