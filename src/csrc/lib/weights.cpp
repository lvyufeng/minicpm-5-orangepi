#include "minicpmv/weights.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace minicpmv {
namespace {

DType parse_dtype(const std::string& s) {
    if (s == "F16") return DType::Float16;
    if (s == "BF16") return DType::BFloat16;
    if (s == "F32") return DType::Float32;
    if (s == "I32") return DType::Int32;
    if (s == "I64") return DType::Int64;
    if (s == "U8") return DType::UInt8;
    throw std::runtime_error("unsupported safetensors dtype: " + s);
}

uint16_t bf16_bits_to_f16_bits(uint16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
        ++i;
    }
}

void expect(const std::string& s, size_t& i, char ch) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != ch) {
        std::ostringstream oss;
        oss << "expected '" << ch << "' at position " << i;
        throw std::runtime_error(oss.str());
    }
    ++i;
}

std::string parse_string(const std::string& s, size_t& i) {
    skip_ws(s, i);
    expect(s, i, '"');
    std::string out;
    while (i < s.size()) {
        char c = s[i++];
        if (c == '"') {
            return out;
        }
        if (c == '\\') {
            if (i >= s.size()) throw std::runtime_error("bad escape in json string");
            char esc = s[i++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default: throw std::runtime_error("unsupported json escape");
            }
        } else {
            out.push_back(c);
        }
    }
    throw std::runtime_error("unterminated json string");
}

uint64_t parse_uint(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) {
        throw std::runtime_error("expected unsigned integer");
    }
    uint64_t value = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        value = value * 10 + static_cast<uint64_t>(s[i] - '0');
        ++i;
    }
    return value;
}

std::vector<int64_t> parse_int64_array(const std::string& s, size_t& i) {
    std::vector<int64_t> out;
    expect(s, i, '[');
    skip_ws(s, i);
    if (i < s.size() && s[i] == ']') {
        ++i;
        return out;
    }
    while (true) {
        out.push_back(static_cast<int64_t>(parse_uint(s, i)));
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(s, i, ']');
    return out;
}

std::vector<uint64_t> parse_uint64_array(const std::string& s, size_t& i) {
    std::vector<uint64_t> out;
    expect(s, i, '[');
    skip_ws(s, i);
    if (i < s.size() && s[i] == ']') {
        ++i;
        return out;
    }
    while (true) {
        out.push_back(parse_uint(s, i));
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(s, i, ']');
    return out;
}

void skip_json_value(const std::string& s, size_t& i);

void skip_json_object(const std::string& s, size_t& i) {
    expect(s, i, '{');
    skip_ws(s, i);
    if (i < s.size() && s[i] == '}') {
        ++i;
        return;
    }
    while (true) {
        (void)parse_string(s, i);
        expect(s, i, ':');
        skip_json_value(s, i);
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(s, i, '}');
}

void skip_json_array(const std::string& s, size_t& i) {
    expect(s, i, '[');
    skip_ws(s, i);
    if (i < s.size() && s[i] == ']') {
        ++i;
        return;
    }
    while (true) {
        skip_json_value(s, i);
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(s, i, ']');
}

void skip_json_value(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size()) throw std::runtime_error("unexpected end of json");
    char c = s[i];
    if (c == '{') {
        skip_json_object(s, i);
    } else if (c == '[') {
        skip_json_array(s, i);
    } else if (c == '"') {
        (void)parse_string(s, i);
    } else {
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') {
            ++i;
        }
    }
}

struct SafetensorsHeader {
    std::string json;
    uint64_t data_base;
    uint64_t file_size;
};

SafetensorsHeader read_safetensors_header(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open safetensors file: " + path);
    }
    in.seekg(0, std::ios::end);
    auto file_size = static_cast<uint64_t>(in.tellg());
    if (file_size < 8) {
        throw std::runtime_error("invalid safetensors file: " + path);
    }
    in.seekg(0, std::ios::beg);
    uint64_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), sizeof(uint64_t));
    if (!in || 8 + header_len > file_size) {
        throw std::runtime_error("invalid safetensors header length: " + path);
    }
    std::string header(static_cast<size_t>(header_len), '\0');
    if (header_len != 0) {
        in.read(&header[0], static_cast<std::streamsize>(header.size()));
        if (!in) {
            throw std::runtime_error("failed to read safetensors header: " + path);
        }
    }
    return SafetensorsHeader{std::move(header), 8 + header_len, file_size};
}

std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path);
    }
    in.seekg(0, std::ios::end);
    auto size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(size);
    if (size != 0) {
        in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
        if (!in) {
            throw std::runtime_error("failed to read file: " + path);
        }
    }
    return bytes;
}

std::vector<uint8_t> read_file_range(const std::string& path, uint64_t begin, uint64_t end) {
    if (end < begin) {
        throw std::runtime_error("invalid safetensors data offsets in " + path);
    }
    uint64_t bytes_u64 = end - begin;
    if (bytes_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("tensor too large to load from " + path);
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open safetensors file: " + path);
    }
    in.seekg(0, std::ios::end);
    auto file_size = static_cast<uint64_t>(in.tellg());
    if (end > file_size) {
        throw std::runtime_error("safetensors tensor range exceeds file size: " + path);
    }
    in.seekg(static_cast<std::streamoff>(begin), std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(bytes_u64));
    if (!bytes.empty()) {
        in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!in) {
            throw std::runtime_error("failed to read tensor bytes from " + path);
        }
    }
    return bytes;
}

std::vector<std::string> parse_safetensors_weight_map_files(const std::string& index_path) {
    std::vector<uint8_t> bytes = read_file_bytes(index_path);
    std::string json(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    size_t i = 0;
    std::vector<std::string> files;
    std::unordered_set<std::string> seen;

    expect(json, i, '{');
    skip_ws(json, i);
    if (i < json.size() && json[i] == '}') {
        return files;
    }
    while (true) {
        std::string key = parse_string(json, i);
        expect(json, i, ':');
        if (key == "weight_map") {
            expect(json, i, '{');
            skip_ws(json, i);
            if (i < json.size() && json[i] == '}') {
                ++i;
            } else {
                while (true) {
                    (void)parse_string(json, i);
                    expect(json, i, ':');
                    std::string file = parse_string(json, i);
                    if (seen.insert(file).second) {
                        files.push_back(std::move(file));
                    }
                    skip_ws(json, i);
                    if (i < json.size() && json[i] == ',') {
                        ++i;
                        continue;
                    }
                    break;
                }
                expect(json, i, '}');
            }
        } else {
            skip_json_value(json, i);
        }
        skip_ws(json, i);
        if (i < json.size() && json[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(json, i, '}');
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<std::string> resolve_safetensors_paths(const std::string& path) {
    namespace fs = std::filesystem;
    fs::path p(path);
    if (fs::is_regular_file(p)) {
        if (p.filename() == "model.safetensors.index.json") {
            std::vector<std::string> files;
            for (const auto& file : parse_safetensors_weight_map_files(p.string())) {
                fs::path shard = p.parent_path() / file;
                if (!fs::is_regular_file(shard)) {
                    throw std::runtime_error("safetensors index references missing shard: " + shard.string());
                }
                files.push_back(shard.string());
            }
            if (files.empty()) {
                throw std::runtime_error("safetensors index has no weight_map entries: " + p.string());
            }
            return files;
        }
        return {p.string()};
    }
    if (!fs::is_directory(p)) {
        return {path};
    }

    fs::path index_path = p / "model.safetensors.index.json";
    if (fs::is_regular_file(index_path)) {
        std::vector<std::string> files;
        for (const auto& file : parse_safetensors_weight_map_files(index_path.string())) {
            fs::path shard = p / file;
            if (!fs::is_regular_file(shard)) {
                throw std::runtime_error("safetensors index references missing shard: " + shard.string());
            }
            files.push_back(shard.string());
        }
        if (files.empty()) {
            throw std::runtime_error("safetensors index has no weight_map entries: " + index_path.string());
        }
        return files;
    }

    const std::vector<std::string> candidates = {
        "model.safetensors",
        "model-00000-of-00001.safetensors",
    };
    for (const auto& name : candidates) {
        fs::path candidate = p / name;
        if (fs::is_regular_file(candidate)) {
            return {candidate.string()};
        }
    }

    std::vector<fs::path> shards;
    for (const auto& entry : fs::directory_iterator(p)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            shards.push_back(entry.path());
        }
    }
    std::sort(shards.begin(), shards.end());
    if (shards.empty()) {
        throw std::runtime_error("expected a safetensors file or snapshot directory: " + path);
    }
    std::vector<std::string> files;
    files.reserve(shards.size());
    for (const auto& shard : shards) {
        files.push_back(shard.string());
    }
    return files;
}

size_t tensor_numel(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
}

}  // namespace

WeightsIndex::WeightsIndex(const std::string& safetensors_path) : paths_(resolve_safetensors_paths(safetensors_path)) {
    if (paths_.empty()) {
        throw std::runtime_error("no safetensors files resolved from: " + safetensors_path);
    }
    for (const auto& path : paths_) {
        parse_file(path);
    }
}

const TensorInfo& WeightsIndex::at(const std::string& name) const {
    return tensors_.at(name);
}

bool WeightsIndex::contains(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

std::vector<std::string> WeightsIndex::names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& kv : tensors_) {
        out.push_back(kv.first);
    }
    std::sort(out.begin(), out.end());
    return out;
}

Tensor WeightsIndex::load_to_device(const std::string& name) const {
    const auto& info = at(name);
    Tensor tensor(info.shape, info.dtype);
    tensor.allocate();
    std::vector<uint8_t> bytes = read_file_range(info.path, info.data_begin, info.data_end);
    if (bytes.size() != tensor.size_bytes()) {
        throw std::runtime_error("safetensors byte size mismatch for " + name);
    }
    tensor.copy_from_host(bytes.data(), bytes.size());
    return tensor;
}

Tensor WeightsIndex::load_to_device_as(const std::string& name, DType target_dtype) const {
    const auto& info = at(name);
    if (info.dtype == target_dtype) {
        return load_to_device(name);
    }
    if (!(info.dtype == DType::BFloat16 && target_dtype == DType::Float16)) {
        throw std::runtime_error("unsupported dtype conversion in load_to_device_as");
    }

    const size_t numel = tensor_numel(info.shape);
    std::vector<uint8_t> bytes = read_file_range(info.path, info.data_begin, info.data_end);
    if (bytes.size() != numel * dtype_size(info.dtype)) {
        throw std::runtime_error("safetensors byte size mismatch for " + name);
    }

    std::vector<uint16_t> converted(numel);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(bytes.data());
    for (size_t i = 0; i < numel; ++i) {
        converted[i] = bf16_bits_to_f16_bits(src[i]);
    }

    Tensor tensor(info.shape, target_dtype);
    tensor.allocate();
    tensor.copy_from_host(converted.data(), converted.size() * sizeof(uint16_t));
    return tensor;
}

void WeightsIndex::parse_file(const std::string& path) {
    SafetensorsHeader st = read_safetensors_header(path);
    const std::string& header = st.json;
    const uint64_t data_base = st.data_base;

    size_t i = 0;
    expect(header, i, '{');
    skip_ws(header, i);
    if (i < header.size() && header[i] == '}') {
        ++i;
        return;
    }

    while (true) {
        std::string tensor_name = parse_string(header, i);
        expect(header, i, ':');
        if (tensor_name == "__metadata__") {
            skip_json_object(header, i);
        } else {
            expect(header, i, '{');
            TensorInfo info{};
            info.path = path;
            bool have_dtype = false, have_shape = false, have_offsets = false;
            while (true) {
                std::string key = parse_string(header, i);
                expect(header, i, ':');
                if (key == "dtype") {
                    info.dtype = parse_dtype(parse_string(header, i));
                    have_dtype = true;
                } else if (key == "shape") {
                    info.shape = parse_int64_array(header, i);
                    have_shape = true;
                } else if (key == "data_offsets") {
                    auto offsets = parse_uint64_array(header, i);
                    if (offsets.size() != 2) {
                        throw std::runtime_error("data_offsets size must be 2");
                    }
                    info.data_begin = data_base + offsets[0];
                    info.data_end = data_base + offsets[1];
                    have_offsets = true;
                } else {
                    skip_json_value(header, i);
                }
                skip_ws(header, i);
                if (i < header.size() && header[i] == ',') {
                    ++i;
                    continue;
                }
                break;
            }
            expect(header, i, '}');
            if (!have_dtype || !have_shape || !have_offsets) {
                throw std::runtime_error("incomplete tensor entry for " + tensor_name);
            }
            if (info.data_end > st.file_size) {
                throw std::runtime_error("tensor data range exceeds safetensors file for " + tensor_name);
            }
            if (!tensors_.emplace(tensor_name, std::move(info)).second) {
                throw std::runtime_error("duplicate tensor in safetensors shards: " + tensor_name);
            }
        }
        skip_ws(header, i);
        if (i < header.size() && header[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(header, i, '}');
}

std::string resolve_safetensors_path(const std::string& path) {
    std::vector<std::string> paths = resolve_safetensors_paths(path);
    if (paths.size() == 1) {
        return paths.front();
    }
    return path;
}

std::string default_safetensors_path() {
    if (const char* env = std::getenv("MINICPM5_MODEL_PATH")) {
        if (*env) {
            return resolve_safetensors_path(env);
        }
    }
    if (const char* env = std::getenv("MINICPMV_MODEL_PATH")) {
        if (*env) {
            return resolve_safetensors_path(env);
        }
    }
    return resolve_safetensors_path("./models/MiniCPM5-1B");
}

}  // namespace minicpmv
