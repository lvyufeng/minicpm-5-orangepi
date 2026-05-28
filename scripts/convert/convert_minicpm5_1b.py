#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from safetensors import safe_open
except ImportError as exc:  # pragma: no cover
    raise SystemExit("install safetensors first: pip install safetensors") from exc

EXPECTED_CONFIG = {
    "model_type": "llama",
    "hidden_size": 1536,
    "intermediate_size": 4608,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "vocab_size": 130560,
}

EXPECTED_TENSORS = {
    "model.embed_tokens.weight": [130560, 1536],
    "model.norm.weight": [1536],
    "lm_head.weight": [130560, 1536],
}

PER_LAYER_TENSORS = {
    "self_attn.q_proj.weight": [2048, 1536],
    "self_attn.k_proj.weight": [256, 1536],
    "self_attn.v_proj.weight": [256, 1536],
    "self_attn.o_proj.weight": [1536, 2048],
    "mlp.gate_proj.weight": [4608, 1536],
    "mlp.up_proj.weight": [4608, 1536],
    "mlp.down_proj.weight": [1536, 4608],
    "input_layernorm.weight": [1536],
    "post_attention_layernorm.weight": [1536],
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        data = load_json(index_path)
        return data["weight_map"]

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors found in {model_dir}")

    weight_map: dict[str, str] = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                weight_map[key] = file.name
    return weight_map


def tensor_shape(model_dir: Path, weight_map: dict[str, str], name: str) -> list[int]:
    file_name = weight_map.get(name)
    if file_name is None:
        raise KeyError(f"missing tensor: {name}")
    with safe_open(model_dir / file_name, framework="pt", device="cpu") as sf:
        return list(sf.get_slice(name).get_shape())


def validate_config(config: dict[str, Any]) -> None:
    errors = []
    for key, expected in EXPECTED_CONFIG.items():
        actual = config.get(key)
        if actual != expected:
            errors.append(f"config.{key}: expected {expected!r}, got {actual!r}")

    q_size = config["num_attention_heads"] * config["head_dim"]
    kv_size = config["num_key_value_heads"] * config["head_dim"]
    if q_size != 2048 or kv_size != 256:
        errors.append(f"projection sizes mismatch: q={q_size}, kv={kv_size}")
    if config["hidden_size"] // config["num_attention_heads"] == config["head_dim"]:
        errors.append("unexpected config: head_dim should be explicit, not hidden_size / num_attention_heads")

    if errors:
        raise ValueError("MiniCPM5-1B config validation failed:\n" + "\n".join(errors))


def validate_tensors(model_dir: Path, weight_map: dict[str, str]) -> dict[str, list[int]]:
    expected = dict(EXPECTED_TENSORS)
    for layer in range(24):
        for suffix, shape in PER_LAYER_TENSORS.items():
            expected[f"model.layers.{layer}.{suffix}"] = shape

    actual_shapes: dict[str, list[int]] = {}
    errors = []
    for name, expected_shape in expected.items():
        try:
            actual_shape = tensor_shape(model_dir, weight_map, name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            continue
        actual_shapes[name] = actual_shape
        if actual_shape != expected_shape:
            errors.append(f"{name}: expected {expected_shape}, got {actual_shape}")

    if errors:
        raise ValueError("MiniCPM5-1B tensor validation failed:\n" + "\n".join(errors))
    return actual_shapes


def write_manifest(output_dir: Path, config: dict[str, Any], shapes: dict[str, list[int]], weight_map: dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": 1,
        "model_id": "openbmb/MiniCPM5-1B",
        "source_dtype": config.get("torch_dtype", "bfloat16"),
        "runtime_dtype": "fp16",
        "num_tensors": len(shapes),
        "weight_files": sorted(set(weight_map.values())),
        "shapes": shapes,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and prepare MiniCPM5-1B safetensors for the OrangePi runtime.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to the Hugging Face MiniCPM5-1B snapshot")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/minicpm5-1b"))
    parser.add_argument("--validate-only", action="store_true", help="Only validate config and tensor shapes")
    args = parser.parse_args()

    config = load_json(args.model_dir / "config.json")
    validate_config(config)
    weight_map = load_index(args.model_dir)
    shapes = validate_tensors(args.model_dir, weight_map)

    if not args.validate_only:
        write_manifest(args.output_dir, config, shapes, weight_map)

    print(f"validated MiniCPM5-1B: {len(shapes)} tensors")
    if not args.validate_only:
        print(f"wrote {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
