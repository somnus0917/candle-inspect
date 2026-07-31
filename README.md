# candle-inspect

A learning-first Rust project that gradually becomes a useful model inspection CLI.

## Why this repository exists

This repository has two parallel tracks:

1. `examples/` contains isolated Candle exercises.
2. `src/` contains code that belongs to the real `candle-inspect` tool.

Do not move an exercise into `src/` until it is tested and useful to the actual tool.

## Start here

```bash
cargo run -- tensor-demo
cargo run --example w01_tensor_basics
cargo run --example w01_shape_ops
cargo run --example w01_matmul
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

Inspect a small Safetensors file on CPU:

```bash
cargo run -- inspect ./models/model.safetensors --limit 30
```

Apple Silicon Metal build:

```bash
cargo run --features metal -- --device metal tensor-demo
```

NVIDIA CUDA build:

```bash
cargo run --features cuda -- --device cuda tensor-demo
```

## Important limitation of v0.1

The current `inspect` command loads tensors through Candle, so it is intended for small test files. A later milestone should add metadata-only inspection before loading large model weights.

## Eight-week milestones

- Week 1: tensor creation, shape operations, broadcasting, matmul
- Week 2: `candle-nn`, a small MLP, tests, CPU/Metal or CPU/CUDA comparison
- Week 3: Safetensors metadata and model summary
- Week 4: Hugging Face model download and cache handling
- Week 5: read one Candle model implementation and reproduce its shape flow
- Week 6: run one small real model and compare output with Python
- Week 7: improve errors, docs, tests, and benchmarks
- Week 8: turn one discovered problem into a small upstream Candle contribution

## Commit discipline

Prefer small commits such as:

```text
chore: initialize candle learning project
learn: add tensor reshape and transpose examples
learn: add matrix multiplication test
feat: inspect safetensors tensor metadata
```
