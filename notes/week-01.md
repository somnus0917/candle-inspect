# Week 1 — Tensor basics

## This week's target

- [ ] Create tensors on CPU
- [ ] Understand shape, rank, dtype, and device
- [ ] Practice reshape, transpose, broadcast, and matmul
- [ ] Write one test for each operation learned
- [ ] Record the PyTorch equivalent of every Candle API used

## Daily log template

### Date

**What I ran:**

**Input/output shapes:**

**Rust concept I got stuck on:**

**ML concept I got stuck on:**

**What the error message actually meant:**

**One question worth checking in Candle source code:**

## Broadcast add

### 输入 shape
[[1,2,3],[4,5,6]]
[10,20,30]
### 输出 shape
[[11,22,33],[14,25,36]]

### 为什么 [2, 3] 和 [3] 可以相加
因为 [2, 3] 的维度可以广播到 [3]
### 为什么 [2, 3] 和 [2] 不能直接广播
因为 [2, 3] 的维度不能广播到 [2]

### 本次遇到的 Rust 问题
🈚️
