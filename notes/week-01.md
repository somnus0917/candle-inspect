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

### 输入

- input values: [[1, 2, 3], [4, 5, 6]]
- input shape: [2, 3]
- bias values: [10, 20, 30]
- bias shape: [3]

### 输出

- output values: [[11, 22, 33], [14, 25, 36]]
- output shape: [2, 3]
### 为什么 [2, 3] 和 [3] 可以相加
广播从最后一个维度向前比较。[3] 可以看作 [1, 3]，最后一维 3 == 3，前一维 1 可以扩展为 2，所以最终广播为 [2, 3]。

### 为什么 [2, 3] 和 [2] 不可以相加

广播从最后一个维度向前比较。[2] 可以看作 [1, 2]，最后一个维度不是相等的，所以是不可以相加的。
末尾维度比较：3 和 2，它们不相等；3 不是 1；2 也不是 1。因此无法广播

### 本次遇到的 Rust 问题

🈚️
