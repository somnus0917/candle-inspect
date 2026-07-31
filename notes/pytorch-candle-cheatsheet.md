# PyTorch ↔ Candle notes

| Purpose | PyTorch | Candle |
|---|---|---|
| Create tensor | `torch.tensor(...)` | `Tensor::new(..., &device)?` |
| Zeros | `torch.zeros((2, 3))` | `Tensor::zeros((2, 3), DType::F32, &device)?` |
| Reshape | `x.reshape(2, 3)` | `x.reshape((2, 3))?` |
| Matrix multiply | `a @ b` | `a.matmul(&b)?` |
| Move device | `x.to("cuda")` | `x.to_device(&device)?` |
| Change dtype | `x.to(torch.float16)` | `x.to_dtype(DType::F16)?` |
