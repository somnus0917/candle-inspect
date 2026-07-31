use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let tensor = Tensor::arange(0f32, 12f32, &device)?.reshape((3, 4))?;
    let transposed = tensor.transpose(0, 1)?;
    let flattened = transposed.flatten_all()?;

    println!("original {:?}: {tensor}", tensor.dims());
    println!("transposed {:?}: {transposed}", transposed.dims());
    println!("flattened {:?}: {flattened}", flattened.dims());

    Ok(())
}
