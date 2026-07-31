use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::new(&[[1, 2, 3], [4, 5, 6]], &device)?;
    let bias = Tensor::new(&[10, 20, 30], &device)?;
    let output = input.broadcast_add(&bias)?;
    println!(
        "input shape: {:?}\nbias shape: {:?}\noutput shape: {:?}",
        input.shape(),
        bias.shape(),
        output.shape()
    );
    println!("output: {}", output);

    let error_bias = Tensor::new(&[1, 2], &device)?;
    if let Err(error) = output.broadcast_add(&error_bias) {
        println!("{:?}", error)
    }
    Ok(())
}
