use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};
fn main() -> Result<()> {
    let device = &Device::Cpu;
    let weight1 = Tensor::new(
        &[
            [1.0f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, -1.0, 0.0],
        ],
        &device,
    )?;
    let weight2 = Tensor::new(&[[1.0f32, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], &device)?;
    let bias1 = Tensor::new(&[0.0f32, -1.0, 1.0, 0.0], &device)?;
    let bias2 = Tensor::new(&[0.0f32, -1.0], &device)?;

    let linear1 = Linear::new(weight1.clone(), Some(bias1.clone()));
    let linear2 = Linear::new(weight2.clone(), Some(bias2.clone()));

    let input = Tensor::new(&[[1.0f32, 2.0, 3.0], [-1.0, 1.0, 2.0]], &device)?;
    let output1 = linear1.forward(&input)?;
    let output_relu = output1.relu()?;
    println!("after relu is {:?}", output_relu.to_vec2::<f32>());
    let output2 = linear2.forward(&output_relu)?;
    println!("final output is {:?}", output2.to_vec2::<f32>());
    println!(
        "shape:\ninput:{:?},first output :{:?},after relu:{:?},final output:{:?}",
        input.shape(),
        output1.shape(),
        output_relu.shape(),
        output2.shape()
    );

    Ok(())
}
