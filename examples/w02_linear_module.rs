use anyhow::{ensure, Result};
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};
fn main() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::from_vec(
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32],
        (2, 3),
        &device,
    )?;
    let weight = Tensor::from_vec(
        vec![
            1.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 1.0f32, 1.0f32,
            1.0f32,
        ],
        (4, 3),
        &device,
    )?;
    let bias = Tensor::from_vec(vec![10.0f32, 20.0f32, 30.0f32, 40.0f32], 4, &device)?;
    let output = input.matmul(&(weight.t()?))?.broadcast_add(&bias)?;

    println!("{:?}", output);
    println!("{:?}", output.to_vec2::<f32>()?);

    let expected_output = Tensor::from_vec(
        vec![11.0f32, 22.0, 33.0, 46.0, 14.0, 25.0, 36.0, 55.0],
        (2, 4),
        &device,
    )?;
    ensure!(output.shape() == expected_output.shape());
    ensure!(output.to_vec2::<f32>()? == expected_output.to_vec2::<f32>()?);

    let error_input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let error_output = error_input
        .matmul(&(weight.t()?))
        .and_then(|t| t.broadcast_add(&bias));
    match error_output {
        Ok(tensor) => println!("output is :{:?}", tensor),
        Err(e) => println!("error!!!{:?}", e),
    }
    let linear_input = Linear::new(weight, Some(bias));
    let module_output = linear_input.forward(&input)?;
    println!("module output : {:?}", module_output.to_vec2::<f32>()?);
    Ok(())
}
