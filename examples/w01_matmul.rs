use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::new(&[[1f32, 2., 3.]], &device)?;
    let weight = Tensor::new(&[[1f32, 0.], [0., 1.], [1., 1.]], &device)?;
    let output = input.matmul(&weight)?;

    assert_eq!(output.dims(), &[1, 2]);
    println!("output: {:?}", output.to_vec2::<f32>()?);

    Ok(())
}
