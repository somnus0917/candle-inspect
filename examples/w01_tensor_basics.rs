use candle_core::{DType, Device, Result, Tensor};

fn main() -> Result<()> {
    let device = Device::Cpu;

    let zeros = Tensor::zeros((2, 3), DType::F32, &device)?;
    let values = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &device)?;
    let shifted = values.broadcast_add(&Tensor::new(&[10f32, 20., 30.], &device)?)?;

    println!("zeros: {zeros}");
    println!("values: {values}");
    println!("broadcast result: {shifted}");
    println!("shape: {:?}, dtype: {:?}", shifted.dims(), shifted.dtype());

    Ok(())
}
