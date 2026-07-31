use candle_core::{Device, Result, Tensor};

pub fn run(device: &Device) -> Result<()> {
    let a = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], device)?;
    let b = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], device)?;
    let c = a.matmul(&b)?;

    println!("a shape: {:?}", a.dims());
    println!("b shape: {:?}", b.dims());
    println!("c = a @ b, shape: {:?}", c.dims());
    println!("c values: {:?}", c.to_vec2::<f32>()?);

    Ok(())
}
