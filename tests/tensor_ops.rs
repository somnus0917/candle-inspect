use candle_core::{Device, Result, Tensor};

#[test]
fn matrix_multiplication_has_expected_shape_and_values() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::new(&[[1f32, 2., 3.]], &device)?;
    let b = Tensor::new(&[[1f32, 0.], [0., 1.], [1., 1.]], &device)?;
    let output = a.matmul(&b)?;

    assert_eq!(output.dims(), &[1, 2]);
    assert_eq!(output.to_vec2::<f32>()?, vec![vec![4.0, 5.0]]);
    Ok(())
}
