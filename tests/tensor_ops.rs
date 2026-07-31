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

#[test]
fn broadcast_add_produces_expected_output() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::new(&[[1, 2, 3], [4, 5, 6]], &device)?;
    let bias = Tensor::new(&[10, 20, 30], &device)?;
    let output = input.broadcast_add(&bias)?;

    let expected_result = Tensor::new(&[[11, 22, 33], [14, 25, 36]], &device)?;
    assert_eq!(output.to_vec2::<i32>()?, expected_result.to_vec2::<i32>()?);
    Ok(())
}
