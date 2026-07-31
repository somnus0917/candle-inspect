use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, Activation, VarBuilder, VarMap};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let variables = VarMap::new();
    let builder = VarBuilder::from_varmap(&variables, DType::F32, &device);

    let layer1 = linear(4, 8, builder.pp("layer1"))?;
    let layer2 = linear(8, 2, builder.pp("layer2"))?;
    let input = Tensor::randn(0f32, 1f32, (1, 4), &device)?;

    let hidden = layer1.forward(&input)?;
    let hidden = Activation::Relu.forward(&hidden)?;
    let output = layer2.forward(&hidden)?;

    println!("input shape: {:?}", input.dims());
    println!("hidden shape: {:?}", hidden.dims());
    println!("output shape: {:?}", output.dims());

    Ok(())
}
