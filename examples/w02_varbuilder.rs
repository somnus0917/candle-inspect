use anyhow::ensure;
use candle_core::{Context, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};
struct Mlp {
    pub layer1: Linear,
    pub layer2: Linear,
}
impl Mlp {
    fn new(
        inputdims: usize,
        hiddendims: usize,
        outputdims: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let linear1 = linear(inputdims, hiddendims, vb.pp("layer1"))?;
        let linear2 = linear(hiddendims, outputdims, vb.pp("layer2"))?;
        Ok(Self {
            layer1: linear1,
            layer2: linear2,
        })
    }
}
impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.layer1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.layer2.forward(&xs)?;
        Ok(xs)
    }
}
fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, -1.0, 1.0, 2.0], (2, 3), &device)?;
    let m = Mlp::new(3, 4, 2, vb)?;
    let output1 = m.forward(&input)?;
    println!("output shape:{:?}", output1.dims());
    let output2 = m.forward(&input)?;
    println!("output second time:{:?}", output2.dims());

    println!(
        "weight:: layer1 shape:{:?}, layer2 shape:{:?}\nbias:: layer1 shape:{:?}, layer2 shape:{:?}",
        m.layer1.weight().dims(),
        m.layer2.weight().dims(),
        m.layer1.bias().context("error")?.dims(),
        m.layer2.bias().context("error")?.dims(),
    );
    ensure!(output1.to_vec2::<f32>()? == output2.to_vec2::<f32>()?);

    Ok(())
}
