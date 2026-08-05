use anyhow::ensure;
use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module};

struct Mlp {
    layer1: Linear,
    layer2: Linear,
}

impl Mlp {
    pub fn custom(device: &Device) -> Result<Self> {
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
        Ok(Self {
            layer1: linear1,
            layer2: linear2,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<Tensor> {
        let xs = self.layer1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.layer2.forward(&xs)?;
        Ok(xs)
    }
}
fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let my_module = Mlp::custom(&device)?;
    let input = Tensor::new(
        &[[1.0f32, 2.0f32, 3.0f32], [-1.0f32, 1.0f32, 2.0f32]],
        &device,
    )?;
    let output = my_module.forward(&input)?;
    println!("output:{:?}", output.to_vec2::<f32>()?);
    ensure!(output.to_vec2::<f32>()? == [[2.0f32, 3.0f32], [0.0f32, 2.0f32]]);
    Ok(())
}
