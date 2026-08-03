use anyhow::Result;
use candle_core::{Device, Tensor};
fn main() -> Result<()> {
    let device = &Device::Cpu;
    let hwc = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        (2, 2, 3),
        device,
    )?;
    println!("hwc shape :{:?}", hwc.shape());
    println!("hwc :{:?}", hwc.to_vec3::<f32>().unwrap());
    let chw = hwc.permute((2, 0, 1))?;
    println!("chw shape :{:?}", chw.shape());
    let nchw = chw.unsqueeze(0)?;
    println!("nchw shape :{:?}", nchw.shape());
    let new_chw = nchw.squeeze(0)?;
    println!("new_chw shape :{:?}", new_chw.shape());
    let new_hcw = new_chw.permute((1, 2, 0))?;
    println!("new_hcw shape :{:?}", new_hcw.shape());
    println!("new_hcw :{:?}", new_hcw.to_vec3::<f32>().unwrap());
    Ok(())
}
