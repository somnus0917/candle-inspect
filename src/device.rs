use anyhow::{bail, Result};
use candle_core::Device;
use clap::ValueEnum;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum DeviceArg {
    Cpu,
    Metal,
    Cuda,
}

pub fn create_device(device: DeviceArg) -> Result<Device> {
    match device {
        DeviceArg::Cpu => Ok(Device::Cpu),
        DeviceArg::Metal => {
            #[cfg(feature = "metal")]
            {
                Ok(Device::new_metal(0)?)
            }
            #[cfg(not(feature = "metal"))]
            {
                bail!("Metal support is disabled. Re-run with --features metal")
            }
        }
        DeviceArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok(Device::new_cuda(0)?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("CUDA support is disabled. Re-run with --features cuda")
            }
        }
    }
}
