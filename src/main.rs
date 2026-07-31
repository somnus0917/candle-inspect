use std::path::PathBuf;

use anyhow::Result;
use candle_inspect::{
    device::{create_device, DeviceArg},
    inspect, tensor_demo,
};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "candle-inspect")]
#[command(about = "Learn Candle by building a model inspection tool")]
struct Cli {
    #[arg(long, value_enum, default_value_t = DeviceArg::Cpu, global = true)]
    device: DeviceArg,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run a minimal matrix multiplication exercise.
    TensorDemo,

    /// Load a small .safetensors file with Candle and print tensor metadata.
    Inspect {
        path: PathBuf,

        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = create_device(cli.device)?;

    match cli.command {
        Command::TensorDemo => tensor_demo::run(&device)?,
        Command::Inspect { path, limit } => {
            let summary = inspect::inspect_file(&path, &device)?;
            inspect::print_summary(&summary, limit);
        }
    }

    Ok(())
}
