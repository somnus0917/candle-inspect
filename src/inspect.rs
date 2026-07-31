use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{safetensors, Device};

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub elements: usize,
}

#[derive(Debug, Clone)]
pub struct ModelSummary {
    pub tensors: Vec<TensorInfo>,
    pub total_elements: usize,
}

pub fn inspect_file(path: &Path, device: &Device) -> Result<ModelSummary> {
    let tensors = safetensors::load(path, device)
        .with_context(|| format!("failed to load safetensors file: {}", path.display()))?;

    let mut infos = tensors
        .into_iter()
        .map(|(name, tensor)| TensorInfo {
            name,
            dtype: format!("{:?}", tensor.dtype()),
            shape: tensor.dims().to_vec(),
            elements: tensor.elem_count(),
        })
        .collect::<Vec<_>>();

    infos.sort_by(|left, right| left.name.cmp(&right.name));
    let total_elements = infos.iter().map(|tensor| tensor.elements).sum();

    Ok(ModelSummary {
        tensors: infos,
        total_elements,
    })
}

pub fn print_summary(summary: &ModelSummary, limit: usize) {
    println!("tensor count: {}", summary.tensors.len());
    println!("total elements: {}", summary.total_elements);
    println!();
    println!(
        "{:<64} {:<10} {:<24} {:>14}",
        "name", "dtype", "shape", "elements"
    );
    println!("{}", "-".repeat(116));

    for tensor in summary.tensors.iter().take(limit) {
        println!(
            "{:<64} {:<10} {:<24} {:>14}",
            truncate(&tensor.name, 63),
            tensor.dtype,
            format!("{:?}", tensor.shape),
            tensor.elements
        );
    }

    if summary.tensors.len() > limit {
        println!(
            "\n... {} more tensors (use --limit to show more)",
            summary.tensors.len() - limit
        );
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_owned();
    }

    let mut result = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    result.push('…');
    result
}
