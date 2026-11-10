use alphax_core::error::Result;

use crate::{FactorInputs, FactorSet, FactorValues, eval};

/// Executes a compiled factor set against one input panel.
pub trait ComputeBackend: std::fmt::Debug + Send + Sync {
    /// Stable backend name used by logs and benchmarks.
    fn name(&self) -> &'static str;

    /// Evaluates all factors in output order.
    fn evaluate(&self, set: &FactorSet, inputs: &FactorInputs<'_>) -> Result<FactorValues>;
}

/// The built-in column-major matrix executor.
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeBackend;

impl ComputeBackend for NativeBackend {
    fn name(&self) -> &'static str {
        "native"
    }

    fn evaluate(&self, set: &FactorSet, inputs: &FactorInputs<'_>) -> Result<FactorValues> {
        eval::evaluate(set, inputs)
    }
}
