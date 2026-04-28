//! Error types for CUDA operations.

use cudarc::driver::DriverError;
use cudarc::nvrtc::CompileError;
use thiserror::Error;

/// Errors that can occur during CUDA operations.
#[derive(Debug, Error)]
pub enum CudaError {
    /// CUDA driver error.
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),

    /// CUDA kernel compilation error.
    #[error("CUDA compilation error: {0}")]
    Compile(#[from] CompileError),

    /// No CUDA device available.
    #[error("No CUDA device available")]
    NoDevice,

    /// Dimension mismatch.
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Kernel not found.
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),

    /// Invalid state or CRT invariant violation.
    #[error("{0}")]
    InvalidState(String),
}

/// Result type for CUDA operations.
pub type Result<T> = std::result::Result<T, CudaError>;
