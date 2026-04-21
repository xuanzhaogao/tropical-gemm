//! Launch wrapper for the CountingTropical CUDA kernels (spec C).
//!
//! Reuses `GpuMatrix<T>` for values and `GpuMatrix<i32>` for count residues.
//! The kernel expects column-major data; the rest of the crate already uploads
//! column-major, so no extra transposition is needed.

use crate::context::CudaContext;
use crate::error::Result;
use crate::memory::GpuMatrix;
use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits};
use tropical_gemm::types::{Max, Min, TropicalDirection};

pub trait CountingCudaKernel<T, D>
where
    T: DeviceRepr + ValidAsZeroBits + Default + Clone + Copy + 'static,
    D: TropicalDirection,
{
    const KERNEL_NAME: &'static str;

    fn launch_counting_gemm(
        ctx: &CudaContext,
        value_a: &GpuMatrix<T>,
        count_a: &GpuMatrix<i32>,
        value_b: &GpuMatrix<T>,
        count_b: &GpuMatrix<i32>,
        value_c: &mut GpuMatrix<T>,
        count_c: &mut GpuMatrix<i32>,
        modulus: i32,
    ) -> Result<()> {
        let m = value_a.rows();
        let k = value_a.cols();
        let n = value_b.cols();

        assert_eq!(count_a.rows(), m);
        assert_eq!(count_a.cols(), k);
        assert_eq!(value_b.rows(), k);
        assert_eq!(count_b.rows(), k);
        assert_eq!(count_b.cols(), n);
        assert_eq!(value_c.rows(), m);
        assert_eq!(value_c.cols(), n);
        assert_eq!(count_c.rows(), m);
        assert_eq!(count_c.cols(), n);

        let kernel = ctx.get_kernel(Self::KERNEL_NAME)?;
        let cfg = LaunchConfig {
            grid_dim: CudaContext::counting_grid_dims(m, n),
            block_dim: CudaContext::counting_block_dims(),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(
                cfg,
                (
                    value_a.as_slice(),
                    count_a.as_slice(),
                    value_b.as_slice(),
                    count_b.as_slice(),
                    value_c.as_slice_mut(),
                    count_c.as_slice_mut(),
                    m as i32,
                    n as i32,
                    k as i32,
                    modulus,
                ),
            )?;
        }

        ctx.device().synchronize()?;
        Ok(())
    }
}

impl CountingCudaKernel<f32, Max> for (f32, Max) {
    const KERNEL_NAME: &'static str = "counting_gemm_f32_max";
}
impl CountingCudaKernel<f32, Min> for (f32, Min) {
    const KERNEL_NAME: &'static str = "counting_gemm_f32_min";
}
impl CountingCudaKernel<f64, Max> for (f64, Max) {
    const KERNEL_NAME: &'static str = "counting_gemm_f64_max";
}
impl CountingCudaKernel<f64, Min> for (f64, Min) {
    const KERNEL_NAME: &'static str = "counting_gemm_f64_min";
}

pub fn launch_counting_gemm<T, D>(
    ctx: &CudaContext,
    value_a: &GpuMatrix<T>,
    count_a: &GpuMatrix<i32>,
    value_b: &GpuMatrix<T>,
    count_b: &GpuMatrix<i32>,
    value_c: &mut GpuMatrix<T>,
    count_c: &mut GpuMatrix<i32>,
    modulus: i32,
) -> Result<()>
where
    T: DeviceRepr + ValidAsZeroBits + Default + Clone + Copy + 'static,
    D: TropicalDirection,
    (T, D): CountingCudaKernel<T, D>,
{
    <(T, D) as CountingCudaKernel<T, D>>::launch_counting_gemm(
        ctx, value_a, count_a, value_b, count_b, value_c, count_c, modulus,
    )
}
