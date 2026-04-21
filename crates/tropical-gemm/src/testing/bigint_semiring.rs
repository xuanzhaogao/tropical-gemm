//! Reference counting semiring with `BigInt` counts — slow but exact.
//!
//! Used as an oracle for CRT-driver correctness tests. Not a public API.

use num_bigint::BigInt;
use num_traits::{One, Zero};
use std::marker::PhantomData;

use crate::types::{TropicalDirection, TropicalScalar};

/// CountingTropical with count field = `BigInt`.
#[derive(Clone, Debug, PartialEq)]
pub struct CountingBigInt<T: TropicalScalar, D: TropicalDirection> {
    pub value: T,
    pub count: BigInt,
    _dir: PhantomData<D>,
}

// BigInt is not Copy; TropicalSemiring requires Copy. We cannot impl
// TropicalSemiring directly. Instead, we plug CountingBigInt into a
// naive reference matmul loop (below), not via the optimized pipeline.

impl<T: TropicalScalar, D: TropicalDirection> CountingBigInt<T, D> {
    pub fn new(value: T, count: BigInt) -> Self {
        Self { value, count, _dir: PhantomData }
    }

    pub fn zero() -> Self {
        Self { value: D::zero_value::<T>(), count: BigInt::zero(), _dir: PhantomData }
    }

    pub fn one() -> Self {
        Self { value: T::scalar_zero(), count: BigInt::one(), _dir: PhantomData }
    }

    pub fn tropical_mul(&self, rhs: &Self) -> Self {
        Self {
            value: self.value.scalar_add(rhs.value),
            count: &self.count * &rhs.count,
            _dir: PhantomData,
        }
    }

    pub fn tropical_add(&self, rhs: &Self) -> Self {
        if D::is_strictly_better(self.value, rhs.value) {
            self.clone()
        } else if D::is_strictly_better(rhs.value, self.value) {
            rhs.clone()
        } else {
            Self {
                value: self.value,
                count: &self.count + &rhs.count,
                _dir: PhantomData,
            }
        }
    }
}

/// Reference row-major matmul for `CountingBigInt<T, D>`. Slow (O(m·n·k)
/// BigInt ops), intended only as an oracle for tests.
pub fn reference_matmul<T: TropicalScalar, D: TropicalDirection>(
    a_values: &[T], m: usize, k: usize,
    b_values: &[T], n: usize,
) -> (Vec<T>, Vec<BigInt>) {
    assert_eq!(a_values.len(), m * k);
    assert_eq!(b_values.len(), k * n);

    let mut out_values = vec![D::zero_value::<T>(); m * n];
    let mut out_counts = vec![BigInt::zero(); m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = CountingBigInt::<T, D>::zero();
            for kk in 0..k {
                let a_ij = CountingBigInt::<T, D>::new(a_values[i * k + kk], BigInt::one());
                let b_ij = CountingBigInt::<T, D>::new(b_values[kk * n + j], BigInt::one());
                let prod = a_ij.tropical_mul(&b_ij);
                acc = acc.tropical_add(&prod);
            }
            out_values[i * n + j] = acc.value;
            out_counts[i * n + j] = acc.count;
        }
    }
    (out_values, out_counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Max;

    #[test]
    fn reference_matmul_1x1() {
        let a = [3.0_f32];
        let b = [4.0_f32];
        let (v, c) = reference_matmul::<f32, Max>(&a, 1, 1, &b, 1);
        assert_eq!(v, vec![7.0]);
        assert_eq!(c, vec![BigInt::from(1)]);
    }

    #[test]
    fn reference_matmul_tie() {
        let a = [2.0_f32, 3.0];
        let b = [3.0_f32, 2.0];
        let (v, c) = reference_matmul::<f32, Max>(&a, 1, 2, &b, 1);
        assert_eq!(v, vec![5.0]);
        assert_eq!(c, vec![BigInt::from(2)]);
    }
}
