use super::direction::{Max, TropicalDirection};
use super::scalar::TropicalScalar;
use super::traits::{SimdTropical, TropicalSemiring, TropicalWithArgmax};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

/// CountingTropical semiring: tracks both the tropical value and the count of optimal paths.
///
/// For TropicalMaxPlus semantics (D = Max, the default):
/// - Multiplication: (n₁, c₁) ⊗ (n₂, c₂) = (n₁ + n₂, c₁ × c₂)
/// - Addition: (n₁, c₁) ⊕ (n₂, c₂) =
///   - if n₁ > n₂: (n₁, c₁)
///   - if n₁ < n₂: (n₂, c₂)
///   - if n₁ = n₂: (n₁, c₁ + c₂)
///
/// For TropicalMinPlus semantics (D = Min):
/// - Same multiplication.
/// - Addition prefers the *smaller* value; ties still merge counts.
///
/// This is used for:
/// - Counting optimal paths in dynamic programming
/// - Computing partition functions
/// - Gradient computations in certain neural network architectures
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct CountingTropical<
    T: TropicalScalar,
    C: TropicalScalar = T,
    D: TropicalDirection = Max,
> {
    /// The tropical value.
    pub value: T,
    /// The count of paths achieving this value.
    pub count: C,
    _dir: PhantomData<D>,
}

impl<T: TropicalScalar, C: TropicalScalar, D: TropicalDirection> CountingTropical<T, C, D> {
    /// Create a new CountingTropical value.
    #[inline(always)]
    pub fn new(value: T, count: C) -> Self {
        Self { value, count, _dir: PhantomData }
    }

    /// Create a CountingTropical from a single value with count 1.
    #[inline(always)]
    pub fn from_value(value: T) -> Self {
        Self {
            value,
            count: C::scalar_one(),
            _dir: PhantomData,
        }
    }
}

impl<T: TropicalScalar, C: TropicalScalar, D: TropicalDirection> TropicalSemiring
    for CountingTropical<T, C, D>
{
    type Scalar = T;

    #[inline(always)]
    fn tropical_zero() -> Self {
        Self {
            value: D::zero_value::<T>(),
            count: C::scalar_zero(),
            _dir: PhantomData,
        }
    }

    #[inline(always)]
    fn tropical_one() -> Self {
        Self {
            value: T::scalar_zero(),
            count: C::scalar_one(),
            _dir: PhantomData,
        }
    }

    #[inline(always)]
    fn tropical_add(self, rhs: Self) -> Self {
        if D::is_strictly_better(self.value, rhs.value) {
            self
        } else if D::is_strictly_better(rhs.value, self.value) {
            rhs
        } else {
            // Equal values: add counts
            Self {
                value: self.value,
                count: self.count.scalar_add(rhs.count),
                _dir: PhantomData,
            }
        }
    }

    #[inline(always)]
    fn tropical_mul(self, rhs: Self) -> Self {
        Self {
            value: self.value.scalar_add(rhs.value),
            count: self.count.scalar_mul(rhs.count),
            _dir: PhantomData,
        }
    }

    #[inline(always)]
    fn value(&self) -> T {
        self.value
    }

    #[inline(always)]
    fn from_scalar(s: T) -> Self {
        Self::from_value(s)
    }
}

impl<T: TropicalScalar, C: TropicalScalar, D: TropicalDirection> TropicalWithArgmax
    for CountingTropical<T, C, D>
{
    type Index = u32;

    #[inline(always)]
    fn tropical_add_argmax(self, self_idx: u32, rhs: Self, rhs_idx: u32) -> (Self, u32) {
        if D::is_strictly_better(self.value, rhs.value) {
            (self, self_idx)
        } else if D::is_strictly_better(rhs.value, self.value) {
            (rhs, rhs_idx)
        } else {
            // Equal values: add counts, keep first index
            (
                Self {
                    value: self.value,
                    count: self.count.scalar_add(rhs.count),
                    _dir: PhantomData,
                },
                self_idx,
            )
        }
    }
}

impl<T: TropicalScalar, C: TropicalScalar, D: TropicalDirection> SimdTropical
    for CountingTropical<T, C, D>
{
    // SIMD for CountingTropical requires SOA layout
    const SIMD_AVAILABLE: bool = true;
    const SIMD_WIDTH: usize = 8;
}

impl<T: TropicalScalar, C: TropicalScalar, D: TropicalDirection> Add
    for CountingTropical<T, C, D>
{
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.tropical_add(rhs)
    }
}

impl<T: TropicalScalar, C: TropicalScalar, D: TropicalDirection> Mul
    for CountingTropical<T, C, D>
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.tropical_mul(rhs)
    }
}

impl<T: TropicalScalar, C: TropicalScalar, D: TropicalDirection> Default
    for CountingTropical<T, C, D>
{
    #[inline(always)]
    fn default() -> Self {
        Self::tropical_zero()
    }
}

impl<T: TropicalScalar, C: TropicalScalar, D: TropicalDirection> fmt::Debug
    for CountingTropical<T, C, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CountingTropical({}, {})", self.value, self.count)
    }
}

impl<T: TropicalScalar, C: TropicalScalar, D: TropicalDirection> fmt::Display
    for CountingTropical<T, C, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.value, self.count)
    }
}

impl<T: TropicalScalar, C: TropicalScalar, D: TropicalDirection> From<T>
    for CountingTropical<T, C, D>
{
    #[inline(always)]
    fn from(value: T) -> Self {
        Self::from_value(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semiring_identity() {
        let a = CountingTropical::<f64>::new(5.0, 2.0);
        let zero = CountingTropical::tropical_zero();
        let one = CountingTropical::tropical_one();

        // a ⊕ 0 = a
        let result = a.tropical_add(zero);
        assert_eq!(result.value, a.value);
        assert_eq!(result.count, a.count);

        // a ⊗ 1 = a
        let result = a.tropical_mul(one);
        assert_eq!(result.value, a.value);
        assert_eq!(result.count, a.count);
    }

    #[test]
    fn test_multiplication() {
        let a = CountingTropical::<f64>::new(3.0, 2.0);
        let b = CountingTropical::<f64>::new(5.0, 3.0);

        let result = a.tropical_mul(b);
        // value = 3 + 5 = 8
        assert_eq!(result.value, 8.0);
        // count = 2 * 3 = 6
        assert_eq!(result.count, 6.0);
    }

    #[test]
    fn test_addition_different_values() {
        let a = CountingTropical::<f64>::new(3.0, 2.0);
        let b = CountingTropical::<f64>::new(5.0, 3.0);

        let result = a.tropical_add(b);
        // max(3, 5) = 5, keep count of winner
        assert_eq!(result.value, 5.0);
        assert_eq!(result.count, 3.0);
    }

    #[test]
    fn test_addition_equal_values() {
        let a = CountingTropical::<f64>::new(5.0, 2.0);
        let b = CountingTropical::<f64>::new(5.0, 3.0);

        let result = a.tropical_add(b);
        // same value, add counts
        assert_eq!(result.value, 5.0);
        assert_eq!(result.count, 5.0);
    }

    #[test]
    fn test_addition_self_wins() {
        let a = CountingTropical::<f64>::new(7.0, 1.0);
        let b = CountingTropical::<f64>::new(5.0, 3.0);

        let result = a.tropical_add(b);
        // max(7, 5) = 7, keep count of winner
        assert_eq!(result.value, 7.0);
        assert_eq!(result.count, 1.0);
    }

    #[test]
    fn test_path_counting_example() {
        // Example: counting paths in a graph
        // Path A->B has value 3, count 1 (one path)
        // Path A->C->B has value 3, count 2 (two equivalent paths)
        // Total paths A->B with optimal value: 1 + 2 = 3

        let path1 = CountingTropical::<f64>::new(3.0, 1.0);
        let path2 = CountingTropical::<f64>::new(3.0, 2.0);

        let result = path1.tropical_add(path2);
        assert_eq!(result.value, 3.0);
        assert_eq!(result.count, 3.0);
    }

    #[test]
    fn test_operator_overloads() {
        let a = CountingTropical::<f64>::new(3.0, 2.0);
        let b = CountingTropical::<f64>::new(5.0, 3.0);

        // Add operator
        let result = a + b;
        assert_eq!(result.value, 5.0);
        assert_eq!(result.count, 3.0);

        // Mul operator
        let result = a * b;
        assert_eq!(result.value, 8.0);
        assert_eq!(result.count, 6.0);
    }

    #[test]
    fn test_default() {
        let d = CountingTropical::<f64>::default();
        assert!(d.value.is_infinite() && d.value < 0.0); // -inf
        assert_eq!(d.count, 0.0);
    }

    #[test]
    fn test_display_debug() {
        let a = CountingTropical::<f64>::new(3.0, 2.0);

        assert_eq!(format!("{}", a), "(3, 2)");
        assert_eq!(format!("{:?}", a), "CountingTropical(3, 2)");
    }

    #[test]
    fn test_from() {
        let a: CountingTropical<f64> = 5.0.into();
        assert_eq!(a.value, 5.0);
        assert_eq!(a.count, 1.0); // Default count is 1

        let b = CountingTropical::<f64>::from(3.0);
        assert_eq!(b.value, 3.0);
        assert_eq!(b.count, 1.0);
    }

    #[test]
    fn test_from_value() {
        let a = CountingTropical::<f64>::from_value(7.0);
        assert_eq!(a.value, 7.0);
        assert_eq!(a.count, 1.0);
    }

    #[test]
    fn test_value_and_from_scalar() {
        let a = CountingTropical::<f64>::new(5.0, 2.0);
        assert_eq!(a.value(), 5.0);

        let b = CountingTropical::<f64>::from_scalar(3.0);
        assert_eq!(b.value(), 3.0);
        assert_eq!(b.count, 1.0);
    }

    #[test]
    fn test_argmax_self_wins() {
        let a = CountingTropical::<f64>::new(7.0, 2.0);
        let b = CountingTropical::<f64>::new(3.0, 1.0);

        let (result, idx) = a.tropical_add_argmax(1, b, 2);
        assert_eq!(result.value, 7.0);
        assert_eq!(result.count, 2.0);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_argmax_rhs_wins() {
        let a = CountingTropical::<f64>::new(3.0, 1.0);
        let b = CountingTropical::<f64>::new(7.0, 2.0);

        let (result, idx) = a.tropical_add_argmax(1, b, 2);
        assert_eq!(result.value, 7.0);
        assert_eq!(result.count, 2.0);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_argmax_equal_counts_added() {
        // Equal values: counts are added, first index is kept
        let a = CountingTropical::<f64>::new(5.0, 2.0);
        let b = CountingTropical::<f64>::new(5.0, 3.0);

        let (result, idx) = a.tropical_add_argmax(1, b, 2);
        assert_eq!(result.value, 5.0);
        assert_eq!(result.count, 5.0); // 2 + 3
        assert_eq!(idx, 1); // First index is kept
    }

    #[test]
    fn test_argmax_chain() {
        let mut acc = CountingTropical::<f64>::tropical_zero();
        let mut idx = 0u32;

        // Values with different counts
        let values = [(3.0, 1.0), (7.0, 2.0), (7.0, 3.0), (5.0, 1.0)];
        for (k, &(val, count)) in values.iter().enumerate() {
            let candidate = CountingTropical::new(val, count);
            (acc, idx) = acc.tropical_add_argmax(idx, candidate, k as u32);
        }

        // Max value is 7.0, first encountered at k=1
        // Counts: 2 + 3 = 5 (both k=1 and k=2 have value 7.0)
        assert_eq!(acc.value, 7.0);
        assert_eq!(acc.count, 5.0);
        assert_eq!(idx, 1); // First index where max occurred
    }

    #[test]
    fn test_simd_tropical() {
        assert!(CountingTropical::<f64>::SIMD_AVAILABLE);
        assert_eq!(CountingTropical::<f64>::SIMD_WIDTH, 8);
    }

    #[test]
    fn test_clone_copy() {
        let a = CountingTropical::<f64>::new(5.0, 2.0);
        let a_copy = a;
        let a_clone = a.clone();

        assert_eq!(a.value, a_copy.value);
        assert_eq!(a.count, a_copy.count);
        assert_eq!(a.value, a_clone.value);
        assert_eq!(a.count, a_clone.count);
    }

    #[test]
    fn test_eq() {
        let a1 = CountingTropical::<f64>::new(5.0, 2.0);
        let a2 = CountingTropical::<f64>::new(5.0, 2.0);
        let b = CountingTropical::<f64>::new(5.0, 3.0);

        assert_eq!(a1, a2);
        assert_ne!(a1, b);
    }

    #[test]
    fn test_f32() {
        let a = CountingTropical::<f32>::new(3.0, 2.0);
        let b = CountingTropical::<f32>::new(5.0, 3.0);

        let result = a.tropical_mul(b);
        assert!((result.value - 8.0).abs() < 1e-6);
        assert!((result.count - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_different_count_type() {
        // Use different types for value and count
        let a = CountingTropical::<f64, f32>::new(3.0, 2.0);
        let b = CountingTropical::<f64, f32>::new(5.0, 3.0);

        let result = a.tropical_mul(b);
        assert_eq!(result.value, 8.0);
        assert!((result.count - 6.0).abs() < 1e-6);
    }

    #[test]
    fn counting_min_prefers_smaller_value() {
        use super::super::direction::Min;
        let a = CountingTropical::<f64, f64, Min>::new(3.0, 2.0);
        let b = CountingTropical::<f64, f64, Min>::new(5.0, 7.0);
        let r = a.tropical_add(b);
        assert_eq!(r.value, 3.0);
        assert_eq!(r.count, 2.0);
    }

    #[test]
    fn counting_min_merges_on_tie() {
        use super::super::direction::Min;
        let a = CountingTropical::<f64, f64, Min>::new(3.0, 2.0);
        let b = CountingTropical::<f64, f64, Min>::new(3.0, 5.0);
        let r = a.tropical_add(b);
        assert_eq!(r.value, 3.0);
        assert_eq!(r.count, 7.0);
    }

    #[test]
    fn counting_min_zero_is_pos_infinity() {
        use super::super::direction::Min;
        let z = CountingTropical::<f64, f64, Min>::tropical_zero();
        assert!(z.value.is_infinite() && z.value > 0.0);
        assert_eq!(z.count, 0.0);
    }
}
