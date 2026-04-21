//! Direction marker for CountingTropical: selects Max or Min tropical semantics.

use super::scalar::TropicalScalar;

/// Marker trait selecting tropical direction (max or min).
pub trait TropicalDirection:
    Copy + Clone + Default + std::fmt::Debug + PartialEq + Send + Sync + 'static
{
    /// The tropical zero (additive identity) for this direction in scalar `T`.
    fn zero_value<T: TropicalScalar>() -> T;

    /// True iff `candidate` is strictly better than `incumbent` for this direction.
    fn is_strictly_better<T: TropicalScalar>(candidate: T, incumbent: T) -> bool;
}

/// Maximization direction: zero = -inf, larger is better.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct Max;

/// Minimization direction: zero = +inf, smaller is better.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct Min;

impl TropicalDirection for Max {
    #[inline(always)]
    fn zero_value<T: TropicalScalar>() -> T { T::neg_infinity() }
    #[inline(always)]
    fn is_strictly_better<T: TropicalScalar>(c: T, i: T) -> bool { c > i }
}

impl TropicalDirection for Min {
    #[inline(always)]
    fn zero_value<T: TropicalScalar>() -> T { T::pos_infinity() }
    #[inline(always)]
    fn is_strictly_better<T: TropicalScalar>(c: T, i: T) -> bool { c < i }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_prefers_larger() {
        assert!(Max::is_strictly_better::<f64>(5.0, 3.0));
        assert!(!Max::is_strictly_better::<f64>(3.0, 5.0));
        assert!(!Max::is_strictly_better::<f64>(3.0, 3.0));
    }

    #[test]
    fn min_prefers_smaller() {
        assert!(Min::is_strictly_better::<f64>(3.0, 5.0));
        assert!(!Min::is_strictly_better::<f64>(5.0, 3.0));
        assert!(!Min::is_strictly_better::<f64>(3.0, 3.0));
    }

    #[test]
    fn zero_values() {
        assert!(Max::zero_value::<f64>().is_infinite() && Max::zero_value::<f64>() < 0.0);
        assert!(Min::zero_value::<f64>().is_infinite() && Min::zero_value::<f64>() > 0.0);
    }
}
