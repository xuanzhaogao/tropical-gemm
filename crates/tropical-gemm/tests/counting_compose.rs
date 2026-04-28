use tropical_gemm::{tropical_matmul_t, CountingTropical, Max, Min, TropicalSemiring};

fn ct_max(v: f32, c: u64) -> CountingTropical<f32, u64, Max> {
    CountingTropical::new(v, c)
}

fn ct_min(v: f32, c: u64) -> CountingTropical<f32, u64, Min> {
    CountingTropical::new(v, c)
}

#[test]
fn counting_tropical_max_small_matmul() {
    // A is 2x3, B is 3x2, row-major. All input counts = 1.
    let a = [
        ct_max(1.0, 1), ct_max(2.0, 1), ct_max(3.0, 1),
        ct_max(4.0, 1), ct_max(5.0, 1), ct_max(6.0, 1),
    ];
    let b = [
        ct_max(1.0, 1), ct_max(2.0, 1),
        ct_max(3.0, 1), ct_max(4.0, 1),
        ct_max(5.0, 1), ct_max(6.0, 1),
    ];

    let c = tropical_matmul_t::<CountingTropical<f32, u64, Max>>(&a, 2, 3, &b, 2);

    // C[0,0] = max_k(A[0,k] + B[k,0]) = max(1+1, 2+3, 3+5) = 8, unique k=2, count=1
    assert_eq!(c[0].value, 8.0);
    assert_eq!(c[0].count, 1);
    // C[0,1] = max(1+2, 2+4, 3+6) = 9, count=1
    assert_eq!(c[1].value, 9.0);
    assert_eq!(c[1].count, 1);
    // C[1,0] = max(4+1, 5+3, 6+5) = 11, count=1
    assert_eq!(c[2].value, 11.0);
    assert_eq!(c[2].count, 1);
    // C[1,1] = max(4+2, 5+4, 6+6) = 12, count=1
    assert_eq!(c[3].value, 12.0);
    assert_eq!(c[3].count, 1);
}

#[test]
fn counting_tropical_max_merges_ties() {
    // 1x2 * 2x1. A = [2, 3], B = [3, 2].
    // C[0,0] = max(2+3, 3+2) = max(5, 5) = 5; both k tie, counts merge to 2.
    let a = [ct_max(2.0, 1), ct_max(3.0, 1)];
    let b = [ct_max(3.0, 1), ct_max(2.0, 1)];
    let c = tropical_matmul_t::<CountingTropical<f32, u64, Max>>(&a, 1, 2, &b, 1);
    assert_eq!(c[0].value, 5.0);
    assert_eq!(c[0].count, 2);
}

#[test]
fn counting_tropical_max_multiplies_counts() {
    // Single-k case: input counts multiply.
    // A = [(3.0, 2)], B = [(4.0, 5)] → C = [(7.0, 10)]
    let a = [CountingTropical::<f32, u64, Max>::new(3.0, 2)];
    let b = [CountingTropical::<f32, u64, Max>::new(4.0, 5)];
    let c = tropical_matmul_t::<CountingTropical<f32, u64, Max>>(&a, 1, 1, &b, 1);
    assert_eq!(c[0].value, 7.0);
    assert_eq!(c[0].count, 10);
}

#[test]
fn counting_tropical_min_small_matmul() {
    // Same shape as the Max case; minimize instead.
    let a = [
        ct_min(1.0, 1), ct_min(2.0, 1), ct_min(3.0, 1),
        ct_min(4.0, 1), ct_min(5.0, 1), ct_min(6.0, 1),
    ];
    let b = [
        ct_min(1.0, 1), ct_min(2.0, 1),
        ct_min(3.0, 1), ct_min(4.0, 1),
        ct_min(5.0, 1), ct_min(6.0, 1),
    ];
    let c = tropical_matmul_t::<CountingTropical<f32, u64, Min>>(&a, 2, 3, &b, 2);

    // C[0,0] = min(1+1, 2+3, 3+5) = 2, count=1
    assert_eq!(c[0].value, 2.0);
    assert_eq!(c[0].count, 1);
    // C[0,1] = min(1+2, 2+4, 3+6) = 3
    assert_eq!(c[1].value, 3.0);
    assert_eq!(c[1].count, 1);
    // C[1,0] = min(4+1, 5+3, 6+5) = 5
    assert_eq!(c[2].value, 5.0);
    assert_eq!(c[2].count, 1);
    // C[1,1] = min(4+2, 5+4, 6+6) = 6
    assert_eq!(c[3].value, 6.0);
    assert_eq!(c[3].count, 1);
}

#[test]
fn counting_tropical_min_merges_ties() {
    // Same as the Max tie test: the tie still happens because both products equal 5.
    let a = [ct_min(2.0, 1), ct_min(3.0, 1)];
    let b = [ct_min(3.0, 1), ct_min(2.0, 1)];
    let c = tropical_matmul_t::<CountingTropical<f32, u64, Min>>(&a, 1, 2, &b, 1);
    assert_eq!(c[0].value, 5.0);
    assert_eq!(c[0].count, 2);
}

// TropicalSemiring import used for trait methods in constructors above; silence lint.
#[allow(dead_code)]
fn _use_semiring_trait() {
    let _ = CountingTropical::<f32, u64, Max>::tropical_zero();
}
