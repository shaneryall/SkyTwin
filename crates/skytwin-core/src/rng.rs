//! Deterministic random numbers: xoshiro256++ seeded through `SplitMix64`.
//!
//! Source: Blackman & Vigna, "Scrambled linear pseudorandom number
//! generators", ACM TOMS 47(4), 2021.

/// xoshiro256++ generator.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
    /// Second value of the last normal pair.
    spare: Option<f64>,
}

impl Rng {
    /// Seed the generator. Any seed (including 0) is valid: the four state
    /// words are the first four `SplitMix64` outputs from `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            s: std::array::from_fn(|i| {
                splitmix64(seed.wrapping_add(GOLDEN.wrapping_mul(i as u64)))
            }),
            spare: None,
        }
    }

    /// The stream for item `index` of a family keyed by `key`.
    ///
    /// Every item gets its own generator, so what one item draws never
    /// shifts another's numbers (pixels' noise stays put when a source is
    /// added elsewhere), and items can be drawn in any order or in
    /// parallel. The key goes through `SplitMix64` first, so neighbouring
    /// keys (frame seeds `k` and `k + 1`) give unrelated families.
    pub fn for_item(key: u64, index: u64) -> Self {
        Self::new(splitmix64(key).wrapping_add(index.wrapping_mul(0xD1B5_4A32_D192_ED03)))
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Standard normal (Marsaglia's polar method: a pair per accepted
    /// draw, one log and one square root, no trigonometry; the second value
    /// is kept for the next call). Marsaglia & Bray, SIAM Review 6, 260, 1964.
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        loop {
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let k = libm::sqrt(-2.0 * libm::log(s) / s);
                self.spare = Some(v * k);
                return u * k;
            }
        }
    }

    /// Poisson deviate with mean `lambda`.
    ///
    /// Knuth's product method for small means; a rounded normal for
    /// `lambda > 30`, where the error is far below read noise.
    pub fn poisson(&mut self, lambda: f64) -> f64 {
        if lambda <= 0.0 {
            return 0.0;
        }
        if lambda > 30.0 {
            return libm::round(lambda + libm::sqrt(lambda) * self.normal()).max(0.0);
        }
        let limit = libm::exp(-lambda);
        let mut k = 0.0;
        let mut p = self.uniform();
        while p > limit {
            k += 1.0;
            p *= self.uniform();
        }
        k
    }
}

/// `SplitMix64` increment: 2⁶⁴ over the golden ratio.
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// The `SplitMix64` output for state `z`: `z` advanced by [`GOLDEN`], then
/// mixed (Steele, Lea & Flood, OOPSLA 2014). Successive outputs of a stream
/// seeded with `s` are `splitmix64(s + k · GOLDEN)` for `k = 0, 1, …`.
pub(crate) fn splitmix64(z: u64) -> u64 {
    let mut z = z.wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn poisson_mean_and_variance() {
        let mut r = Rng::new(7);
        for &lam in &[0.5, 4.0, 25.0, 400.0] {
            let n = 200_000;
            let xs: Vec<f64> = (0..n).map(|_| r.poisson(lam)).collect();
            let mean = xs.iter().sum::<f64>() / f64::from(n);
            let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / f64::from(n);
            assert!(
                (mean - lam).abs() < 0.02 * lam.max(1.0),
                "lam {lam} mean {mean}"
            );
            assert!(
                (var - lam).abs() < 0.05 * lam.max(1.0),
                "lam {lam} var {var}"
            );
        }
    }
}
