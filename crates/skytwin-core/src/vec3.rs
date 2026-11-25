//! Small 3-vector and 3×3 matrix helpers (row-major, `libm` maths).

/// A 3-vector.
pub type V3 = [f64; 3];
/// A 3×3 matrix, row-major.
pub type M3 = [[f64; 3]; 3];

/// Dot product.
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product.
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Euclidean length.
pub fn norm(a: V3) -> f64 {
    libm::sqrt(dot(a, a))
}

/// `a * s`.
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// `a + b`.
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `a - b`.
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Unit vector along `a`.
pub fn unit(a: V3) -> V3 {
    scale(a, 1.0 / norm(a))
}

/// `m · v`.
pub fn mul_mv(m: &M3, v: V3) -> V3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

/// `mᵀ · v`.
pub fn mul_mtv(m: &M3, v: V3) -> V3 {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// `a · b`.
pub fn mul_mm(a: &M3, b: &M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Frame rotation about x by `a` radians (rotates the axes, not the vector).
pub fn rot_x(a: f64) -> M3 {
    let (s, c) = libm::sincos(a);
    [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]]
}

/// Frame rotation about y by `a` radians.
pub fn rot_y(a: f64) -> M3 {
    let (s, c) = libm::sincos(a);
    [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]
}

/// Frame rotation about z by `a` radians.
pub fn rot_z(a: f64) -> M3 {
    let (s, c) = libm::sincos(a);
    [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// Unit vector for right ascension and declination, degrees.
pub fn radec_to_unit(ra_deg: f64, dec_deg: f64) -> V3 {
    let (sa, ca) = libm::sincos(ra_deg.to_radians());
    let (sd, cd) = libm::sincos(dec_deg.to_radians());
    [cd * ca, cd * sa, sd]
}

/// Right ascension in `[0, 360)` and declination, degrees, of a vector.
pub fn unit_to_radec(v: V3) -> (f64, f64) {
    let ra = libm::atan2(v[1], v[0]).to_degrees().rem_euclid(360.0);
    let dec = libm::atan2(v[2], libm::hypot(v[0], v[1])).to_degrees();
    (ra, dec)
}

/// Angle between two vectors, radians (stable for small angles).
pub fn angle(a: V3, b: V3) -> f64 {
    libm::atan2(norm(cross(a, b)), dot(a, b))
}
