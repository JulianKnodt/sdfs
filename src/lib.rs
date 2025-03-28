#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]

#[cfg(not(feature = "f64"))]
pub type F = f32;

#[cfg(feature = "f64")]
pub type F = f64;

pub mod sdf;
pub mod svd;
pub mod sym;
pub mod to_mesh;

pub mod vec;
pub use vec::*;

#[inline]
pub fn faces_to_neg_idx<const N: usize>(v: &[[usize; N]]) -> impl Iterator<Item = [i32; N]> + '_ {
    assert!(!v.is_empty());
    let max = *v.iter().flatten().max().unwrap();
    faces_to_neg_idx_with_max(v, max)
}

#[inline]
pub fn faces_to_neg_idx_with_max<const N: usize>(
    v: &[[usize; N]],
    max: usize,
) -> impl Iterator<Item = [i32; N]> + '_ {
    assert!(!v.is_empty());
    v.iter()
        .copied()
        .map(move |vis| vis.map(|vi| vi as i32 - max as i32 - 1))
}

#[inline]
pub fn face_iter_to_neg_idx(
    v: impl Iterator<Item = usize> + Clone,
    max: i32,
) -> impl Iterator<Item = i32> {
    v.map(move |vi| vi as i32 - max - 1)
}

#[inline]
pub fn face_iter_to_neg_idx_reversible(
    v: impl Iterator<Item = usize> + DoubleEndedIterator + Clone,
    max: i32,
) -> impl Iterator<Item = i32> + DoubleEndedIterator {
    v.map(move |vi| vi as i32 - max - 1)
}

/// Converts a tri to a quad
#[inline]
pub fn quad_to_tri([a, b, c, d]: [usize; 4]) -> [[usize; 3]; 2] {
    [[a, b, c], [a, c, d]]
}
