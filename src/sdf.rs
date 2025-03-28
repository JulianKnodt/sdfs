use super::sym::SymMatrix3;
use super::to_mesh::{capsule_to_mesh, cylinder_caps, frustum_to_quad_mesh, ico_sphere};
use super::{
    add, cross, dot, kmul, length, normalize, orthogonal, quat_from_standard, quat_rot, sub, F,
};
use core::ops::Neg;
use std::array::from_fn;

const PI: F = std::f64::consts::PI as F;
const TAU: F = std::f64::consts::TAU as F;

/// Possible output SDFs allowed for minimal volume shapes.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SDFs {
    CappedCylinder(CappedCylinder),
    OrientedBox(OrientedBox),
    Capsule(Capsule),
    IsoscelesTrapezoidalPrism(IsoscelesTrapezoidalPrism),
    Frustum(Frustum),
    Sphere(Sphere),
    // IsoscelesTriangularPrism(IsoscelesTriangularPrism),
    // Unused because Capsule is essentially a sphere
}

impl SDFs {
    pub fn signed_dist(&self, p: [F; 3]) -> F {
        use SDFs::*;
        match self {
            CappedCylinder(cc) => cc.signed_dist(p),
            OrientedBox(ob) => ob.signed_dist(p),
            Capsule(cap) => cap.signed_dist(p),
            IsoscelesTrapezoidalPrism(itp) => itp.signed_dist(p),
            Frustum(f) => f.signed_dist(p),
            Sphere(s) => s.signed_dist(p),
        }
    }
    pub fn scale(&mut self, s: F) {
        use SDFs::*;
        match self {
            CappedCylinder(cc) => cc.scale(s),
            OrientedBox(ob) => ob.scale(s),
            Capsule(cap) => cap.scale(s),
            IsoscelesTrapezoidalPrism(itp) => itp.scale(s),
            Frustum(f) => f.scale(s),
            Sphere(sp) => sp.scale(s),
        }
    }
    pub fn is_oriented_box(&self) -> bool {
        matches!(self, SDFs::OrientedBox(_))
    }
    pub fn is_capsule(&self) -> bool {
        matches!(self, SDFs::Capsule(_))
    }
    pub fn is_capped_cylinder(&self) -> bool {
        matches!(self, SDFs::CappedCylinder(_))
    }
    pub fn is_isosceles_trapezoidal_prism(&self) -> bool {
        matches!(self, SDFs::IsoscelesTrapezoidalPrism(_))
    }
    pub fn is_frustum(&self) -> bool {
        matches!(self, SDFs::Frustum(_))
    }
    pub fn is_sphere(&self) -> bool {
        matches!(self, SDFs::Sphere(_))
    }
    pub fn volume(&self) -> F {
        use SDFs::*;
        match self {
            CappedCylinder(cc) => cc.volume(),
            OrientedBox(ob) => ob.volume(),
            Capsule(cap) => cap.volume(),
            IsoscelesTrapezoidalPrism(itp) => itp.volume(),
            Frustum(f) => f.volume(),
            Sphere(s) => s.volume(),
        }
    }
    pub fn surface_area(&self) -> F {
        use SDFs::*;
        match self {
            CappedCylinder(cc) => cc.surface_area(),
            OrientedBox(ob) => ob.surface_area(),
            Capsule(cap) => cap.surface_area(),
            IsoscelesTrapezoidalPrism(itp) => itp.surface_area(),
            Frustum(f) => f.surface_area(),
            Sphere(s) => s.surface_area(),
        }
    }

    /// Returns a random sample within the SDF, seeded by `i`.
    pub fn sample(&self, i: usize) -> [F; 3] {
        use SDFs::*;
        fn to_unit(f: F) -> F {
            (f + 1.) / 2.
        }
        let params = std::array::from_fn(|j| to_unit((2371.504 + (1337 * j + i) as F).sin()));
        let extra_param = to_unit((i as F * 379.605).sin());
        match self {
            CappedCylinder(cc) => cc.sample(params),
            OrientedBox(b) => b.sample(params),
            Capsule(cap) => cap.sample(extra_param, params),
            IsoscelesTrapezoidalPrism(itp) => itp.sample(params),
            Sphere(s) => s.sample(params),
            Frustum(f) => f.sample(params),
        }
    }
    /// True if this box entirely contains all corners of the other box.
    pub fn contains_oriented_box(&self, o: &OrientedBox) -> bool {
        o.corners()
            .into_iter()
            .all(|c| self.signed_dist(c) <= 1e-10)
    }

    pub fn write_to(&self, mut dst: impl std::io::Write) -> std::io::Result<()> {
        use super::{faces_to_neg_idx, faces_to_neg_idx_with_max};
        use SDFs::*;
        macro_rules! write_v {
            ($c_v: expr) => {
                for [x, y, z] in &$c_v {
                    writeln!(dst, "v {x} {y} {z}")?;
                }
            };
        }
        match self {
            CappedCylinder(cc) => {
                if cc.height == 0. && cc.cylinder.radius == 0. {
                    return Ok(());
                }
                let (c_v, c_f, caps) = cc.to_mesh(16);
                write_v!(c_v);

                writeln!(dst, "usemtl cylinder")?;
                let max = *c_f.iter().flatten().max().unwrap() as i32;
                for [vi0, vi1, vi2, vi3] in faces_to_neg_idx(&c_f) {
                    writeln!(dst, "f {vi0} {vi1} {vi2} {vi3}")?;
                }

                let caps = caps.map(|c| super::face_iter_to_neg_idx(c, max));
                for c in caps {
                    write!(dst, "f ")?;
                    for vi in c {
                        write!(dst, "{vi} ")?;
                    }
                    writeln!(dst)?;
                }
            }
            OrientedBox(ob) => {
                let (c_v, c_f) = ob.to_mesh();
                write_v!(c_v);
                writeln!(dst, "usemtl default")?;
                for [vi0, vi1, vi2, vi3] in faces_to_neg_idx(&c_f) {
                    writeln!(dst, "f {vi0} {vi1} {vi2} {vi3}")?;
                }
            }
            Capsule(cap) => {
                if cap.h == 0. && cap.radius == 0. {
                    return Ok(());
                }
                let (c_v, q_f, t_f) = cap.to_mesh(12);
                write_v!(c_v);
                writeln!(dst, "usemtl capsule")?;
                for [vi0, vi1, vi2, vi3] in faces_to_neg_idx_with_max(&q_f, c_v.len() - 1) {
                    writeln!(dst, "f {vi0} {vi1} {vi2} {vi3}")?;
                }
                for [vi0, vi1, vi2] in faces_to_neg_idx_with_max(&t_f, c_v.len() - 1) {
                    writeln!(dst, "f {vi0} {vi1} {vi2}")?;
                }
            }
            IsoscelesTrapezoidalPrism(itp) => {
                if itp.half_length <= 0.
                    || itp.half_height <= 0.
                    || (itp.half_width_bot <= 0. && itp.half_width_top <= 0.)
                {
                    return Ok(());
                }
                write_v!(itp.corners());
                writeln!(dst, "usemtl trapezoid")?;
                let faces = itp.to_mesh();
                for [vi0, vi1, vi2, vi3] in faces_to_neg_idx(&faces) {
                    writeln!(dst, "f {vi0} {vi1} {vi2} {vi3}")?;
                }
            }
            Frustum(f) => {
                if (f.top_radius == 0. && f.bot_radius == 0.) || f.height() == 0. {
                    return Ok(());
                }
                let (f_v, f_f, caps) = f.to_mesh(16);
                write_v!(f_v);
                writeln!(dst, "usemtl frustum")?;
                let max = *f_f.iter().flatten().max().unwrap() as i32;
                for [vi0, vi1, vi2, vi3] in faces_to_neg_idx(&f_f) {
                    writeln!(dst, "f {vi0} {vi1} {vi2} {vi3}")?;
                }
                let caps = caps.map(|c| super::face_iter_to_neg_idx(c, max));
                for c in caps {
                    write!(dst, "f ")?;
                    for vi in c {
                        write!(dst, "{vi} ")?;
                    }
                    writeln!(dst)?;
                }
            }
            Sphere(sp) => {
                let divs = if sp.radius < 4. { 1 } else { 2 };

                let (mut vs, fs) = ico_sphere(divs);
                for v in &mut vs {
                    *v = kmul(sp.radius, *v);
                    *v = add(*v, sp.center);
                }
                write_v!(vs);
                writeln!(dst, "usemtl sphere")?;
                for [vi0, vi1, vi2] in faces_to_neg_idx(&fs) {
                    writeln!(dst, "f {vi0} {vi1} {vi2}")?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Cylinder {
    pub p: [F; 3],
    pub axis: [F; 3],
    pub radius: F,
}

impl Cylinder {
    pub fn new(p: [F; 3], axis: [F; 3], radius: F) -> Self {
        Self { p, axis, radius }
    }
    pub fn sample_ring(&self, [theta, r]: [F; 2]) -> [F; 3] {
        let tan = orthogonal(self.axis);
        let dir = super::axis_angle_rot(self.axis, tan, theta * TAU);
        let dir = normalize(dir);

        add(self.p, kmul(r.sqrt() * self.radius, dir))
    }
    pub fn scale(&mut self, s: F) {
        self.p = kmul(s, self.p);
        self.radius *= s;
    }
    pub fn to_tuple(self) -> ([F; 3], [F; 3], F) {
        (self.p, self.axis, self.radius)
    }
    pub fn volume(&self, h: F) -> F {
        assert!(h >= 0.);
        PI * self.radius * self.radius * (h * length(self.axis))
    }
    pub fn surface_area(&self, h: F) -> F {
        let two_pi_r = PI * 2. * self.radius;
        // sides
        two_pi_r * h +
       // top and bottom
       two_pi_r * self.radius
    }
    /// Sets the radius of this cylinder to 0.
    pub fn zero(&mut self) {
        self.radius = 0.;
    }
    /// Expand this cylinder to include point p
    pub fn expand_to(&mut self, p: [F; 3]) {
        let d = self.signed_dist(p);
        if d <= 0. {
            return;
        }
        self.radius += d;
    }

    /// Computes the signed distance from a cylinder to a point
    pub fn signed_dist(&self, p: [F; 3]) -> F {
        assert!((length(self.axis) - 1.).abs() < 1e-4, "{self:?}");
        assert!(length(self.axis) < 1.0001, "{self:?}");
        let proj = SymMatrix3::ident() - SymMatrix3::outer(self.axis);
        length(proj.vec_mul(sub(self.p, p))) - self.radius
    }

    /// h is the total height of the capsule.
    pub fn to_capsule(&self, h: F) -> Capsule {
        Capsule {
            p: self.p,
            axis: normalize(self.axis),
            h: (h - 2. * self.radius).max(0.),
            radius: self.radius,
        }
    }

    pub fn to_capped_cylinder(&self, h: F) -> CappedCylinder {
        CappedCylinder::new(*self, h)
    }
    pub fn contains(&self, p: [F; 3]) -> bool {
        let lp = sub(p, self.p);
        let tan = sub(lp, kmul(dot(self.axis, lp), self.axis));
        //println!("{} {}", length(tan), self.radius);
        length(tan) <= self.radius
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CappedCylinder {
    pub cylinder: Cylinder,
    pub height: F,
}

fn sign(x: F) -> F {
    use std::cmp::Ordering::*;
    match x.total_cmp(&0.) {
        Equal => 0.,
        Greater => 1.,
        Less => -1.,
    }
}

impl CappedCylinder {
    pub fn new(cylinder: Cylinder, height: F) -> Self {
        Self { cylinder, height }
    }
    pub fn scale(&mut self, s: F) {
        self.cylinder.scale(s);
        self.height *= s;
    }
    pub fn sample(&self, [hs, theta, r]: [F; 3]) -> [F; 3] {
        let p = self.cylinder.sample_ring([theta, r]);
        add(p, kmul(self.height * hs, self.cylinder.axis))
    }
    pub fn zero(&mut self) {
        self.cylinder.zero();
        self.height = 0.;
    }
    pub fn expand_to(&mut self, p: [F; 3]) {
        self.cylinder.expand_to(p);

        let dist_h = dot(self.cylinder.axis, sub(p, self.cylinder.p));
        if 0. <= dist_h && dist_h <= self.height {
            return;
        }

        let shift_amt = dist_h;
        if dist_h < 0. {
            // have to shift point backwards, since h should only be positive.
            self.cylinder.p = add(self.cylinder.p, kmul(shift_amt, self.cylinder.axis));
            self.height -= shift_amt;
        } else if dist_h > self.height {
            self.height += shift_amt - self.height;
        }
        assert!(self.height >= 0., "{dist_h}");
    }
    pub fn center(&self) -> [F; 3] {
        add(self.cylinder.p, kmul(self.height / 2., self.cylinder.axis))
    }
    /// Returns the quaternion that would transform the y-axis to this cylinder
    pub fn rot_from_y(&self) -> [F; 4] {
        super::quat_from_to([0., 1., 0.], self.cylinder.axis)
    }
    /// Returns the quaternion that would transform the y-axis to this cylinder
    pub fn rot_from_neg_y(&self) -> [F; 4] {
        super::quat_from_to([0., -1., 0.], self.cylinder.axis)
    }
    pub fn volume(&self) -> F {
        self.cylinder.volume(self.height)
    }
    pub fn surface_area(&self) -> F {
        self.cylinder.surface_area(self.height)
    }
    pub fn signed_dist(&self, p: [F; 3]) -> F {
        let a = self.cylinder.p;
        let pa = sub(p, a);
        if self.height.abs() < 1e-10 {
            if self.cylinder.radius == 0. {
                return length(pa);
            }
            // Compute orthogonal distance to cylinder axis
            // and parallel height distance to the point.
            let proj = SymMatrix3::ident() - SymMatrix3::outer(self.cylinder.axis);
            let dist_to_cyl_axis =
                length(proj.vec_mul(sub(self.cylinder.p, p))) - self.cylinder.radius;

            let dist_to_cyl_h = dot(self.cylinder.axis, sub(self.cylinder.p, p));
            return dist_to_cyl_axis.max(0.).hypot(dist_to_cyl_h);
        }
        debug_assert!(length(self.cylinder.axis) > 1e-3);

        let b = add(self.cylinder.p, kmul(self.height, self.cylinder.axis));
        let ba = sub(b, a);
        let baba = dot(ba, ba);
        debug_assert_ne!(baba, 0., "{:?}", self);
        let paba = dot(pa, ba);
        let x = length(sub(kmul(baba, pa), kmul(paba, ba))) - self.cylinder.radius * baba;
        let y = (paba - baba * 0.5).abs() - baba * 0.5;
        let x2 = x * x;
        let y2 = y * y * baba;
        let d = if x.max(y) < 0.0 {
            -x2.min(y2)
        } else {
            let if_pos_then = |a, b| if a > 0. { b } else { 0. };
            if_pos_then(x, x2) + if_pos_then(y, y2)
        };
        sign(d) * d.abs().sqrt() / baba
    }
    pub fn is_degenerate(&self) -> bool {
        length(self.cylinder.axis) < 1e-4 || self.height == 0. || self.cylinder.radius == 0.
    }
    pub fn recenter(&mut self, vs: impl Iterator<Item = [F; 3]>) {
        if length(self.cylinder.axis) == 0. {
            return;
        }

        let mut l = F::INFINITY;
        let mut h = F::NEG_INFINITY;
        assert!((length(self.cylinder.axis) - 1.).abs() < 1e-4,);

        for v in vs {
            let e = dot(self.cylinder.axis, sub(v, self.cylinder.p));
            l = l.min(e);
            h = h.max(e);
        }

        assert!(h >= l);
        self.height = h - l;

        self.cylinder.p = add(self.cylinder.p, kmul(l, self.cylinder.axis));
    }
    pub fn to_mesh(
        &self,
        n: usize,
    ) -> (
        Vec<[F; 3]>,
        Vec<[usize; 4]>,
        [impl Iterator<Item = usize> + Clone; 2],
    ) {
        let (v, f) = frustum_to_quad_mesh(
            n,
            self.cylinder.p,
            self.cylinder.axis,
            self.cylinder.radius,
            self.cylinder.radius,
            0.,
            self.height,
        );
        let nv = v.len();
        (v, f, cylinder_caps(nv))
    }
    /// h is the total height of the capsule.
    pub fn to_capsule(&self) -> Capsule {
        let rad = self.cylinder.radius;
        Capsule {
            p: self.cylinder.p,
            axis: normalize(self.cylinder.axis),
            h: (self.height - 2. * rad).max(0.),
            radius: rad,
        }
    }
    pub fn contains(&self, p: [F; 3]) -> bool {
        let h = self.height;
        let ph = dot(self.cylinder.axis, sub(p, self.cylinder.p));
        (0.0..=(h * h)).contains(&ph) && self.cylinder.contains(p)
    }
}

#[test]
fn test_capped_cylinder_sampling() {
    let cc = CappedCylinder::new(Cylinder::new([3.; 3], [0., 1., 0.], 0.3), 1.5);
    let steps = [0., 0.25, 0.5, 0.75, 1.];
    for i in steps {
        for j in steps {
            for k in steps {
                let p = cc.sample([i, j, k]);
                assert!(cc.contains(p), "{p:?}");
            }
        }
    }
}

#[test]
fn test_capped_cylinder() {
    let mut cc = CappedCylinder::new(Cylinder::new([0.; 3], [0., 1., 0.], 1.), 1.);
    assert_eq!(cc.signed_dist([0.; 3]), 0.);
    assert_eq!(cc.signed_dist([0., 1., 0.]), 0.);
    assert_eq!(cc.signed_dist([0., 2., 0.]), 1.);
    assert_eq!(cc.signed_dist([0., -1., 0.]), 1.);

    cc.zero();
    cc.cylinder.radius = 10.;
    assert_eq!(cc.signed_dist([1.; 3]), 1.);
    assert_eq!(cc.signed_dist([0.; 3]), 0.);

    cc.zero();
    cc.expand_to([0., 1., 0.]);
    assert_eq!(cc.cylinder.radius, 0.);
    assert_eq!(cc.height, 1.);

    cc.zero();
    cc.expand_to([1., 0., 0.]);
    assert_eq!(cc.cylinder.radius, 1.);
    assert_eq!(cc.height, 0.);

    cc.zero();
    cc.expand_to([0., -1., 0.]);
    assert_eq!(cc.cylinder.radius, 0.);
    assert_eq!(cc.height, 1.);
    assert_eq!(cc.cylinder.p, [0., -1., 0.]);

    cc.zero();
    cc.expand_to([0.5; 3]);
    assert_eq!(cc.signed_dist([0.5; 3]), 0.);
}

#[derive(Default, Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Capsule {
    pub(crate) p: [F; 3],
    pub(crate) axis: [F; 3],
    pub(crate) h: F,

    pub radius: F,
}

impl Capsule {
    pub fn new(p: [F; 3], axis: [F; 3], radius: F, h: F) -> Self {
        Self { p, axis, h, radius }
    }
    pub fn scale(&mut self, s: F) {
        self.p = kmul(s, self.p);
        self.h *= s;
        self.radius *= s;
    }
    pub fn sample(&self, choice: F, params: [F; 3]) -> [F; 3] {
        let r2 = self.radius * self.radius;
        let sph_vol = 4. / 3. * PI * r2 * self.radius;
        let cyl_vol = PI * r2 * self.h * length(self.axis);
        let total_vol = sph_vol + cyl_vol;
        let c = choice * total_vol;

        if c > sph_vol {
            // sample from cylinder here
            let cc = CappedCylinder::new(Cylinder::new(self.p, self.axis, self.radius), self.h);
            cc.sample(params)
        } else {
            // sample from sphere here
            let s = Sphere::new(self.p, self.radius);
            let sample = s.sample(params);
            // if it is inside the cylinder, move it to the correct position
            if dot(sub(sample, self.p), self.axis) > 0. {
                add(sample, kmul(self.h, self.axis))
            } else {
                sample
            }
        }
    }
    pub fn from_ends(p: [F; 3], end: [F; 3], radius: F) -> Self {
        let dir = sub(end, p);
        let h = length(dir);
        let axis = normalize(dir);
        Self { p, h, axis, radius }
    }
    pub fn zero(&mut self) {
        self.radius = 0.;
        self.h = 0.;
    }
    pub fn ends(&self) -> [[F; 3]; 2] {
        [self.p, add(self.p, kmul(self.h, self.axis))]
    }
    pub fn is_sphere(&self) -> bool {
        self.h == 0.
    }
    pub fn is_degenerate(&self) -> bool {
        length(self.axis) < 1e-4 || self.radius == 0.
    }
    pub fn volume(&self) -> F {
        assert!(self.h >= 0.);
        assert!(self.radius >= 0.);
        let r2 = self.radius * self.radius;
        let h = self.h * length(self.axis);
        PI * r2 * (4. / 3. * self.radius + h)
    }
    pub fn surface_area(&self) -> F {
        let h = self.h * length(self.axis);
        let two_pi_r = 2. * PI * self.radius;
        // sides
        h * two_pi_r  +
        // top and bottom
        2. * two_pi_r * self.radius
    }
    /// Minimal volume expansion to include point `p`.
    /// First expands it to fit within the cylinder, then expands the height of the cylinder to
    /// correctly fit the point.
    pub fn expand_to(&mut self, p: [F; 3]) {
        let proj = SymMatrix3::ident() - SymMatrix3::outer(self.axis);
        let cyl_rad = length(proj.vec_mul(sub(self.p, p))) - self.radius;
        let internal_rad = if cyl_rad > 0. {
            self.radius += cyl_rad;
            0.
        } else {
            assert!(self.radius >= -cyl_rad);
            (self.radius + cyl_rad).max(0.)
        };
        assert!(internal_rad >= 0., "{internal_rad}");
        assert!(
            internal_rad <= self.radius,
            "{internal_rad} > {}",
            self.radius
        );
        // Given a distance in a sphere, compute height
        let add_h = (self.radius * self.radius - internal_rad * internal_rad)
            .max(0.)
            .sqrt();

        let dist_h = dot(self.axis, sub(p, self.p));
        if -add_h <= dist_h && dist_h <= self.h + add_h {
            return;
        }

        let shift_amt = dist_h + add_h;
        if dist_h < -add_h {
            // have to shift point backwards, since h should only be positive.
            self.p = add(self.p, kmul(shift_amt, self.axis));
            self.h -= shift_amt;
        } else if dist_h > self.h + add_h {
            self.h += shift_amt - self.h - add_h;
        }
        assert!(self.h >= 0., "{dist_h}");
    }
    /// <https://iquilezles.org/articles/distfunctions/>
    #[inline]
    pub fn signed_dist(&self, p: [F; 3]) -> F {
        let pa = sub(p, self.p);
        if self.h == 0. {
            // sphere
            return length(pa) - self.radius;
        }
        let ba = sub(add(self.p, kmul(self.h, self.axis)), self.p);
        let h = dot(pa, ba) / dot(ba, ba);
        let h = h.clamp(0., 1.);

        length(sub(pa, kmul(h, ba))) - self.radius
    }

    /// Computes a loose mesh that approximates this capsule
    pub fn to_mesh(&self, n: usize) -> (Vec<[F; 3]>, Vec<[usize; 4]>, Vec<[usize; 3]>) {
        capsule_to_mesh(n, self.p, self.axis, self.h, 0., self.radius)
    }

    pub fn recenter(&mut self, vs: impl Iterator<Item = [F; 3]> + Clone) {
        self.radius = 0.;
        for v in vs.clone() {
            let delta = sub(v, self.p);
            let e = dot(self.axis, delta);
            let new_r = length(sub(delta, kmul(e, self.axis)));
            self.radius = self.radius.max(new_r);
        }

        let mut l = F::INFINITY;
        let mut h = F::NEG_INFINITY;
        assert!((length(self.axis) - 1.).abs() < 1e-4, "{self:?}");

        for v in vs {
            let delta = sub(v, self.p);
            let e = dot(self.axis, delta);
            let r = length(sub(delta, kmul(e, self.axis))) - self.radius;
            let dh = if r >= 0. {
                0.
            } else {
                (self.radius * self.radius - r * r).max(0.).sqrt()
            };
            l = l.min(e + dh);
            h = h.max(e - dh);
        }

        self.h = (h - l).max(0.);

        if h <= l {
            assert_eq!(self.h, 0.);
            -(h + l) * 0.5
        } else {
            l
        };
        self.p = add(self.p, kmul(l, self.axis));
    }
    pub fn contains(&self, p: [F; 3]) -> bool {
        if self.h == 0. {
            return length(sub(self.p, p)) <= self.radius;
        }
        let [s, e] = self.ends();
        let t = dot(sub(p, s), sub(e, s)) / (self.h * self.h);
        let t = t.clamp(0., 1.);
        let proj = add(kmul(1. - t, s), kmul(t, e));
        length(sub(proj, p)) <= self.radius
    }
}

#[test]
fn test_capsule_sampling() {
    let cap = Capsule::new([3.; 3], [0., 1., 0.], 0.3, 1.5);
    let steps = [0., 0.25, 0.5, 0.75, 0.999];
    for e in steps {
        for i in steps {
            for j in steps {
                for k in steps {
                    let p = cap.sample(e, [i, j, k]);
                    assert!(cap.contains(p), "{p:?}");
                }
            }
        }
    }
}

#[test]
fn test_expand_capsule() {
    let mut capsule = Capsule::new([0., 0., 0.], [0., 1., 0.], 0., 0.);
    capsule.expand_to([1., 0., 0.]);
    assert_eq!(capsule.radius, 1.);
    assert_eq!(capsule.h, 0.);
    capsule.zero();

    capsule.expand_to([0., 1., 0.]);
    assert_eq!(capsule.radius, 0.);
    assert_eq!(capsule.h, 1.);
    capsule.zero();

    capsule.expand_to([0., -1., 0.]);
    assert_eq!(capsule.radius, 0.);
    assert_eq!(capsule.h, 1.);
    assert_eq!(capsule.p, [0., -1., 0.]);
    capsule.zero();

    let p = [0.5, 0.5, 0.5];
    capsule.expand_to(p);
    assert_eq!(capsule.signed_dist(p), 0.);
    assert_eq!(capsule.radius, (0.5 as F).sqrt());
}

/*
#[test]
fn test_capsule_to_mesh() {
    let capsule = Capsule::new([0., 0., 0.], [0., 1., 0.], 2., 1.);
    let (vs, quads, tris) = capsule.to_mesh(8);

    for [x, y, z] in vs {
        println!("v {x} {y} {z}");
    }

    for q in quads {
        let [a, b, c, d] = q.map(|i| i + 1);
        println!("f {a} {b} {c} {d}");
    }

    for t in tris {
        let [a, b, c] = t.map(|i| i + 1);
        println!("f {a} {b} {c}");
    }
    todo!()
}
*/

#[derive(Default, Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Sphere {
    pub center: [F; 3],
    pub radius: F,
}

fn uv_to_elev_azim(uv: [F; 2]) -> [F; 2] {
    assert!((0.0..=1.0).contains(&uv[0]));
    assert!((0.0..=1.0).contains(&uv[1]));

    let [u, v] = uv.map(|v| (v * 2. - 1.).clamp(-1., 1.));
    let elev = v.asin();
    let azim = u.atan2((1.0 - u * u - v * v).max(0.).sqrt());
    [elev, azim]
}
fn elev_azim_to_dir([el, az]: [F; 2]) -> [F; 3] {
    let (els, elc) = el.sin_cos();
    let (azs, azc) = az.sin_cos();

    [azs * elc, azc * elc, els]
}

impl Sphere {
    pub fn new(center: [F; 3], radius: F) -> Self {
        Self { center, radius }
    }
    pub fn to_tuple(self) -> ([F; 3], F) {
        (self.center, self.radius)
    }
    pub fn zero(&mut self) {
        self.radius = 0.;
    }
    pub fn sample(&self, [u, v, rad]: [F; 3]) -> [F; 3] {
        assert!((0.0..=1.0).contains(&rad));
        // convert uv to direction
        let dir = elev_azim_to_dir(uv_to_elev_azim([u, v]));
        let r = rad.sqrt();
        add(self.center, kmul(self.radius * r, dir))
    }
    pub fn inf(&mut self) {
        self.radius = F::INFINITY;
    }
    pub fn scale(&mut self, s: F) {
        self.center = kmul(s, self.center);
        self.radius *= s;
    }
    pub fn shrink_below(&mut self, v: [F; 3]) {
        self.radius = length(sub(v, self.center)).abs().min(self.radius);
    }
    pub fn expand_to(&mut self, v: [F; 3]) {
        self.radius = length(sub(v, self.center)).abs().max(self.radius);
    }
    pub fn signed_dist(&self, v: [F; 3]) -> F {
        length(sub(v, self.center)) - self.radius
    }
    pub fn volume(&self) -> F {
        4. * PI * (self.radius * self.radius * self.radius) / 3.
    }
    pub fn surface_area(&self) -> F {
        4. * PI * self.radius * self.radius
    }
    pub fn contains(&self, p: [F; 3]) -> bool {
        length(sub(self.center, p)) <= self.radius
    }
}

#[test]
fn test_sphere_sampling() {
    let sph = Sphere::new([3.; 3], 1.5);
    let steps = [0., 0.25, 0.5, 0.75, 1.];
    for i in steps {
        for j in steps {
            for k in steps {
                let p = sph.sample([i, j, k]);
                assert!(sph.contains(p));
            }
        }
    }
    let p = Sphere::new([0.; 3], 1.).sample([0., 0., 1.]);
    assert_eq!(length(p), 1.);
}

/*
#[derive(Debug, Clone, Copy, PartialEq)]
struct AxisAlignedBoundingBox {
  pub center: [F; 3],
  pub radii: [F; 3],
}
*/

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OrientedBox {
    pub center: [F; 3],
    pub radii: [F; 3],

    pub(crate) axes: [[F; 3]; 3],

    // redundant but useful to keep around
    rot: [F; 4],
}

impl OrientedBox {
    pub fn new(center: [F; 3], mut axes: [[F; 3]; 3]) -> Self {
        let radii = axes.map(length);
        if dot(cross(axes[0], axes[1]), axes[2]) < 0. {
            axes[2] = axes[2].map(Neg::neg);
        }
        let axes = axes.map(normalize);
        let rot = quat_from_standard(axes[0], axes[1]);
        let rot = super::conj(rot);
        assert!((length(rot) - 1.).abs() < 1e-4, "{} {axes:?}", length(rot));

        Self {
            center,
            radii,
            rot,
            axes,
        }
    }
    pub fn sample(&self, params: [F; 3]) -> [F; 3] {
        let ws = params.map(|v| v * 2. - 1.);
        let [o0, o1, o2] = std::array::from_fn(|d| kmul(self.radii[d] * ws[d], self.axes[d]));
        add(self.center, add(o0, add(o1, o2)))
    }

    pub fn scale(&mut self, s: F) {
        self.center = kmul(s, self.center);
        self.radii = kmul(s, self.radii);
    }

    /// Zeros the radii of this obb.
    pub fn zero(&mut self) {
        self.radii = [0.; 3];
    }

    /// The rotation from the standard basis to this OBB.
    pub fn rot(&self) -> [F; 4] {
        self.rot
    }

    pub fn rot_from_basis(&self, b0: [F; 3], b1: [F; 3]) -> [F; 4] {
        let rot = super::quat_from_basis(self.axes[0], self.axes[1], b0, b1);
        super::conj(rot)
    }

    pub fn contains(&self, p: [F; 3]) -> bool {
        let local_p = sub(p, self.center);
        let e = self.axes.map(|a| dot(a, local_p));
        e.into_iter()
            .enumerate()
            .all(|(i, e)| e <= self.radii[i] + 1e-8)
    }

    pub fn corners(&self) -> [[F; 3]; 8] {
        [
            [false, false, false],
            [false, false, true],
            [false, true, false],
            [false, true, true],
            [true, false, false],
            [true, false, true],
            [true, true, false],
            [true, true, true],
        ]
        .map(|axes_sign| {
            (0..3)
                .map(|i| {
                    let a = kmul(self.radii[i], self.axes[i]);
                    if axes_sign[i] {
                        a.map(Neg::neg)
                    } else {
                        a
                    }
                })
                .fold(self.center, add)
        })
    }
    pub fn inf(&mut self) {
        self.radii = [F::INFINITY; 3];
    }

    /// expands this obb to include v.
    pub fn expand_to(&mut self, v: [F; 3]) {
        if self.signed_dist(v) <= 0. {
            return;
        }
        let rs = self.axes.map(|a| dot(sub(v, self.center), a));
        self.radii = from_fn(|i| rs[i].abs().max(self.radii[i]))
    }
    /// Shrinks this obb to barely not contain v.
    pub fn shrink_below(&mut self, v: [F; 3]) {
        let rs = self.axes.map(|a| dot(sub(v, self.center), a));
        self.radii = from_fn(|i| rs[i].abs().min(self.radii[i]));
    }

    /// Recenters this obb with its original axes inside of a set of points.
    pub fn recenter(&mut self, vs: impl Iterator<Item = [F; 3]>) {
        let mut bds = [[F::INFINITY, F::NEG_INFINITY]; 3];

        for v in vs {
            let es = self.axes.map(|a| dot(v, a));
            for i in 0..3 {
                let [lo, hi] = bds[i];
                bds[i] = [lo.min(es[i]), hi.max(es[i])];
            }
        }

        self.radii = bds.map(|[l, h]| (h - l) / 2.);
        self.center = bds
            .into_iter()
            .enumerate()
            .map(|(i, [l, _])| kmul(self.radii[i] + l, self.axes[i]))
            .fold([0.; 3], add);
    }
    pub fn to_capped_cylinder(&self) -> CappedCylinder {
        let mut axis_ord = [0, 1, 2];
        axis_ord.sort_unstable_by(|&a, &b| F::total_cmp(&self.radii[a], &self.radii[b]));
        let largest_axis = self.axes[axis_ord[2]];
        let height = self.radii[axis_ord[2]];
        // in theory it should be a parameter if it's sqrt(smallest^2 + 2nd smallest^2)
        // or just 2nd smallest.
        let snd_largest_radii = self.radii[axis_ord[1]];

        CappedCylinder::new(
            Cylinder::new(
                sub(self.center, largest_axis),
                normalize(largest_axis),
                snd_largest_radii,
            ),
            height,
        )
    }

    pub fn to_capped_cylinder_on_axis(&self, axis: usize) -> CappedCylinder {
        assert!(axis < 3);
        let height = self.radii[axis];
        let prim_axis = self.axes[axis];
        let rad = match axis {
            0 => [1, 2],
            1 => [0, 2],
            2 => [0, 1],
            _ => unreachable!(),
        }
        .map(|i| self.radii[i])
        .into_iter()
        .fold(0., F::max);

        CappedCylinder::new(
            Cylinder::new(sub(self.center, prim_axis), normalize(prim_axis), rad),
            height,
        )
    }

    pub fn to_capsule(&self) -> Capsule {
        let mut axis_ord = [0, 1, 2];
        axis_ord.sort_unstable_by(|&a, &b| F::total_cmp(&self.radii[a], &self.radii[b]));
        let largest_axis = self.axes[axis_ord[2]];
        let height = self.radii[axis_ord[2]];
        // in theory it should be a parameter if it's sqrt(smallest^2 + 2nd smallest^2)
        // or just 2nd smallest.
        let snd_largest_radii = self.radii[axis_ord[1]];

        Capsule::new(
            sub(self.center, largest_axis),
            normalize(largest_axis),
            snd_largest_radii,
            height * 2.,
        )
    }

    /// Quad mesh from a given bounding box.
    pub fn to_mesh(&self) -> ([[F; 3]; 8], [[usize; 4]; 6]) {
        let c = self.center;
        let p = |px: bool, py: bool, pz: bool| {
            let mut curr = c;
            for (i, p) in [px, py, pz].into_iter().enumerate() {
                let m = if p { self.radii[i] } else { -self.radii[i] };
                curr = add(curr, kmul(m, self.axes[i]));
            }
            curr
        };
        let verts = [
            p(false, false, false),
            p(false, false, true),
            p(false, true, false),
            p(false, true, true),
            p(true, false, false),
            p(true, false, true),
            p(true, true, false),
            p(true, true, true),
        ];

        let faces = [
            [0, 1, 3, 2],
            [6, 7, 5, 4],
            [4, 5, 1, 0],
            [7, 6, 2, 3],
            [3, 1, 5, 7],
            [6, 4, 0, 2],
        ];
        (verts, faces)
    }
    pub fn volume(&self) -> F {
        // note that 8 is included because these are half-extents
        self.radii.iter().copied().map(F::abs).product::<F>() * 8.
    }
    pub fn surface_area(&self) -> F {
        let [l0, l1, l2] = self.radii;
        4. * (l0 * l1 + l1 * l2 + l0 * l2)
    }
    /// Original signed distance implementation which uses a quaternion rotation to transform to
    /// a local space
    pub fn signed_dist2(&self, p: [F; 3]) -> F {
        let local_p = quat_rot(sub(p, self.center), self.rot);
        let local_p = local_p.map(F::abs);
        let q = sub(local_p, self.radii);

        length(q.map(|v| v.max(0.))) + q[0].max(q[1]).max(q[2]).min(0.)
    }
    pub fn signed_dist(&self, p: [F; 3]) -> F {
        let local_p = sub(p, self.center);
        let l_pas = self.axes.map(|a| dot(a, local_p).abs());
        let q = sub(l_pas, self.radii);

        length(q.map(|v| v.max(0.))) + q[0].max(q[1]).max(q[2]).min(0.)
    }
}

#[test]
fn test_obb() {
    let centers = [[0.; 3], [1., 0., 0.]];
    for c in centers {
        let obb = OrientedBox::new(c, [[2., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        assert_eq!(obb.volume(), 16.);
        assert_eq!(obb.signed_dist(add(c, [0., 0., 0.])), -1.);
        assert_eq!(obb.signed_dist(add(c, [1., 0., 0.])), -1.);
        assert_eq!(obb.signed_dist(add(c, [0., 1., 0.])), 0.);
        assert_eq!(obb.signed_dist(add(c, [0., 1., 1.])), 0.);
    }

    let a0 = normalize([0.5, 0.5, 0.]);
    let a1 = normalize([0., 0.5, 0.5]);
    let a1 = normalize(sub(a1, kmul(dot(a0, a1), a0)));
    let axes = [kmul(2., a0), kmul(3., a1), cross(a0, a1)];
    let center = [0., 1., 0.];
    let obb = OrientedBox::new(center, axes);
    for c in obb.corners() {
        assert!(obb.signed_dist(c) < 1e-10, "{obb:?}");
        assert!(obb.contains(c));
    }
    assert_eq!(obb.signed_dist(center), -1.);
    assert!(obb.signed_dist([10.; 3]) > 0.);
}

#[test]
fn test_obb_corners() {
    let obb = OrientedBox::new([0.; 3], [[2., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
    assert_eq!(
        obb.corners(),
        [
            [2.0, 1.0, 1.0],
            [2.0, 1.0, -1.0],
            [2.0, -1.0, 1.0],
            [2.0, -1.0, -1.0],
            [-2.0, 1.0, 1.0],
            [-2.0, 1.0, -1.0],
            [-2.0, -1.0, 1.0],
            [-2.0, -1.0, -1.0]
        ]
    );
}

#[test]
fn test_obb_sampling() {
    let obb = OrientedBox::new([3.; 3], [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]);
    let steps = [0., 0.25, 0.5, 0.75, 1.];
    for i in steps {
        for j in steps {
            for k in steps {
                let p = obb.sample([i, j, k]);
                assert!(obb.contains(p));
            }
        }
    }
    assert_eq!(obb.sample([1., 1., 1.]), [4., 5., 6.]);
}
/*
#[test]
fn test_obb_to_mesh() {
    let obb = OrientedBox::new([0.; 3], [[2., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
    let (v, f) = obb.to_mesh();
    for [x, y, z] in v {
        println!("v {x} {y} {z}");
    }
    for vis in f {
        let [i, j, k, l] = vis.map(|v| v + 1);
        println!("f {i} {j} {k} {l}");
    }
}
*/

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct IsoscelesTriangularPrism {
    pub center: [F; 3],

    pub axis: [F; 3],
    pub half_length: F,

    pub up: [F; 3],
    pub half_height: F,

    // L/R = cross(axis, up)
    pub half_width: F,
}

impl IsoscelesTriangularPrism {
    pub fn empty() -> Self {
        Self::default()
    }
    pub fn scale(&mut self, s: F) {
        self.center = kmul(s, self.center);
        self.half_length *= s;
        self.half_height *= s;
        self.half_width *= s;
    }
    pub fn expand_to(&mut self, op: [F; 3]) {
        let p = sub(op, self.center);

        self.half_length = self.half_length.max(dot(p, self.axis).abs());
        let y = dot(p, self.up);
        // tiny epsilon so that there will always be a little space to expand width later
        self.half_height = self.half_height.max(y.abs() + 1e-10);

        // width is trickier, because it depends on the height.
        assert!((-self.half_height..=self.half_height).contains(&y));

        // [-self.hh, self.hh] -> [-1, 1] -> [0, 2] -> [0,1] -> [0, 1] (flipped)
        let y_ratio = (y / self.half_height) + 1.;
        let y_ratio = 1. - (y_ratio / 2.);
        assert!((0.0..=1.0).contains(&y_ratio));

        let w = dot(self.r(), p).abs();

        if w < 1e-6 {
            return;
        }

        self.half_width = self.half_width.max(w / y_ratio);
        //assert!(self.signed_dist(op) <= 1e-5, "{} {}", self.signed_dist(op), self.volume());
    }
    /// Constructs the set of all triangular prisms which fit inside this OBB.
    #[inline]
    pub fn from_obb(obb: &OrientedBox) -> impl Iterator<Item = Self> + '_ {
        let center = obb.center;
        [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
            [0, 2, 1],
            [1, 0, 2],
            [2, 1, 0],
        ]
        .into_iter()
        .map(move |[a0, a1, a2]| Self {
            center,
            axis: normalize(obb.axes[a0]),
            half_length: obb.radii[a0],

            up: normalize(obb.axes[a1]),
            half_height: obb.radii[a1],

            half_width: obb.radii[a2],
        })
        .flat_map(|t| [t, t.flip_vert()].into_iter())
    }
    fn flip_vert(&self) -> Self {
        let mut s = *self;
        s.up = s.up.map(Neg::neg);
        s
    }
    #[inline]
    fn r(&self) -> [F; 3] {
        cross(self.axis, self.up)
    }
    pub fn signed_dist(&self, p: [F; 3]) -> F {
        let lp = sub(p, self.center);
        let [x, y, z] = [self.r(), self.up, self.axis].map(|a| dot(a, lp));

        // TODO need to see if I'm swapping half length and half width
        let [x, y] = [x.abs() - self.half_width, y + self.half_height];
        let e = [-self.half_width, 2. * self.half_height];
        let lin_dist = dot([x, y], e) / dot(e, e);
        let q = sub([x, y], kmul(lin_dist.clamp(0., 1.), e));

        let mut d1 = length(q);
        if q[0].max(q[1]) < 0. {
            d1 = -d1.min(y);
        }
        let d2 = z.abs() - self.half_length;

        length([d1.max(0.), d2.max(0.)]) + d1.max(d2).min(0.)
    }
    pub fn corners(&self) -> [[F; 3]; 6] {
        let t0_center = sub(self.center, kmul(self.half_length, self.axis));
        let base_point = add(kmul(-self.half_height, self.up), t0_center);
        let r = self.r();
        let [t0a, t0b, t0c] = [
            add(kmul(self.half_height, self.up), t0_center),
            add(base_point, kmul(self.half_width, r)),
            add(base_point, kmul(-self.half_width, r)),
        ];

        let shift = kmul(2. * self.half_length, self.axis);
        [
            t0a,
            t0b,
            t0c,
            add(shift, t0a),
            add(shift, t0b),
            add(shift, t0c),
        ]
    }
    pub fn volume(&self) -> F {
        self.half_height * self.half_width * self.half_length * 4.
    }
    /// Returns the indices of points that describes this triangular prism.
    pub fn to_mesh(&self) -> ([[usize; 4]; 3], [[usize; 3]; 2]) {
        let caps = [[0, 2, 1], [3, 4, 5]];
        let faces = [[0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]];
        (faces, caps)
    }
}

/*
#[test]
fn test_tri_prism() {
    let prism = IsoscelesTriangularPrism {
        center: [0.; 3],
        axis: [1., 0., 0.],
        up: [0., 1., 0.],
        half_height: 10.,
        half_width: 10.,
        half_length: 1.,
    };

    for v in prism.corners() {
        assert_eq!(prism.signed_dist(v), 0.);
    }
    assert_eq!(prism.signed_dist([0.; 3]), -1.);
}
*/

/*
#[test]
fn test_obb_to_tri_to_mesh() {
    let obb = OrientedBox::new([0.; 3], [[2., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
    let (vs, f) = obb.to_mesh();
    for [x, y, z] in vs {
        println!("v {x} {y} {z}");
    }
    for vis in f {
        let [i, j, k, l] = vis.map(|v| v + 1);
        println!("f {i} {j} {k} {l}");
    }

    let prism = IsoscelesTriangularPrism::from_obb(&obb).next().unwrap();
    for [x, y, z] in prism.corners() {
        println!("v {x} {y} {z}");
    }

    let (faces, caps) = prism.to_mesh();
    for vis in faces {
        let [i, j, k, l] = vis.map(|v| v + 1 + vs.len());
        println!("f {i} {j} {k} {l}");
    }
    for vis in caps {
        let [i, j, k] = vis.map(|v| v + 1 + vs.len());
        println!("f {i} {j} {k}");
    }
    todo!();
}
*/

/*
#[test]
fn test_tri_prism_to_mesh() {
    let prism = IsoscelesTriangularPrism {
        center: [0.; 3],
        axis: [1., 0., 0.],
        up: [0., 1., 0.],
        half_height: 10.,
        half_width: 5.,
        half_length: 1.,
    };
    for [x, y, z] in prism.corners() {
        println!("v {x} {y} {z}");
    }

    let (faces, caps) = prism.to_mesh();
    for vis in faces {
        let [i, j, k, l] = vis.map(|v| v + 1);
        println!("f {i} {j} {k} {l}");
    }
    for vis in caps {
        let [i, j, k] = vis.map(|v| v + 1);
        println!("f {i} {j} {k}");
    }
    todo!();
}
*/

#[derive(Default, Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IsoscelesTrapezoidalPrism {
    pub center: [F; 3],

    pub axis: [F; 3],
    pub half_length: F,

    pub up: [F; 3],
    pub half_height: F,

    // L/R = cross(axis, up)
    pub half_width_top: F,
    pub half_width_bot: F,
}

impl IsoscelesTrapezoidalPrism {
    pub fn empty() -> Self {
        Self::default()
    }
    pub fn scale(&mut self, s: F) {
        self.center = kmul(s, self.center);
        self.half_length *= s;
        self.half_height *= s;
        self.half_width_top *= s;
        self.half_width_bot *= s;
    }
    /// h in 0 (bot), 1 (top)
    pub fn half_width_at(&self, t: F) -> F {
        self.half_width_bot * (1. - t) + self.half_width_top * t
    }
    pub fn sample(&self, dirs: [F; 3]) -> [F; 3] {
        let dy_og = dirs[1];
        let [dx, dy, dz] = dirs.map(|v| v * 2. - 1.);
        let h = kmul(self.half_height * dy, self.up);
        let l = kmul(self.half_length * dx, self.axis);

        let right = self.r();
        let r = kmul(dz * self.half_width_at(dy_og), right);

        add(self.center, add(h, add(l, r)))
    }
    pub fn expand_to(&mut self, ops: impl Iterator<Item = [F; 3]> + Clone) {
        // one pass to compute max length and max height
        let r = self.r();
        for op in ops.clone() {
            let p = sub(op, self.center);

            self.half_length = self.half_length.max(dot(p, self.axis).abs());
            self.half_height = self.half_height.max(dot(p, self.up).abs());
        }

        // (w, y_ratio, width at boundaries)
        let mut top_w = (self.half_width_top, 1., self.half_width_top);
        let mut bot_w = (self.half_width_bot, 0., self.half_width_bot);

        assert!(length(r) > 1e-2);
        for op in ops.clone() {
            let p = sub(op, self.center);
            let y = dot(p, self.up);
            debug_assert!((-self.half_height..=self.half_height).contains(&y));

            // [-self.hh, self.hh] -> [-1, 1] -> [0, 2] -> [0,1]
            let y_ratio = y / self.half_height;
            debug_assert!((-1.0..=1.0).contains(&y_ratio));

            let w = dot(r, p).abs();

            let unit_y = (y_ratio + 1.) / 2.;

            use std::cmp::Ordering::*;
            match y_ratio.total_cmp(&0.) {
                Equal | Less => {
                    debug_assert!(unit_y <= 0.5);
                    let next_bot = (w - top_w.2 * unit_y) / (1. - unit_y);
                    let next_bot = next_bot.abs();

                    if next_bot > bot_w.2 {
                        bot_w = (w, unit_y, next_bot);

                        let next_top_w = (top_w.0 - next_bot * (1. - top_w.1)) / top_w.1;
                        top_w.2 = next_top_w.abs();
                    }
                }
                Greater => {
                    debug_assert!(unit_y >= 0.5, "{unit_y}");
                    let next_top = (w - bot_w.2 * (1. - unit_y)) / unit_y;
                    let next_top = next_top.abs();

                    if next_top > top_w.2 {
                        top_w = (w, unit_y, next_top);

                        let next_bot_w = (bot_w.0 - next_top * bot_w.1) / (1. - bot_w.1);
                        bot_w.2 = next_bot_w.abs();
                    }
                }
            };
        }

        debug_assert!(top_w.2 >= 0.);
        debug_assert!(bot_w.2 >= 0.);
        self.half_width_top = top_w.2;
        self.half_width_bot = bot_w.2;

        // fix one side, then optimize the other. This should make it more optimal:
        // (only optimize thinner side)

        macro_rules! fix_side {
            ($top: expr) => {
                let top = $top;
                std::mem::take(if top {
                    &mut self.half_width_top
                } else {
                    &mut self.half_width_bot
                });

                for op in ops.clone() {
                    let p = sub(op, self.center);
                    let y = dot(p, self.up);
                    assert!((-self.half_height..=self.half_height).contains(&y));

                    let w = dot(r, p).abs();

                    // [-self.hh, self.hh] -> [-1, 1] -> [0, 2] -> [0,1]
                    // 0 is at bot, 1 is at top.
                    let unit_y = ((y / self.half_height) + 1.) / 2.;
                    assert!((0.0..=1.0).contains(&unit_y));

                    let next_hw = if top {
                        if unit_y <= 1e-10 {
                            self.half_width_bot = self.half_width_bot.max(w);
                            continue;
                        }
                        (w - self.half_width_bot * (1. - unit_y)) / unit_y
                    } else {
                        if unit_y >= 1. - 1e-10 {
                            self.half_width_top = self.half_width_top.max(w);
                            continue;
                        }
                        (w - self.half_width_top * unit_y) / (1. - unit_y)
                    };
                    assert!(next_hw.is_finite(), "{next_hw} {unit_y}");

                    if top {
                        self.half_width_top = self.half_width_top.max(next_hw);
                    } else {
                        self.half_width_bot = self.half_width_bot.max(next_hw);
                    }
                }
            };
        }

        let top_thinner = self.half_width_top < self.half_width_bot;
        fix_side!(top_thinner);
        fix_side!(!top_thinner);
    }

    /// Constructs the set of all triangular prisms which fit inside this OBB.
    /// Must be expanded.
    #[inline]
    pub fn from_obb(obb: &OrientedBox) -> [IsoscelesTrapezoidalPrism; 6] {
        let center = obb.center;
        [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
            [0, 2, 1],
            [1, 0, 2],
            [2, 1, 0],
        ]
        .map(move |[a0, a1, _a2]| Self {
            center,
            axis: normalize(obb.axes[a0]),
            half_length: obb.radii[a0],

            up: normalize(obb.axes[a1]),
            half_height: obb.radii[a1],

            half_width_top: 0.,
            half_width_bot: 0.,
        })
    }
    #[inline]
    fn r(&self) -> [F; 3] {
        cross(self.axis, self.up)
    }
    pub fn signed_dist(&self, p: [F; 3]) -> F {
        let lp = sub(p, self.center);
        let [x, y, z] = [self.r(), self.up, self.axis].map(|a| dot(a, lp));

        let x = x.abs() - 0.5 * (self.half_width_top + self.half_width_bot);
        let e = [
            0.5 * (self.half_width_top - self.half_width_bot),
            self.half_height,
        ];

        let lin_dist = dot([x, y], e) / dot(e, e);
        let q = sub([x, y], kmul(lin_dist.clamp(-1., 1.), e));

        let mut d1 = length(q);
        if q[0] < 0. {
            d1 = (-d1).max(y.abs() - self.half_height);
        }
        let d2 = z.abs() - self.half_length;

        length([d1.max(0.), d2.max(0.)]) + d1.max(d2).min(0.)
    }
    pub fn corners(&self) -> [[F; 3]; 8] {
        let t0_center = sub(self.center, kmul(self.half_length, self.axis));
        let base_point = add(kmul(-self.half_height, self.up), t0_center);

        let r = self.r();
        let top_point = add(kmul(self.half_height, self.up), t0_center);
        let [t0a, t0b, t0c, t0d] = [
            add(top_point, kmul(self.half_width_top, r)),
            add(top_point, kmul(-self.half_width_top, r)),
            add(base_point, kmul(self.half_width_bot, r)),
            add(base_point, kmul(-self.half_width_bot, r)),
        ];

        let shift = kmul(2. * self.half_length, self.axis);
        [
            t0a,
            t0b,
            t0c,
            t0d,
            add(shift, t0a),
            add(shift, t0b),
            add(shift, t0c),
            add(shift, t0d),
        ]
    }
    pub fn volume(&self) -> F {
        (self.half_width_top + self.half_width_bot) * self.half_height * self.half_length * 4.
    }
    pub fn surface_area(&self) -> F {
        let wt = self.half_width_top * 2.;
        let wb = self.half_width_bot * 2.;
        let h = self.half_height * 2.;
        let l = self.half_length * 2.;

        fn sq(x: F) -> F {
            x * x
        }

        let slant_len = (sq(wt - wb) + sq(h)).sqrt();
        // two trapezoidal faces
        0.5 * (wt + wb) * h +
        // top & bot
        wt * l + wb * l +
        // slants
        2. * slant_len * l
    }
    /// Returns the indices of points that describes this trapezoidal prism.
    pub fn to_mesh(&self) -> [[usize; 4]; 6] {
        [
            [0, 1, 3, 2],
            [6, 7, 5, 4],
            [0, 2, 6, 4],
            [1, 5, 7, 3],
            [0, 4, 5, 1],
            [3, 7, 6, 2],
        ]
    }
    pub fn contains(&self, p: [F; 3]) -> bool {
        let p = sub(p, self.center);
        let h = dot(self.up, p);
        if h.abs() > self.half_height {
            return false;
        }
        let l = dot(self.axis, p);
        if l.abs() > self.half_length {
            return false;
        }

        let half_width = self.half_width_at(h.max(0.).sqrt() / self.half_height);
        dot(self.r(), p).abs() <= half_width
    }
}

#[test]
fn test_iso_trapezoid_prism() {
    let mut prism = IsoscelesTrapezoidalPrism {
        center: [0.; 3],
        axis: [1., 0., 0.],
        up: [0., 1., 0.],
        half_height: 1.,
        half_length: 1.,
        half_width_top: 0.,
        half_width_bot: 0.,
    };
    let p = [1., 0.25, 1.];
    prism.expand_to([p].into_iter());
    assert_eq!(prism.signed_dist(p), 0.);
}

#[test]
fn test_prism_sampling() {
    let itp = IsoscelesTrapezoidalPrism {
        center: [3.; 3],
        axis: [1., 0., 0.],
        up: [0., 1., 0.],
        half_height: 3.,
        half_length: 2.,
        half_width_top: 0.5,
        half_width_bot: 1.5,
    };

    let steps = [0., 0.25, 0.5, 0.75, 1.];
    for i in steps {
        for j in steps {
            for k in steps {
                let p = itp.sample([i, j, k]);
                assert!(itp.contains(p), "{p:?} {i} {j} {j}");
            }
        }
    }
}

/*
#[test]
fn test_trap_prism_to_mesh() {
    let prism = IsoscelesTrapezoidalPrism {
        center: [0.; 3],
        axis: [1., 0., 0.],
        up: [0., 1., 0.],
        half_height: 10.,
        half_length: 1.,
        half_width_top: 5.,
        half_width_bot: 2.,
    };
    for [x, y, z] in prism.corners() {
        println!("v {x} {y} {z}");
    }

    let faces = prism.to_mesh();
    for vis in faces {
        let [i, j, k, l] = vis.map(|v| v + 1);
        println!("f {i} {j} {k} {l}");
    }
    todo!();
}
*/

/// An 18 sided discrete oriented polytope.
/// The axes are half-way vectors between each set of primary axes
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct K18DOP {
    primary_axes: [[F; 3]; 3],
    radii: [[F; 2]; 9],
    center: [F; 3],
    rot: [F; 4],
}

impl K18DOP {
    pub fn new(ob: OrientedBox) -> Self {
        Self {
            primary_axes: ob.axes,
            radii: std::array::from_fn(|i| {
                ob.radii
                    .get(i)
                    .map(|&r| [-r, r])
                    .unwrap_or([F::NEG_INFINITY, F::INFINITY])
            }),
            center: ob.center,
            rot: ob.rot,
        }
    }
    /// Returns all axes for this DOP. Note that each axes corresponds to 2 values,
    /// One positive and one negative
    pub fn axes(&self) -> [[F; 3]; 9] {
        let [a0, a1, a2] = self.primary_axes;
        const SQRT2: F = std::f64::consts::SQRT_2 as F;
        let halfway = |a, b| kmul(SQRT2, add(a, b));
        let flip_on = |v, on| sub(v, kmul(2. * dot(v, on), on));

        let ha01 = halfway(a0, a1);
        let ha12 = halfway(a1, a2);
        let ha20 = halfway(a2, a0);

        let ha01f0 = flip_on(ha01, a0);
        let ha12f1 = flip_on(ha12, a1);
        let ha20f2 = flip_on(ha20, a2);

        [a0, a1, a2, ha01, ha12, ha20, ha01f0, ha12f1, ha20f2]
    }
    pub fn zero(&mut self) {
        self.radii.fill([F::INFINITY, F::NEG_INFINITY]);
    }
    /// Compute the volume of this discrete oriented polytope.
    pub fn volume(&self) -> F {
        todo!()
    }
    pub fn expand_many(&mut self, ps: impl Iterator<Item = [F; 3]>) {
        let axes = self.axes();
        for p in ps {
            let local_p = sub(p, self.center);
            for (ai, &axis) in axes.iter().enumerate() {
                let [lb, ub] = self.radii[ai];
                let d = dot(local_p, axis);
                self.radii[ai] = [lb.min(d), ub.max(d)]
            }
        }
    }
    pub fn internal_signed_dist(&self, p: [F; 3]) -> F {
        let lp = sub(p, self.center);
        let a = self.axes().map(|a| dot(a, lp));
        let q: [F; 9] = std::array::from_fn(|i| {
            let [lb, ub] = self.radii[i];
            a[i].abs() - if a[i] < 0. { lb } else { ub }
        });

        length(q.map(|v| v.max(0.))) + q.into_iter().max_by(F::total_cmp).unwrap_or(0.).min(0.)
    }
    pub fn contains(&self, p: [F; 3]) -> bool {
        let lp = sub(p, self.center);
        self.axes().into_iter().enumerate().all(|(ai, a)| {
            let [lb, ub] = self.radii[ai];
            (lb..=ub).contains(&dot(a, lp))
        })
    }
    pub fn to_mesh(&self) -> (Vec<[F; 3]>, Vec<Vec<usize>>) {
        let c = self.center;
        let axes = self.axes();
        let p = |px: bool, py: bool, pz: bool| {
            let mut curr = c;
            for (i, p) in [px, py, pz].into_iter().enumerate() {
                let m = if p {
                    self.radii[i][1]
                } else {
                    self.radii[i][0]
                };
                curr = add(curr, kmul(m, axes[i]));
            }
            curr
        };
        let verts = [
            p(false, false, false),
            p(false, false, true),
            p(false, true, false),
            p(false, true, true),
            p(true, false, false),
            p(true, false, true),
            p(true, true, false),
            p(true, true, true),
        ]
        .to_vec();
        let faces = [
            [0, 1, 3, 2],
            [6, 7, 5, 4],
            [4, 5, 1, 0],
            [7, 6, 2, 3],
            [3, 1, 5, 7],
            [6, 4, 0, 2],
        ]
        .map(|v| v.to_vec())
        .to_vec();

        // for each vertex, should store which axes it is bound by?
        (verts, faces)
    }
}

// kept for visualization purposes?
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cone {
    pub p: [F; 3],
    pub axis: [F; 3],
    pub radius: F,
}

impl Cone {
    #[inline]
    pub fn height(&self) -> F {
        length(self.axis)
    }
}

/// Part of a cone between two parallel planes.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct Frustum {
    pub p: [F; 3],
    pub axis: [F; 3],
    pub bot_radius: F,
    pub top_radius: F,
}

impl Frustum {
    #[inline]
    pub fn height(&self) -> F {
        length(self.axis)
    }
    pub fn scale(&mut self, s: F) {
        self.p = kmul(s, self.p);

        self.bot_radius *= s;
        self.top_radius *= s;
        self.axis = kmul(s, self.axis);
    }
    fn radius_at(&self, t: F) -> F {
        self.bot_radius * (1. - t) + self.top_radius * t
    }
    pub fn sample(&self, [hs, theta, r]: [F; 3]) -> [F; 3] {
        let ad = normalize(self.axis);
        let tan = orthogonal(ad);
        let dir = super::axis_angle_rot(ad, tan, theta * TAU);
        let dir = normalize(dir);
        let p = add(self.p, kmul(hs, self.axis));
        let rad = r.sqrt() * self.radius_at(hs);
        add(p, kmul(rad, dir))
    }

    #[inline]
    pub fn from_capped_cylinder(cc: &CappedCylinder) -> Self {
        Self {
            p: cc.cylinder.p,
            axis: kmul(cc.height, normalize(cc.cylinder.axis)),
            // initialize these to zero so that they can be increased in size later.
            bot_radius: cc.cylinder.radius,
            top_radius: cc.cylinder.radius,
        }
    }
    pub fn is_degenerate(&self) -> bool {
        self.top_radius <= 1e-3 || self.bot_radius <= 1e-3 || self.height() <= 1e-3
    }

    pub fn volume(&self) -> F {
        assert!(self.top_radius > 0.);
        assert!(self.bot_radius > 0.);
        let h = self.height();
        let tr = self.top_radius;
        let br = self.bot_radius;
        assert!(h > 0.);
        PI * h / 3. * (sqr(tr) + tr * br + sqr(br))
    }

    pub fn expand_to(&mut self, ops: impl Iterator<Item = [F; 3]> + Clone) {
        let axis = normalize(self.axis);
        self.bot_radius = 0.;
        self.top_radius = 0.;

        // single pass to compute max height
        let mut height: F = 0.;
        for op in ops.clone() {
            let p = sub(op, self.p);
            let h = dot(p, axis);
            if h < 0. {
                self.p = add(self.p, kmul(h - 1e-6, axis));
                assert!(dot(sub(op, self.p), axis) >= 0.);
                height -= h - 2e-6;
            } else {
                height = height.max(h);
            }
        }
        self.axis = kmul(height, axis);
        if height == 0. {
            return;
        }

        // (w, y_ratio, width at boundaries)
        let mut top_rad = (self.top_radius, 1., self.top_radius);
        let mut bot_rad = (self.bot_radius, 0., self.bot_radius);

        for op in ops.clone() {
            let p = sub(op, self.p);

            let o = dot(p, axis);
            let unit_y = o / height;
            assert!((0.0..=1.0).contains(&unit_y), "{unit_y}");

            // orthogonal component
            let ortho = sub(p, kmul(o, axis));

            let rad = length(ortho);

            use std::cmp::Ordering::*;
            match unit_y.partial_cmp(&0.5).unwrap() {
                Equal | Less => {
                    let next_bot = (rad - top_rad.2 * unit_y) / (1. - unit_y);
                    let next_bot = next_bot.abs();

                    if next_bot > bot_rad.2 {
                        bot_rad = (rad, unit_y, next_bot);

                        let next_top_w = (top_rad.0 - next_bot * (1. - top_rad.1)) / top_rad.1;
                        top_rad.2 = next_top_w.abs();
                    }
                }
                Greater => {
                    let next_top = (rad - bot_rad.2 * (1. - unit_y)) / unit_y;
                    let next_top = next_top.abs();

                    if next_top > top_rad.2 {
                        top_rad = (rad, unit_y, next_top);

                        let next_bot_w = (bot_rad.0 - next_top * bot_rad.1) / (1. - bot_rad.1);
                        bot_rad.2 = next_bot_w.abs();
                    }
                }
            };
        }

        self.bot_radius = top_rad.2;
        self.top_radius = bot_rad.2;

        assert!(self.bot_radius >= 0.);
        assert!(self.top_radius >= 0.);
    }
    pub fn surface_area(&self) -> F {
        let h = self.height();
        let l = (sqr(h) + sqr(self.top_radius - self.bot_radius)).sqrt();
        let top_area = sqr(self.top_radius);
        let bot_area = sqr(self.bot_radius);
        let slant_area = l * (self.top_radius * self.bot_radius);
        PI * (slant_area + bot_area * top_area)
    }
    pub fn signed_dist(&self, p: [F; 3]) -> F {
        let p = sub(p, self.p);

        let a = [0.; 3];
        let b = self.axis;

        let ra = self.bot_radius;
        let rb = self.top_radius;

        let rba = rb - ra;

        let ba = sub(b, a);
        let pa = sub(p, a);

        let baba = dot(ba, ba);
        let papa = dot(pa, pa);
        let paba = dot(pa, ba) / baba;

        let x = (papa - sqr(paba) * baba).max(0.).sqrt();

        let cax = x - if paba < 0.5 { ra } else { rb };
        let cax = cax.max(0.);
        let cay = (paba - 0.5).abs() - 0.5;

        let k = sqr(rba) + baba;
        let f = (rba * (x - ra) + paba * baba) / k;
        let f = f.clamp(0., 1.);

        let cbx = x - ra - f * rba;
        let cby = paba - f;

        let v = (sqr(cax) + sqr(cay) * baba)
            .min(sqr(cbx) + sqr(cby) * baba)
            .sqrt();
        if cbx < 0. && cay < 0. {
            -v
        } else {
            v
        }
    }
    pub fn to_mesh(
        &self,
        n: usize,
    ) -> (
        Vec<[F; 3]>,
        Vec<[usize; 4]>,
        [impl Iterator<Item = usize> + Clone; 2],
    ) {
        let (v, f) = frustum_to_quad_mesh(
            n,
            self.p,
            normalize(self.axis),
            self.bot_radius,
            self.top_radius,
            0.,
            self.height(),
        );
        let nv = v.len();
        (v, f, cylinder_caps(nv))
    }
}

#[test]
fn test_frustum_signed_dist() {
    let frustum = Frustum {
        p: [0.; 3],
        axis: [0., 1., 0.],
        top_radius: 0.1,
        bot_radius: 1.,
    };
    let fsd = frustum.signed_dist([1., 0., 0.]);
    assert_eq!(fsd, 0.);

    let fsd = frustum.signed_dist([1., 1.0000001, 1.]);
    assert!(fsd > 0., "{fsd}");

    assert_eq!(frustum.signed_dist([0., 0., 0.]), 0.);
    assert_eq!(frustum.signed_dist([0., 1., 0.]), 0.);

    let fsd = frustum.signed_dist([1., 1., 1.]);
    assert!(fsd > 0., "{fsd}");
}

fn sqr(x: F) -> F {
    x * x
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Torus {
    /// Center point of the torus
    p: [F; 3],
    /// Direction through which you can pass through the center of the torus.
    up: [F; 3],
    /// Radius of the hole through the torus.
    major_radius: F,
    /// The radius of the volume of the torus
    minor_radius: F,
}

impl Torus {
    /// Attempt to construct a torus from a capped cylinder
    pub fn from_capped_cylinder(cc: &CappedCylinder) -> Self {
        let minor_radius = cc.height / 2.;
        let up = normalize(cc.cylinder.axis);
        let p = add(cc.cylinder.p, kmul(minor_radius, up));
        let major_radius = cc.cylinder.radius - minor_radius;

        Self {
            p,
            up,
            minor_radius,
            major_radius,
        }
    }
    pub fn volume(&self) -> F {
        PI * sqr(self.minor_radius) * TAU * self.major_radius
    }
    /// Converts this torus into a series of capsules which approximate the torus
    pub fn to_capsules(&self, n: usize) -> impl Iterator<Item = Capsule> + '_ {
        let tan = normalize(orthogonal(self.up));
        let bit = normalize(cross(self.up, tan));
        (0..n).map(move |i| {
            let t = i as F / n as F;
            let tn = (i + 1) as F / n as F;

            let [p0, p1] = [t, tn]
                .map(|t| add(kmul((t * TAU).cos(), tan), kmul((t * TAU).sin(), bit)))
                .map(|d| add(kmul(self.major_radius, d), self.p));
            Capsule::from_ends(p0, p1, self.minor_radius)
        })
    }
}

#[test]
pub fn test_torus_to_mesh() {
    use super::faces_to_neg_idx_with_max;

    let cyl = Cylinder::new([0.; 3], [0., 1., 0.], 5.);
    let cap_cyl = CappedCylinder::new(cyl, 1.);
    let t = Torus::from_capped_cylinder(&cap_cyl);
    for c in t.to_capsules(8) {
        let (c_v, q_f, t_f) = c.to_mesh(8);
        for &[x, y, z] in &c_v {
            println!("v {x} {y} {z}");
        }
        for [vi0, vi1, vi2, vi3] in faces_to_neg_idx_with_max(&q_f, c_v.len() - 1) {
            println!("f {vi0} {vi1} {vi2} {vi3}");
        }
        for [vi0, vi1, vi2] in faces_to_neg_idx_with_max(&t_f, c_v.len() - 1) {
            println!("f {vi0} {vi1} {vi2}");
        }
    }
}
