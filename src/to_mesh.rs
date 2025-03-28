use super::{F, add, cross, kmul, normalize};
use std::collections::BTreeMap;

/// returns positions on an ico sphere, as well as faces
pub fn ico_sphere(subdivisions: usize) -> (Vec<[F; 3]>, Vec<[usize; 3]>) {
    const X: F = 0.525731112119133606;
    const Z: F = 0.850650808352039932;
    const N: F = 0.;

    let mut vertices = vec![
        [-X, N, Z],
        [X, N, Z],
        [-X, N, -Z],
        [X, N, -Z],
        [N, Z, X],
        [N, Z, -X],
        [N, -Z, X],
        [N, -Z, -X],
        [Z, X, N],
        [-Z, X, N],
        [Z, -X, N],
        [-Z, -X, N],
    ];
    let mut triangles = vec![
        [0, 1, 4],
        [0, 4, 9],
        [9, 4, 5],
        [4, 8, 5],
        [4, 1, 8],
        [8, 1, 10],
        [3, 8, 10],
        [3, 5, 8],
        [2, 5, 3],
        [2, 3, 7],
        [7, 3, 10],
        [7, 10, 6],
        [7, 6, 11],
        [11, 6, 0],
        [0, 6, 1],
        [1, 6, 10],
        [0, 9, 11],
        [9, 2, 11],
        [9, 5, 2],
        [2, 7, 11],
    ];
    let mut map = BTreeMap::new();
    for _ in 0..subdivisions {
        triangles = subdivide(&mut vertices, &triangles, &mut map);
    }
    (vertices, triangles)
}
fn subdivide(
    v: &mut Vec<[F; 3]>,
    fs: &[[usize; 3]],
    map: &mut BTreeMap<[usize; 2], usize>,
) -> Vec<[usize; 3]> {
    let mut out = vec![];
    let edge = |a: usize, b: usize| [a.min(b), a.max(b)];
    let mut v_for_e = |e: [usize; 2]| {
        *map.entry(e).or_insert_with(|| {
            let mid = normalize(add(v[e[0]], v[e[1]]));
            let p = v.len();
            v.push(mid);
            p
        })
    };
    for f in fs {
        let mids = std::array::from_fn(|i| {
            let n = (i + 1) % 3;
            v_for_e(edge(f[i], f[n]))
        });
        out.push([f[0], mids[0], mids[2]]);
        out.push([f[1], mids[1], mids[0]]);
        out.push([f[2], mids[2], mids[1]]);
        out.push(mids);
    }
    out
}

pub fn frustum_to_quad_mesh(
    pts: usize,
    center: [F; 3],
    axis: [F; 3],
    bot_r: F,
    top_r: F,
    down: F,
    up: F,
) -> (Vec<[F; 3]>, Vec<[usize; 4]>) {
    let mut vertices = vec![];
    use super::quat_rot;

    let (axis, up, down, bot_r, top_r) = if super::dot(axis, [0., 1., 0.]) < 0. {
        (axis, up, down, bot_r, top_r)
    } else {
        (axis.map(core::ops::Neg::neg), -down, -up, top_r, bot_r)
    };

    let r_axis = if axis == [0., -1., 0.] {
        [0., 1., 0.]
    } else {
        axis
    };
    let q = super::quat_from_to([0., 1., 0.], r_axis);

    let t = 2. * (std::f64::consts::PI as F);
    let [up, down] = [up, down].map(|v| kmul(v, axis));
    for i in 0..pts {
        let i = i as F / pts as F;

        let r0 = [(i * t).cos(), 0., (i * t).sin()];
        let r0 = quat_rot(r0, q);
        vertices.push(add(kmul(top_r, r0), down));
        vertices.push(add(kmul(bot_r, r0), up));
    }

    for v in &mut vertices {
        *v = add(center, *v);
    }

    let mut faces = vec![];
    for i in 0..pts {
        faces.push([
            i * 2,
            i * 2 + 1,
            (i * 2 + 3) % vertices.len(),
            (i * 2 + 2) % vertices.len(),
        ]);
    }

    (vertices, faces)
}
pub fn cylinder_caps(n: usize) -> [impl Iterator<Item = usize> + Clone; 2] {
    std::array::from_fn(|i| {
        (0..n)
            .map(move |v| if i == 0 { v } else { n - v - 1 })
            .filter(move |v| v % 2 == i)
    })
}

/// Constructs a capsule mesh quantized to a cylinder with n vertices on its ring
pub fn capsule_to_mesh(
    n: usize,
    p: [F; 3],
    axis: [F; 3],

    up: F,
    down: F,
    // radius
    r: F,
) -> (Vec<[F; 3]>, Vec<[usize; 4]>, Vec<[usize; 3]>) {
    let mut vertices = vec![];
    use super::quat_rot;

    let r_axis = if axis == [0., -1., 0.] {
        [0., 1., 0.]
    } else {
        axis
    };
    let q = super::quat_from_to([0., 1., 0.], r_axis);

    let t = 2. * (std::f64::consts::PI as F);
    let [up, down] = [up, down].map(|v| kmul(v, axis));

    let curve_downs = [-60., -30., 0.].map(|v: F| v.to_radians());
    let curve_ups = [0., 30., 60.].map(|v: F| v.to_radians());

    let nc = curve_downs.len() + curve_ups.len();
    for i in 0..n {
        let i = i as F / n as F;

        for c in curve_downs {
            let r0 = [(i * t).cos() * c.cos(), c.sin(), (i * t).sin() * c.cos()];
            let r0 = quat_rot(r0, q);
            let r0 = kmul(r, r0);
            vertices.push(add(r0, down));
        }
        for c in curve_ups {
            let r0 = [(i * t).cos() * c.cos(), c.sin(), (i * t).sin() * c.cos()];
            let r0 = quat_rot(r0, q);
            let r0 = kmul(r, r0);
            vertices.push(add(r0, up));
        }
    }
    assert_eq!(vertices.len() % nc, 0);

    let curve_len = vertices.len();
    // add top vertex
    vertices.push(add(up, kmul(r, axis)));
    let top = vertices.len() - 1;
    vertices.push(add(down, kmul(-r, axis)));
    let bot = vertices.len() - 1;

    for v in &mut vertices {
        *v = add(p, *v);
    }

    let mut quads = vec![];
    for i in 0..n {
        for j in 0..(nc - 1) {
            quads.push([
                i * nc + j,
                i * nc + j + 1,
                ((i + 1) * nc + j + 1) % curve_len,
                ((i + 1) * nc + j) % curve_len,
            ]);
        }
    }

    let mut tris = vec![];
    for i in 0..n {
        tris.push([i * nc, (i + 1) * nc % curve_len, bot]);
        tris.push([(i + 1) * nc - 1, top, ((i + 2) * nc - 1) % curve_len]);
    }

    (vertices, quads, tris)
}

pub fn cone_to_mesh(
    p: [F; 3],
    axis: [F; 3],
    r: F,
    n: usize,
    with_bot: bool,
) -> (Vec<[F; 3]>, Vec<[usize; 3]>) {
    let mut v = vec![];
    let mut f = vec![];
    let a0 = normalize(cross(axis, [0.33, 0.33, 0.33]));
    let a1 = normalize(cross(axis, a0));

    let t = 2. * (std::f64::consts::PI as F);
    for i in 0..n {
        let i = i as F / n as F;

        let cp = add(kmul(r * (i * t).cos(), a0), kmul(r * (i * t).sin(), a1));
        v.push(add(p, cp));
    }

    for i in 0..n {
        f.push([i, (i + 1) % n, v.len()]);
        if with_bot {
            f.push([(i + 1) % n, i, v.len() + 1]);
        }
    }
    v.push(add(p, axis));
    if with_bot {
        v.push(p);
    }

    (v, f)
}

#[ignore]
#[test]
fn cone_to_mesh_test() {
    let (v, f) = cone_to_mesh([0.; 3], [0., 2., 0.], 1., 8, true);

    for [x, y, z] in v {
        println!("v {x} {y} {z}");
    }
    for vis in f {
        let [x, y, z] = vis.map(|vi| vi + 1);
        println!("f {x} {y} {z}");
    }
}
