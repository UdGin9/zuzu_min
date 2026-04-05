use opencv::core::{Mat};
use opencv::prelude::MatTraitConst;
type Mat4 = [[f64; 4]; 4];

pub fn translation_matrix(x: f64, y: f64, z: f64) -> Mat4 {
    [
        [1.0, 0.0, 0.0,  x ],
        [0.0, 1.0, 0.0,  y ],
        [0.0, 0.0, 1.0,  z ],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn rotation_matrix_x(a: f64) -> Mat4 {
    [
        [1.0,    0.0,      0.0,  0.0],
        [0.0,  a.cos(), -a.sin(), 0.0],
        [0.0,  a.sin(),  a.cos(), 0.0],
        [0.0,    0.0,      0.0,  1.0],
    ]
}

pub fn rotation_matrix_y(a: f64) -> Mat4 {
    [
        [ a.cos(), 0.0, a.sin(), 0.0],
        [   0.0,   1.0,   0.0,  0.0],
        [-a.sin(), 0.0, a.cos(), 0.0],
        [   0.0,   0.0,   0.0,  1.0],
    ]
}

pub fn rotation_matrix_z(a: f64) -> Mat4 {
    [
        [a.cos(), -a.sin(), 0.0, 0.0],
        [a.sin(),  a.cos(), 0.0, 0.0],
        [  0.0,     0.0,   1.0, 0.0],
        [  0.0,     0.0,   0.0, 1.0],
    ]
}

pub fn matmul(a: Mat4, b: Mat4) -> Mat4 {
    let mut r = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    r
}

pub fn rotation_matrix_euler(x: f64, y: f64, z: f64) -> Mat4 {
    matmul(matmul(rotation_matrix_y(y.to_radians()), rotation_matrix_x(x.to_radians())), rotation_matrix_z(z.to_radians()))
}

pub fn euler_from_matrix(m: Mat4) -> [f64; 3] {
    let sy = (m[0][0] * m[0][0] + m[1][0] * m[1][0]).sqrt();

    if sy > 1e-6 {
        let x = m[2][1].atan2(m[2][2]);
        let y = (-m[2][0]).atan2(sy);
        let z = m[1][0].atan2(m[0][0]);
        [x, y, z]
    } else {
        let x = (-m[1][2]).atan2(m[1][1]);
        let y = (-m[2][0]).atan2(sy);
        [x, y, 0.0]
    }
}

pub fn mat_to_mat4(m: &Mat) -> [[f64; 4]; 4] {
    let mut r = [[0.0f64; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = *m.at_2d::<f64>(i as i32, j as i32).unwrap();
        }
    }
    r[3][3] = 1.0;
    r
}