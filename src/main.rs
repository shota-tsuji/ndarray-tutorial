use ndarray::{arr1, arr2, array, Array1, Array2, Axis};
use ndarray_linalg::{error, random, Solve};

fn main() {
    let mat1 = arr2(&[
        [4., -17.],
        [15., -4.],
        [30., -7.],
        [100., 50.],
        [200., 70.],
    ]);

    let (v_x, v_y) = mat1.view().split_at(Axis(1), 1);
    let a_11 = v_x.mapv(|v: f64| v.powf(2.)).sum();
    let a_12 = v_x.sum();
    let a_22 = v_x.mapv(|_| 1.).sum();

    let b_1 = v_x.iter().zip(v_y.iter()).map(|(x, y)| x * y).sum();
    let b_2 = v_y.sum();

    let mat_a: Array2<f64> = arr2(&[
        [a_11, a_12],
        [a_12, a_22],
    ]);
    println!("{}", mat_a);

    let v_b = arr1(&[b_1, b_2]);
    println!("{}", v_b);

    let x = mat_a.solve(&v_b).unwrap();
    println!("{}", x);

}
