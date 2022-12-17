use ndarray::{arr1, arr2, array, Array1, Array2, Axis};
use ndarray_linalg::{error, random, Solve};
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{PointMarker, PointStyle};
use plotlib::view::ContinuousView;

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

    let data1 = [
        (-3.0, 2.3),
        (-1.6, 5.3),
        (0.3, 0.7),
        (4.3, -1.4),
        (6.4, 4.3),
        (8.5, 3.7),
    ];

    let s1: Plot = Plot::new(data1).point_style(
        PointStyle::new()
            .marker(PointMarker::Square)
            .colour("#DD3355"),
    );

    let data2 = vec![(-1.4, 2.5), (7.2, -0.3)];
    let s2: Plot = Plot::new(data2).point_style(
        PointStyle::new().colour("#35C788")
    );

    let v = ContinuousView::new()
        .add(s1)
        .add(s2)
        .x_range(-5., 10.)
        .y_range(-2., 6.)
        .x_label("x label")
        .y_label("y label");

    Page::single(&v).save("scatter.svg").unwrap();
}
