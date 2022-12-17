use ndarray::{arr1, arr2, array, Array1, Array2, ArrayView, Axis, Order, Zip};
use ndarray_linalg::{error, random, Solve};
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{LineJoin, LineStyle, PointMarker, PointStyle};
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

    let coefficient_x = mat_a.solve(&v_b).unwrap();
    println!("{}", coefficient_x);

    let precision_y = v_x.map(|&x| f(coefficient_x[0], coefficient_x[1], x)).into_raw_vec();

    let mut data1 = Vec::new();
    Zip::from(v_x)
        .and(v_y)
        .for_each(|&x, &y| data1.push((x, y)));

    let s1: Plot = Plot::new(data1).point_style(
        PointStyle::new().colour("#35C788")
    );

    let mut precision = Vec::new();
    let v_x =  v_x.clone().to_shape(((v_x.len()), Order::RowMajor)).unwrap().view();
    Zip::from(v_x)
        .and(&precision_y)
        .for_each(|&x, &y| precision.push((x, y)));
    let l1 = Plot::new(precision).line_style(
        LineStyle::new()
            .colour("burlywood")
            .linejoin(LineJoin::Round)
    );

    let v = ContinuousView::new()
        .add(s1)
        .add(l1)
        .x_range(0., 230.)
        .y_range(-30., 100.)
        .x_label("x label")
        .y_label("y label");

    Page::single(&v).save("scatter.svg").unwrap();
}

fn f(a: f64, b: f64, x: f64) -> f64 {
    a * x + b
}