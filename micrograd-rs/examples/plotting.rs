extern crate ndarray;
extern crate plotters;

use ndarray::Array;
use plotters::prelude::*;

fn main() {
    let xs = Array::range(-5., 5., 0.25);
    let ys: Vec<f32> = xs.iter().map(|&x| example_f(x)).collect();

    let root = BitMapBackend::new("plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("y = f(x)", ("Arial", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-5f32..5f32, 0f32..100f32)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(
            xs.iter().zip(ys.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))
        .unwrap();
}

fn example_f(x: f32) -> f32 {
    3.0 * x.powf(2.0) + 4.0 * x + 5.0
}
