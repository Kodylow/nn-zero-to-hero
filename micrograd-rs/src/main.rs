extern crate ndarray;
extern crate plotters;

use std::{cell::RefCell, fmt, rc::Rc};

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

#[derive(Clone)]
struct Value {
    data: f64,
    grad: f64,
    _backward: Rc<dyn FnMut()>,
    _prev: Vec<Rc<RefCell<Value>>>,
    _op: String,
    label: String,
}

impl Value {
    fn new(data: f64, _children: Vec<Rc<RefCell<Value>>>, _op: String, label: String) -> Self {
        Value {
            data,
            grad: 0.0,
            _backward: Rc::new(|| {}),
            _prev: _children,
            _op,
            label,
        }
    }

    fn add(&mut self, other: Rc<RefCell<Value>>) -> Rc<RefCell<Value>> {
        let out = Rc::new(RefCell::new(Value::new(
            self.data + other.borrow().data,
            vec![Rc::clone(&other)],
            "+".to_string(),
            "".to_string(),
        )));

        let self_weak = Rc::downgrade(&Rc::new(RefCell::new(self.clone())));
        let other_weak = Rc::downgrade(&other);

        let out_clone = Rc::clone(&out);
        out.borrow_mut()._backward = Rc::new(move || {
            if let (Some(self_rc), Some(other_rc)) = (self_weak.upgrade(), other_weak.upgrade()) {
                self_rc.borrow_mut().grad += out_clone.borrow().grad;
                other_rc.borrow_mut().grad += out_clone.borrow().grad;
            }
        });

        out
    }

    fn mul(&mut self, other: Rc<RefCell<Value>>) -> Rc<RefCell<Value>> {
        let out = Rc::new(RefCell::new(Value::new(
            self.data * other.borrow().data,
            vec![Rc::clone(&other)],
            "*".to_string(),
            "".to_string(),
        )));

        let self_weak = Rc::downgrade(&Rc::new(RefCell::new(self.clone())));
        let other_weak = Rc::downgrade(&other);

        let out_clone = Rc::clone(&out);
        out.borrow_mut()._backward = Rc::new(move || {
            if let (Some(self_rc), Some(other_rc)) = (self_weak.upgrade(), other_weak.upgrade()) {
                self_rc.borrow_mut().grad += other_rc.borrow().data * out_clone.borrow().grad;
                other_rc.borrow_mut().grad += self_rc.borrow().data * out_clone.borrow().grad;
            }
        });

        out
    }

    fn tanh(&mut self) -> Rc<RefCell<Value>> {
        let x = self.data;
        let t = (2.0 * x).exp() - 1.0 / (2.0 * x).exp() + 1.0;
        let out = Rc::new(RefCell::new(Value::new(
            t,
            vec![Rc::new(self.clone().into())],
            "tanh".to_string(),
            "".to_string(),
        )));

        let out_clone = Rc::clone(&out);
        out.borrow_mut()._backward = Rc::new(move || {
            let out_grad = out_clone.borrow().grad;
            let tanh_grad = 1.0 - t.powi(2);
            out_clone.borrow_mut().grad += tanh_grad * out_grad;
        });

        out
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Value")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .field("_prev", &self._prev)
            .field("_op", &self._op)
            .field("label", &self.label)
            .finish()
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.grad == other.grad
            && self._prev == other._prev
            && self._op == other._op
            && self.label == other.label
    }
}
