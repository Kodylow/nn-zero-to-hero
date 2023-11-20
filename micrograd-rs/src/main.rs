use std::fs::File;

use micrograd_rs::{draw_dot, MyF64, Op, Value};
use petgraph::dot::{Config, Dot};
use std::io::Write;
use std::process::Command;

fn main() {
    let _h = Value::new(
        "h".to_string(),
        MyF64(0.0001),
        MyF64(0.0),
        Op::None,
        Vec::new(),
    );
    let a = Value::new(
        "a".to_string(),
        MyF64(2.0),
        MyF64(0.0),
        Op::None,
        Vec::new(),
    );
    let b = Value::new(
        "b".to_string(),
        MyF64(-3.0),
        MyF64(0.0),
        Op::None,
        Vec::new(),
    );
    let c = Value::new(
        "c".to_string(),
        MyF64(10.0),
        MyF64(0.0),
        Op::None,
        Vec::new(),
    );
    let mut e = a.mul(&b);
    e.label = "e".to_string();

    let mut d = e.add(&c);
    d.label = "d".to_string();
    let f = Value::new(
        "f".to_string(),
        MyF64(-2.0),
        MyF64(0.0),
        Op::None,
        Vec::new(),
    );

    let mut l = d.mul(&f);
    l.label = "l".to_string();

    let graph = draw_dot(&l);
    let dot = format!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));

    let mut file = File::create("graph.dot").expect("Could not create file");
    file.write_all(dot.as_bytes())
        .expect("Could not write to file");

    // run `dot -Tpng graph.dot -o graph.png`
    Command::new("dot")
        .args(&["-Tpng", "graph.dot", "-o", "graph.png"])
        .spawn()
        .expect("Could not run dot");
}
