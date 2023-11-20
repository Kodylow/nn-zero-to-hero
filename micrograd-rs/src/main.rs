use std::fs::File;
use std::io::Write;
use std::process::Command;

use micrograd_rs::{draw_dot, MyF64, Op, Value, ValueGraph};
use petgraph::dot::{Config, Dot};

fn main() {
    let mut graph = ValueGraph::new();

    let (a_index, b_index, c_index, f_index) = create_data(&mut graph);

    let e_index = graph.mul(a_index, b_index);
    graph.get_value_mut(e_index).unwrap().label = "e".to_string();

    let d_index = graph.add(e_index, c_index);
    graph.get_value_mut(d_index).unwrap().label = "d".to_string();

    let l_index = graph.mul(d_index, f_index);
    graph.get_value_mut(l_index).unwrap().label = "l".to_string();

    let indices = vec![a_index, b_index, c_index, f_index];
    update_values(&mut graph, indices);

    let dot_graph = draw_dot(&graph, l_index);
    let dot = format!("{:?}", Dot::with_config(&dot_graph, &[Config::EdgeNoLabel]));

    let mut file = File::create("graph.dot").expect("Could not create file");
    file.write_all(dot.as_bytes())
        .expect("Could not write to file");

    // Run `dot -Tpng graph.dot -o graph.png`
    Command::new("dot")
        .args(&["-Tpng", "graph.dot", "-o", "graph.png"])
        .spawn()
        .expect("Could not run dot");
}

fn create_data(graph: &mut ValueGraph) -> (usize, usize, usize, usize) {
    let a_index = graph.add_value(Value::new(
        "a".to_string(),
        MyF64(2.0),
        MyF64(0.0),
        Op::None,
        Vec::new(),
    ));
    let b_index = graph.add_value(Value::new(
        "b".to_string(),
        MyF64(-3.0),
        MyF64(0.0),
        Op::None,
        Vec::new(),
    ));
    let c_index = graph.add_value(Value::new(
        "c".to_string(),
        MyF64(10.0),
        MyF64(0.0),
        Op::None,
        Vec::new(),
    ));
    let f_index = graph.add_value(Value::new(
        "f".to_string(),
        MyF64(-2.0),
        MyF64(0.0),
        Op::None,
        Vec::new(),
    ));

    (a_index, b_index, c_index, f_index)
}

fn update_values(graph: &mut ValueGraph, indices: Vec<usize>) {
    for index in indices {
        let value = graph.get_value_mut(index).unwrap();
        value.data.0 += 0.01 * value.grad.0;
    }
}
