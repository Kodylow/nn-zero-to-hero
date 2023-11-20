use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use petgraph::{stable_graph::NodeIndex, Graph};

#[derive(Debug, Clone, PartialEq)]
pub struct MyF64(pub f64);

impl Eq for MyF64 {}

impl std::hash::Hash for MyF64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Op {
    Add,
    Mul,
    Tanh,
    None,
}

impl fmt::Debug for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::Add => write!(f, "+"),
            Op::Mul => write!(f, "*"),
            Op::Tanh => write!(f, "tanh"),
            Op::None => write!(f, ""),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Value {
    pub label: String,
    pub data: MyF64,
    pub grad: MyF64,
    pub op: Op,
    pub prev: Vec<usize>,
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{{ {} | data {:.4} | grad {:.4} }}",
            self.label, self.data.0, self.grad.0
        )
    }
}

impl Default for Value {
    fn default() -> Self {
        Value {
            label: String::new(),
            data: MyF64(0.0),
            grad: MyF64(0.0),
            op: Op::None,
            prev: Vec::new(),
        }
    }
}

impl Value {
    pub fn new(label: String, data: MyF64, grad: MyF64, op: Op, prev: Vec<usize>) -> Self {
        Value {
            label,
            data,
            grad,
            op,
            prev,
        }
    }
}

pub struct ValueGraph {
    values: HashMap<usize, Value>,
    next_index: usize,
}

impl ValueGraph {
    pub fn new() -> Self {
        ValueGraph {
            values: HashMap::new(),
            next_index: 0,
        }
    }

    pub fn add_value(&mut self, value: Value) -> usize {
        let index = self.next_index;
        self.values.insert(index, value);
        self.next_index += 1;
        index
    }

    pub fn get_value_mut(&mut self, index: usize) -> Option<&mut Value> {
        self.values.get_mut(&index)
    }

    pub fn get_value(&self, index: usize) -> Option<&Value> {
        self.values.get(&index)
    }

    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let a_val = self.get_value(a).expect("Invalid index for 'a'");
        let b_val = self.get_value(b).expect("Invalid index for 'b'");

        let out = Value {
            label: format!("({} + {})", a_val.label, b_val.label),
            data: MyF64(a_val.data.0 + b_val.data.0),
            grad: MyF64(0.0),
            op: Op::Add,
            prev: vec![a, b],
        };

        self.add_value(out)
    }

    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let a_val = self.get_value(a).expect("Invalid index for 'a'");
        let b_val = self.get_value(b).expect("Invalid index for 'b'");

        let out = Value {
            label: format!("({} * {})", a_val.label, b_val.label),
            data: MyF64(a_val.data.0 * b_val.data.0),
            grad: MyF64(0.0),
            op: Op::Mul,
            prev: vec![a, b],
        };

        self.add_value(out)
    }

    pub fn tanh(&mut self, a: usize) -> usize {
        let a_val = self.get_value(a).expect("Invalid index for 'a'");
        let t = (2.0 * a_val.data.0).exp();
        let out_data = (t - 1.0) / (t + 1.0);
        let out = Value {
            label: format!("tanh({})", a_val.label),
            data: MyF64(out_data),
            grad: MyF64(0.0),
            op: Op::Tanh,
            prev: vec![a],
        };

        self.add_value(out)
    }

    pub fn backward(&mut self, index: usize) {
        let mut prev = Vec::new();
        if let Some(value) = self.get_value_mut(index) {
            // Initialize gradient if it's the end of the graph
            if value.prev.is_empty() {
                value.grad = MyF64(1.0);
            }

            let grad = value.grad.0;
            let data = value.data.0;
            prev = value.prev.clone();

            match value.op {
                Op::Add => {
                    for &prev_index in &prev {
                        if let Some(prev_value) = self.get_value_mut(prev_index) {
                            prev_value.grad.0 += grad;
                        }
                    }
                }
                Op::Mul => {
                    for &prev_index in &prev {
                        if let Some(prev_value) = self.get_value_mut(prev_index) {
                            prev_value.grad.0 += grad * (data / prev_value.data.0);
                        }
                    }
                }
                Op::Tanh => {
                    for &prev_index in &prev {
                        if let Some(prev_value) = self.get_value_mut(prev_index) {
                            prev_value.grad.0 += grad * (1.0 - data.powi(2));
                        }
                    }
                }
                _ => {}
            }
        }

        // Recurse into previous values
        for &prev_index in &prev {
            self.backward(prev_index);
        }
    }
}

pub fn trace(
    graph: &ValueGraph,
    index: usize,
    nodes: &mut HashSet<usize>,
    edges: &mut HashSet<(usize, usize)>,
) {
    if !nodes.contains(&index) {
        nodes.insert(index);
        if let Some(value) = graph.get_value(index) {
            for &child_index in &value.prev {
                edges.insert((child_index, index));
                trace(graph, child_index, nodes, edges);
            }
        }
    }
}

pub fn draw_dot(graph: &ValueGraph, root_index: usize) -> Graph<String, String> {
    let mut graph_draw = Graph::<String, String>::new();
    let mut nodes_map = HashMap::new();
    let mut edges_set = HashSet::new();

    // Start the recursive visiting process
    visit(
        graph,
        root_index,
        &mut graph_draw,
        &mut nodes_map,
        &mut edges_set,
    );

    graph_draw
}

// Function to recursively visit nodes and add them to the graph
fn visit(
    graph: &ValueGraph,
    index: usize,
    graph_draw: &mut Graph<String, String>,
    nodes_map: &mut HashMap<String, NodeIndex>,
    edges_set: &mut HashSet<(String, String)>,
) {
    if let Some(value) = graph.get_value(index) {
        let value_label = &value.label;

        if !nodes_map.contains_key(value_label) {
            let label = match value.op {
                Op::None => format!(
                    "  {} | data {:.4} | grad {:.4}  ",
                    value.label, value.data.0, value.grad.0
                ),
                _ => format!(
                    "  {:?} | {} | data {:.4} | grad {:.4}  ",
                    value.op, value.label, value.data.0, value.grad.0
                ),
            };
            let node_index = graph_draw.add_node(label);
            nodes_map.insert(value_label.clone(), node_index);

            for &prev_index in &value.prev {
                if let Some(prev_value) = graph.get_value(prev_index) {
                    visit(graph, prev_index, graph_draw, nodes_map, edges_set);
                    add_edge_if_not_exists(prev_value, value, graph_draw, nodes_map, edges_set);
                }
            }
        }
    }
}

fn add_edge_if_not_exists(
    prev_value: &Value,
    value: &Value,
    graph: &mut Graph<String, String>,
    nodes_map: &mut HashMap<String, NodeIndex>,
    edges_set: &mut HashSet<(String, String)>,
) {
    let prev_label = &prev_value.label;
    let value_label = &value.label;

    // Avoid adding duplicate edges
    if !edges_set.contains(&(prev_label.clone(), value_label.clone())) {
        let prev_index = *nodes_map.get(prev_label).unwrap();
        graph.add_edge(prev_index, nodes_map[value_label], "".to_string());
        edges_set.insert((prev_label.clone(), value_label.clone()));
    }
}
