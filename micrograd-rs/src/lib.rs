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
    None,
}

impl fmt::Debug for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::Add => write!(f, "+"),
            Op::Mul => write!(f, "*"),
            Op::None => write!(f, ""),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Value<'a> {
    pub label: String,
    pub data: MyF64,
    pub grad: MyF64,
    pub op: Op,
    pub prev: Vec<&'a Value<'a>>,
}

impl fmt::Debug for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{{ {} | data {:.4} | grad {:.4} }}",
            self.label, self.data.0, self.grad.0
        )
    }
}

impl Default for Value<'_> {
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

impl Value<'_> {
    pub fn new<'a>(
        label: String,
        data: MyF64,
        grad: MyF64,
        op: Op,
        prev: Vec<&'a Value<'_>>,
    ) -> Value<'a> {
        Value {
            label,
            data,
            grad,
            op,
            prev,
        }
    }
    pub fn add<'a>(&'a self, other: &'a Value<'a>) -> Value<'a> {
        let mut out = Value {
            label: format!("({} + {})", self.label, other.label),
            data: MyF64(self.data.0 + other.data.0),
            grad: MyF64(0.0),
            op: Op::Add,
            prev: Vec::new(),
        };
        out.prev.push(self);
        out.prev.push(other);
        out
    }

    pub fn mul<'a>(&'a self, other: &'a Value<'a>) -> Value<'a> {
        let mut out = Value {
            label: format!("({} * {})", self.label, other.label),
            data: MyF64(self.data.0 * other.data.0),
            grad: MyF64(0.0),
            op: Op::Mul,
            prev: Vec::new(),
        };
        out.prev.push(self);
        out.prev.push(other);
        out
    }
}

pub fn trace<'a>(
    root: &'a Value<'a>,
    nodes: &mut HashSet<Value<'a>>,
    edges: &mut HashSet<(Value<'a>, Value<'a>)>,
) {
    if !nodes.contains(root) {
        nodes.insert(root.clone());
        for child in &root.prev {
            edges.insert(((*child).clone(), root.clone()));
            trace(child, nodes, edges);
        }
    }
}

pub fn draw_dot<'a>(root: &'a Value<'a>) -> Graph<String, String> {
    let mut graph = Graph::<String, String>::new();
    let mut nodes_map = HashMap::new();
    let mut edges_set = HashSet::new();

    // Start the recursive visiting process
    visit(root, &mut graph, &mut nodes_map, &mut edges_set);

    graph
}

// Function to recursively visit nodes and add them to the graph
fn visit<'a>(
    value: &'a Value<'a>,
    graph: &mut Graph<String, String>,
    nodes_map: &mut HashMap<String, NodeIndex>,
    edges_set: &mut HashSet<(String, String)>,
) {
    let value_label = &value.label;

    if !nodes_map.contains_key(value_label) {
        let label = match value.op {
            Op::None => format!(
                "  {} | data {:.4} | grad {:.4}  ",
                value.label, value.data.0, value.grad.0
            ),
            _ => format!(
                "  op{:?} | {} | data {:.4} | grad {:.4}  ",
                value.op, value.label, value.data.0, value.grad.0
            ),
        };
        let index = graph.add_node(label);
        nodes_map.insert(value_label.clone(), index);

        for &prev_value in &value.prev {
            visit(prev_value, graph, nodes_map, edges_set);
            add_edge_if_not_exists(prev_value, value, graph, nodes_map, edges_set);
        }
    }
}

fn add_edge_if_not_exists<'a>(
    prev_value: &'a Value<'a>,
    value: &'a Value<'a>,
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
