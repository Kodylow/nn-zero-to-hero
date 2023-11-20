use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use petgraph::Graph;

#[derive(Debug, Clone, PartialEq)]
pub struct MyF64(pub f64);

impl Eq for MyF64 {}

impl std::hash::Hash for MyF64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Value<'a> {
    pub label: String,
    pub data: MyF64,
    pub grad: MyF64,
    pub op: String,
    pub prev: Vec<&'a Value<'a>>,
}

impl fmt::Display for Value<'_> {
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
            op: String::new(),
            prev: Vec::new(),
        }
    }
}

impl Value<'_> {
    pub fn new<'a>(
        label: String,
        data: MyF64,
        grad: MyF64,
        op: String,
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
            op: "+".to_string(),
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
            op: "*".to_string(),
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

pub fn draw_dot<'a>(root: &'a Value<'_>) -> Graph<Value<'a>, String> {
    let mut nodes = HashSet::new();
    let mut edges = HashSet::new();
    trace(root, &mut nodes, &mut edges);

    let mut graph = Graph::<Value, String>::new();
    let mut index_map = HashMap::new();

    for node in nodes {
        let index = graph.add_node(node.clone());
        let node_clone = node.clone();
        index_map.insert(node, index);
        if !node_clone.op.is_empty() {
            let op_index = graph.add_node(Value {
                label: node_clone.op.clone(),
                ..Default::default()
            });
            graph.add_edge(op_index, index, "".to_string());
        }
    }

    for (n1, n2) in edges {
        if let (Some(&index1), Some(&index2)) = (index_map.get(&n1), index_map.get(&n2)) {
            graph.add_edge(index1, index2, n2.op);
        }
    }

    graph
}
