use std::borrow::Cow;
use std::slice;

use dst::{Output, OutputId, TransformIdx};
use transform::{Transformation, TypeId};

/// Identifies a [`Node`] in a [`DST`]. A node can either be a [`Transformation`],
/// in that case it is identified by a [`TransformIdx`], or an [`OutputId`].
///
/// Use it together with [`DST::get_node`].
#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum NodeId {
    Transform(TransformIdx),
    Output(OutputId),
}

/// Represents a [`Node`], which is either a [`Transformation`] or some
/// [`Output`].
pub enum Node<'a, T: 'a + Clone, E: 'a> {
    Transform(&'a Transformation<T, E>),
    /// [`Output`] is `None` when there is an [`OutputId`] not connected to any
    /// [`Output`].
    Output(Option<&'a Output>),
}

impl<'a, T: Clone, E> Node<'a, T, E> {
    /// Get node's name.
    pub fn name(&'a self, id: &NodeId) -> Cow<'static, str> {
        match self {
            &Node::Transform(t) => Cow::Borrowed(t.name),
            &Node::Output(_) => {
                if let NodeId::Output(output_id) = id {
                    Cow::Owned(format!("Output {:?}", output_id))
                } else {
                    panic!("Expected id to be output")
                }
            }
        }
    }

    /// Iterate over each type of the inputs
    pub fn inputs_iter(&'a self) -> slice::Iter<'a, TypeId> {
        const OUTPUT_NODE_SLOTS: [TypeId; 1] = ["Out"];
        match self {
            &Node::Transform(t) => t.input.iter(),
            &Node::Output(_) => OUTPUT_NODE_SLOTS.iter(),
        }
    }

    /// Return number of inputs
    pub fn inputs_count(&self) -> usize {
        match self {
            &Node::Transform(t) => t.input.len(),
            &Node::Output(_) => 1,
        }
    }

    /// Iterate over each type of the outputs
    pub fn outputs_iter(&'a self) -> slice::Iter<'a, TypeId> {
        const OUTPUT_NODE_SLOTS: [TypeId; 0] = [];
        match self {
            &Node::Transform(t) => t.output.iter(),
            &Node::Output(_) => OUTPUT_NODE_SLOTS.iter(),
        }
    }

    /// Return number of outputs
    pub fn outputs_count(&self) -> usize {
        match self {
            &Node::Transform(t) => t.output.len(),
            &Node::Output(_) => 0,
        }
    }
}
