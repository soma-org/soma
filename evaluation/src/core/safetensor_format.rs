use ndarray::{ArrayD, Axis};
use ndarray_safetensors::parse_tensor_view_data;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use safetensors::SafeTensors;
use std::collections::{HashMap, HashSet};
use types::{
    checksum::Checksum,
    error::{EvaluationError, EvaluationResult},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EmbeddingIndex(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ByteIndex(u64);

const TENSORS_PER_EMBEDDING: usize = 3;
const AUXILLARY_TENSORS: usize = 1;
const MIN_EMBEDDING_INDEX: u64 = 1;

#[derive(Debug)]
struct EmbeddingAggregate {
    num_bytes_represented: usize,
    num_bytes_referenced: usize,
    num_bytes_used: usize,
    referenced_by: Vec<ByteIndex>,
}

pub struct ContextEmbedding {
    byte_index: ByteIndex,
    num_bytes_represented: usize,
    num_bytes_referenced: usize,
    num_bytes_used: usize,
    embedding: ArrayD<f32>,
}

pub struct IndexedTensors<'data> {
    safetensors: SafeTensors<'data>,
    embedding_aggregates: HashMap<EmbeddingIndex, EmbeddingAggregate>,
    byte_to_embedding: HashMap<ByteIndex, EmbeddingIndex>,
}

impl<'data> IndexedTensors<'data> {
    pub fn new(safetensors: SafeTensors<'data>) -> EvaluationResult<Self> {
        let graph = EmbeddingGraph::new(&safetensors)?;
        let embedding_aggregates = graph.aggregate();

        Ok(Self {
            safetensors,
            embedding_aggregates,
            byte_to_embedding: graph.byte_to_embedding,
        })
    }

    pub fn sample_context(
        &self,
        byte_index: ByteIndex,
        seed: Checksum,
        amount: usize,
    ) -> EvaluationResult<Vec<ContextEmbedding>> {
        let main_embedding = self.get_embedding_aggregate(byte_index)?;
        let mut rng = StdRng::from_seed(seed.into());

        // will return less if the referenced by contains fewer than context references
        let referencing_bytes: Vec<&ByteIndex> = main_embedding
            .referenced_by
            .choose_multiple(&mut rng, amount)
            .collect();

        let mut context_embeddings = Vec::new();
        for rb in referencing_bytes {
            let reference_embedding = self.get_embedding_aggregate(*rb)?;
            context_embeddings.push(ContextEmbedding {
                byte_index: *rb,
                num_bytes_represented: reference_embedding.num_bytes_represented,
                num_bytes_referenced: reference_embedding.num_bytes_referenced,
                num_bytes_used: reference_embedding.num_bytes_used,
                embedding: self.get_embedding(byte_index)?,
            });
        }

        Ok(context_embeddings)
    }

    fn get_embedding_aggregate(
        &self,
        byte_index: ByteIndex,
    ) -> EvaluationResult<&EmbeddingAggregate> {
        let embedding_index = self
            .byte_to_embedding
            .get(&byte_index)
            .ok_or(EvaluationError::SafeTensorsFailure("t".to_string()))?;
        let embedding_aggregate = self
            .embedding_aggregates
            .get(embedding_index)
            .ok_or(EvaluationError::SafeTensorsFailure("t".to_string()))?;
        Ok(embedding_aggregate)
    }
    fn get_embedding(&self, byte_index: ByteIndex) -> EvaluationResult<ArrayD<f32>> {
        let embedding_index = self
            .byte_to_embedding
            .get(&byte_index)
            .ok_or(EvaluationError::SafeTensorsFailure("t".to_string()))?;
        let ek = embedding_key(embedding_index);
        let embedding = parse_tensor_view_data::<f32>(
            &self
                .safetensors
                .tensor(&ek)
                .map_err(|e| EvaluationError::SafeTensorsFailure(e.to_string()))?,
        )
        .map_err(|e| EvaluationError::SafeTensorsFailure(e.to_string()))?;
        Ok(embedding)
    }
}

#[derive(Debug)]
struct EmbeddingNode {
    bytes_used: usize,
    bytes_represented: HashSet<ByteIndex>,
    outgoing_edges: HashSet<EmbeddingIndex>,
}

#[derive(Debug)]
struct EmbeddingGraph {
    nodes: HashMap<EmbeddingIndex, EmbeddingNode>,
    byte_to_embedding: HashMap<ByteIndex, EmbeddingIndex>,
    max_byte_index: u64,
}

impl EmbeddingGraph {
    fn new(safetensors: &SafeTensors) -> EvaluationResult<EmbeddingGraph> {
        let num_embedding_tensors = safetensors.names().len() - AUXILLARY_TENSORS;
        if num_embedding_tensors % TENSORS_PER_EMBEDDING != 0 {
            return Err(EvaluationError::SafeTensorsFailure(
                "invalid tensor number".to_string(),
            ));
        }
        let num_embeddings = (num_embedding_tensors / TENSORS_PER_EMBEDDING) as u64;

        let mut nodes = HashMap::with_capacity(num_embeddings as usize);
        let mut byte_to_embedding = HashMap::new();
        let mut max_byte_index = 0;

        for embedding_index in MIN_EMBEDDING_INDEX..=num_embeddings {
            let embedding_index = EmbeddingIndex(embedding_index);
            let ek = embedding_key(&embedding_index);
            let bk = byte_key(&ek);
            let rk = reference_key(&ek);

            // Parse embedding tensor
            let embedding = safetensors
                .tensor(&ek)
                .map_err(|e| EvaluationError::SafeTensorsFailure(e.to_string()))?;
            let bytes_used = embedding.shape()[0] * embedding.dtype().size();

            let byte_ranges = parse_tensor_view_data::<u64>(
                &safetensors
                    .tensor(&bk)
                    .map_err(|e| EvaluationError::SafeTensorsFailure(e.to_string()))?,
            )
            .map_err(|e| EvaluationError::SafeTensorsFailure(e.to_string()))?;

            // TODO: convert ranges to bytes to just return a hashset
            let bytes_represented = ranges_to_bytes(&byte_ranges)?
                .into_iter()
                .map(|i| ByteIndex(i))
                .collect::<HashSet<_>>();

            // Validate no byte overlaps
            for &byte_idx in &bytes_represented {
                if byte_to_embedding
                    .insert(byte_idx, embedding_index)
                    .is_some()
                {
                    return Err(EvaluationError::SafeTensorsFailure(format!(
                        "Byte overlap detected at index {}",
                        byte_idx.0
                    )));
                }
                max_byte_index = max_byte_index.max(byte_idx.0);
            }

            let reference_indices = parse_tensor_view_data::<u64>(
                &safetensors
                    .tensor(&rk)
                    .map_err(|e| EvaluationError::SafeTensorsFailure(e.to_string()))?,
            )
            .map_err(|e| EvaluationError::SafeTensorsFailure(e.to_string()))?;
            let outgoing_edges = reference_indices
                .into_iter()
                .map(|r| {
                    if r == embedding_index.0 {
                        return Err(EvaluationError::SafeTensorsFailure(
                            "Self-referential embedding detected".into(),
                        ));
                    }
                    // TODO double check whether this is right
                    if r < 1 || r > num_embeddings {
                        return Err(EvaluationError::SafeTensorsFailure(
                            "Invalid embedding reference index".into(),
                        ));
                    }
                    Ok(EmbeddingIndex(r))
                })
                .collect::<EvaluationResult<HashSet<EmbeddingIndex>>>()?;

            nodes.insert(
                embedding_index,
                EmbeddingNode {
                    bytes_used,
                    bytes_represented,
                    outgoing_edges,
                },
            );
        }

        let graph = EmbeddingGraph {
            nodes,
            byte_to_embedding,
            max_byte_index,
        };

        graph.validate()?;
        Ok(graph)
    }
    fn validate(&self) -> EvaluationResult<()> {
        // Validate byte coverage
        for byte_idx in 0..=self.max_byte_index {
            let byte_index = ByteIndex(byte_idx);
            if !self.byte_to_embedding.contains_key(&byte_index) {
                return Err(EvaluationError::SafeTensorsFailure(format!(
                    "Missing byte index {} in byte mappings",
                    byte_idx
                )));
            }
        }

        // Validate embedding utility and byte connectivity
        for (idx, node) in &self.nodes {
            // Check non-zero bytes represented
            if node.bytes_represented.is_empty() {
                return Err(EvaluationError::SafeTensorsFailure(format!(
                    "Embedding {} has zero bytes represented",
                    idx.0
                )));
            }

            // Check if embedding is useful (has incoming or outgoing edges)
            let has_outgoing = !node.outgoing_edges.is_empty();
            let has_incoming = self
                .nodes
                .iter()
                .any(|(_, n)| n.outgoing_edges.contains(idx));
            if !has_outgoing && !has_incoming {
                return Err(EvaluationError::SafeTensorsFailure(format!(
                    "Embedding {} is redundant (no incoming or outgoing edges)",
                    idx.0
                )));
            }

            // Check byte connectivity
            for &byte_idx in &node.bytes_represented {
                let has_incoming = self.nodes.iter().any(|(_, n)| {
                    n.outgoing_edges.iter().any(|&ref_idx| {
                        self.nodes
                            .get(&ref_idx)
                            .map_or(false, |rn| rn.bytes_represented.contains(&byte_idx))
                    })
                });
                if !has_incoming {
                    return Err(EvaluationError::SafeTensorsFailure(format!(
                        "Byte {} has no incoming edges",
                        byte_idx.0
                    )));
                }
            }
        }

        Ok(())
    }
    fn aggregate(&self) -> HashMap<EmbeddingIndex, EmbeddingAggregate> {
        let mut embedding_aggregates = HashMap::with_capacity(self.nodes.len());

        for (&idx, node) in &self.nodes {
            let num_bytes_referenced = node
                .outgoing_edges
                .iter()
                .flat_map(|&ref_idx| self.nodes.get(&ref_idx).map(|n| n.bytes_represented.iter()))
                .count();

            embedding_aggregates.insert(
                idx,
                EmbeddingAggregate {
                    num_bytes_represented: node.bytes_represented.len(),
                    num_bytes_referenced,
                    num_bytes_used: node.bytes_used,
                    referenced_by: Vec::new(), // To be populated later
                },
            );
        }

        // Compute referenced_by for each embedding
        for (_source_idx, source_node) in &self.nodes {
            for &ref_idx in &source_node.outgoing_edges {
                if let Some(target_agg) = embedding_aggregates.get_mut(&ref_idx) {
                    target_agg
                        .referenced_by
                        .extend(source_node.bytes_represented.iter().copied());
                }
            }
        }

        // Sort referenced_by for consistency
        for agg in embedding_aggregates.values_mut() {
            agg.referenced_by.sort_by_key(|b| b.0);
        }

        embedding_aggregates
    }
}

// Utility functions
fn embedding_key(embedding_index: &EmbeddingIndex) -> String {
    format!("{:?}", embedding_index)
}

fn byte_key(embedding_key: &str) -> String {
    format!("b{}", embedding_key)
}

fn reference_key(embedding_key: &str) -> String {
    format!("r{}", embedding_key)
}

fn ranges_to_bytes(ranges: &ArrayD<u64>) -> EvaluationResult<Vec<u64>> {
    if ranges.ndim() != 2 || ranges.shape()[1] != 2 {
        return Err(EvaluationError::SafeTensorsFailure(
            "Input array must have shape [N, 2] (start, end)".to_string(),
        ));
    }

    let mut bytes = HashSet::new();
    for row in ranges.axis_iter(Axis(0)) {
        let start = row[0];
        let end = row[1];
        if start > end {
            return Err(EvaluationError::SafeTensorsFailure(format!(
                "Invalid range: start {} exceeds end {}",
                start, end
            )));
        }
        for i in start..=end {
            bytes.insert(i);
        }
    }

    let mut result: Vec<u64> = bytes.into_iter().collect();
    result.sort();
    Ok(result)
}
