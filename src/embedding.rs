pub type Embedding = Vec<f32>;

/// Normalize magnitudes
pub fn l2_normalize(embeddings: &[Embedding]) -> Result<Vec<Embedding>, &'static str> {
    let Some(first) = embeddings.first() else {
        return Ok(Vec::new());
    };
    let dimensions = first.len();
    let normalized: Vec<Vec<f32>> = embeddings
        .iter()
        .map(|embedding| {
            if dimensions == 0 || embedding.len() != dimensions {
                return Err("embeddings must have the same nonzero dimension");
            }
            let norm = embedding
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            if !norm.is_finite() || norm == 0.0 {
                return Err("embeddings must be finite and nonzero");
            }
            Ok(embedding.iter().map(|value| value / norm).collect())
        })
        .collect::<Result<_, _>>()?;

    Ok(normalized)
}

// Sim
pub fn cosine_similarity(left: &[Embedding], right: &[Embedding]) -> f32 {
    let sum = |context: &[Embedding]| -> Embedding {
        (0..context[0].len())
            .map(|dimension| context.iter().map(|embedding| embedding[dimension]).sum())
            .collect()
    };

    let left = sum(left);
    let right = sum(right);
    let norm = (left.iter().map(|value| value * value).sum::<f32>()
        * right.iter().map(|value| value * value).sum::<f32>())
    .sqrt();
    if norm == 0.0 {
        return 0.0;
    }
    left.iter().zip(&right).map(|(a, b)| a * b).sum::<f32>() / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_similarity() {
        assert_eq!(
            cosine_similarity(&[vec![1.0], vec![-1.0]], &[vec![1.0], vec![1.0]]),
            0.0
        );
    }
}
