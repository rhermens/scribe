use crate::{Segment, Window};

/// Groups whole Whisper segments at sentence endings, or after roughly 60 words.
/// Segment boundaries preserve timestamps; a segment containing several sentences stays intact.
pub fn sentence_windows(segments: &[Segment]) -> Vec<Window> {
    let mut groups = Vec::new();
    let mut start = 0;
    let mut words = 0;

    for (index, segment) in segments.iter().enumerate() {
        words += segment.text.split_whitespace().count();
        let ending = segment
            .text
            .trim_end()
            .trim_end_matches(['"', '\'', '”', '’', ')', ']']);
        if words > 0 && (ending.ends_with(['.', '!', '?']) || words >= 60) {
            groups.push(Window {
                segments: start..index + 1,
                text: segment_text(&segments[start..index + 1]),
            });
            start = index + 1;
            words = 0;
        }
    }

    if words > 0 {
        groups.push(Window {
            segments: start..segments.len(),
            text: segment_text(&segments[start..]),
        });
    } else if let Some(last) = groups.last_mut() {
        last.segments.end = segments.len();
    }
    groups
}

fn segment_text(segments: &[Segment]) -> String {
    segments
        .iter()
        .map(|segment| segment.text.as_str())
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Finds subject boundaries at distinct valleys in multi-sentence context similarity.
/// Compares up to eight sentence windows on each side, not isolated sentences.
/// Higher thresholds admit more boundaries. Each resulting paragraph has at least
/// `min_words`, unless the entire transcript is shorter. No split is forced by length.
/// Embeddings must be finite, nonzero, equally sized, and aligned with `groups`.
pub fn topic_paragraphs(
    groups: &[Window],
    embeddings: &[Vec<f32>],
    threshold: f32,
    min_words: usize,
) -> Result<Vec<Window>, &'static str> {
    if !threshold.is_finite() || !(-1.0..=1.0).contains(&threshold) {
        return Err("topic threshold must be between -1 and 1");
    }
    if min_words == 0 {
        return Err("minimum paragraph word count must be positive");
    }
    if groups.len() != embeddings.len() {
        return Err("expected one embedding per sentence window");
    }
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

    if groups.len() == 1 {
        return Ok(vec![paragraph(groups)]);
    }

    // Full context on both sides avoids unreliable cuts near the transcript's ends.
    let context = 8.min(groups.len() / 2);
    let scores: Vec<f32> = (context..=groups.len() - context)
        .map(|index| {
            context_similarity(
                &normalized[index - context..index],
                &normalized[index..index + context],
            )
        })
        .collect();
    let radius = (context / 2).max(1);
    let mut candidates: Vec<usize> = (0..scores.len())
        .filter(|&index| {
            let score = scores[index];
            let left = &scores[index.saturating_sub(radius)..index];
            let right = &scores[index + 1..(index + radius + 1).min(scores.len())];
            let left_peak = left.iter().copied().fold(score, f32::max);
            let right_peak = right.iter().copied().fold(score, f32::max);
            // A flat run of low similarity is not evidence of a subject boundary.
            let prominence = (left_peak + right_peak) / 2.0 - score;
            score < threshold
                && left.iter().all(|&neighbor| neighbor > score)
                && right.iter().all(|&neighbor| neighbor >= score)
                && (scores.len() == 1 || prominence >= 0.05)
        })
        .collect();
    // Prefer the strongest subject boundary when nearby candidates compete.
    candidates.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(a.cmp(&b)));
    let mut words = vec![0_usize];
    for group in groups {
        words.push(words.last().unwrap() + group.text.split_whitespace().count());
    }
    let mut boundaries = vec![0, groups.len()];
    for candidate in candidates {
        let boundary = candidate + context;
        if boundaries
            .iter()
            .all(|&other| words[boundary].abs_diff(words[other]) >= min_words)
        {
            boundaries.push(boundary);
        }
    }
    boundaries.sort_unstable();
    Ok(boundaries
        .windows(2)
        .map(|range| paragraph(&groups[range[0]..range[1]]))
        .collect())
}

fn context_similarity(left: &[Vec<f32>], right: &[Vec<f32>]) -> f32 {
    let sum = |context: &[Vec<f32>]| -> Vec<f32> {
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

fn paragraph(groups: &[Window]) -> Window {
    Window {
        segments: groups[0].segments.start..groups[groups.len() - 1].segments.end,
        text: groups
            .iter()
            .map(|group| group.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments(texts: &[&str]) -> Vec<Segment> {
        texts
            .iter()
            .enumerate()
            .map(|(id, text)| Segment {
                id,
                start: id as i64 * 100,
                end: (id as i64 + 1) * 100,
                text: (*text).to_owned(),
            })
            .collect()
    }

    fn groups(texts: &[&str]) -> Vec<Window> {
        sentence_windows(&segments(texts))
    }

    fn ranges(paragraphs: &[Window]) -> Vec<std::ops::Range<usize>> {
        paragraphs
            .iter()
            .map(|paragraph| paragraph.segments.clone())
            .collect()
    }

    #[test]
    fn sentence_groups_preserve_text_order_and_segment_ranges() {
        let input = segments(&[
            "",
            "We discuss",
            "gardening.\"",
            "Then cooking!",
            "An unfinished thought",
            "",
        ]);
        let windows = sentence_windows(&input);
        assert_eq!(ranges(&windows), vec![0..3, 3..4, 4..6]);
        assert_eq!(windows[0].text, "We discuss gardening.\"");
        assert_eq!(windows[2].text, "An unfinished thought");
    }

    #[test]
    fn sentence_groups_bound_unpunctuated_text() {
        let text = "word ".repeat(30);
        let input = segments(&[&text, &text, "tail"]);
        assert_eq!(ranges(&sentence_windows(&input)), vec![0..2, 2..3]);
    }

    #[test]
    fn empty_and_blank_transcripts_have_no_groups() {
        assert!(sentence_windows(&[]).is_empty());
        assert!(groups(&[" ", ""]).is_empty());
        assert!(topic_paragraphs(&[], &[], 0.6, 1).unwrap().is_empty());
    }

    #[test]
    fn a_brief_question_does_not_split_the_surrounding_subject() {
        let input = groups(&["Garden soil.", "What do you mean?", "Garden plants."]);
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]];
        assert_eq!(
            ranges(&topic_paragraphs(&input, &embeddings, 0.6, 1).unwrap()),
            vec![0..3]
        );
    }

    #[test]
    fn distinct_subjects_form_separate_paragraphs() {
        let input = groups(&[
            "Garden plants.",
            "Garden soil.",
            "Cooking pasta.",
            "Cooking sauce.",
        ]);
        let embeddings = vec![
            vec![2.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 3.0],
            vec![0.0, 1.0],
        ];
        let output = topic_paragraphs(&input, &embeddings, 0.6, 1).unwrap();
        assert_eq!(ranges(&output), vec![0..2, 2..4]);
        assert_eq!(output[0].text, "Garden plants. Garden soil.");
        assert_eq!(output[1].text, "Cooking pasta. Cooking sauce.");
    }

    #[test]
    fn different_aspects_of_each_subject_stay_grouped() {
        let texts: Vec<String> = (0..32).map(|index| format!("Sentence {index}.")).collect();
        let input = groups(&texts.iter().map(String::as_str).collect::<Vec<_>>());
        // Individual sentences share only 0.5 cosine similarity within a subject.
        // Their eight-sentence contexts still clearly separate the two subjects.
        let embeddings: Vec<Vec<f32>> = (0..32)
            .map(|index| {
                let mut embedding = vec![0.0; 34];
                embedding[usize::from(index >= 16)] = 1.0;
                embedding[index + 2] = 1.0;
                embedding
            })
            .collect();
        let output = topic_paragraphs(&input, &embeddings, 0.6, 1).unwrap();
        assert_eq!(ranges(&output), vec![0..16, 16..32]);
        assert_eq!(
            output
                .iter()
                .map(|paragraph| paragraph.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            texts.join(" ")
        );
    }

    #[test]
    fn minimum_words_also_prevents_a_tiny_final_paragraph() {
        let input = groups(&["One two.", "Three four.", "Five."]);
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        assert_eq!(
            ranges(&topic_paragraphs(&input, &embeddings, 0.6, 4).unwrap()),
            vec![0..3]
        );
    }

    #[test]
    fn gradual_sentence_variation_without_a_valley_stays_together() {
        let input = groups(&["First.", "Second.", "Third."]);
        let embeddings = vec![vec![1.0, 0.0], vec![0.8, 0.6], vec![0.28, 0.96]];
        assert_eq!(
            ranges(&topic_paragraphs(&input, &embeddings, 0.7, 1).unwrap()),
            vec![0..3]
        );
    }

    #[test]
    fn higher_threshold_splits_more_often() {
        let input = groups(&["First.", "Second."]);
        let embeddings = vec![vec![1.0, 0.0], vec![0.6, 0.8]];
        assert_eq!(
            topic_paragraphs(&input, &embeddings, 0.5, 1).unwrap().len(),
            1
        );
        assert_eq!(
            topic_paragraphs(&input, &embeddings, 0.7, 1).unwrap().len(),
            2
        );
    }

    #[test]
    fn equality_does_not_split_and_cancelling_contexts_are_defined() {
        let input = groups(&["First.", "Second."]);
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(
            topic_paragraphs(&input, &embeddings, 0.0, 1).unwrap().len(),
            1
        );

        assert_eq!(
            context_similarity(&[vec![1.0], vec![-1.0]], &[vec![1.0], vec![1.0]]),
            0.0
        );
    }

    #[test]
    fn one_sentence_and_trailing_blanks_preserve_range() {
        let input = groups(&["Only sentence.", ""]);
        let output = topic_paragraphs(&input, &[vec![1.0]], 0.6, 40).unwrap();
        assert_eq!(ranges(&output), vec![0..2]);
        assert_eq!(output[0].text, "Only sentence.");
    }

    #[test]
    fn invalid_embeddings_and_options_are_rejected() {
        let input = groups(&["First.", "Second."]);
        for embeddings in [
            vec![],
            vec![vec![], vec![]],
            vec![vec![1.0], vec![1.0, 2.0]],
            vec![vec![0.0], vec![1.0]],
            vec![vec![f32::NAN], vec![1.0]],
            vec![vec![f32::INFINITY], vec![1.0]],
        ] {
            assert!(topic_paragraphs(&input, &embeddings, 0.6, 1).is_err());
        }
        let embeddings = vec![vec![1.0], vec![1.0]];
        for threshold in [f32::NAN, f32::INFINITY, -1.1, 1.1] {
            assert!(topic_paragraphs(&input, &embeddings, threshold, 1).is_err());
        }
        assert!(topic_paragraphs(&input, &embeddings, 0.6, 0).is_err());
    }
}
