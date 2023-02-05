use std::iter::Map;
use crate::enums::{Feature, TrueFalseLabel};
use crate::enums::TrueFalseLabel::False;

pub struct NewsArticle {
    article_id: i32,

    title: String,
    sentences: Vec<String>,

    embedding_title: Vec<f64>,
    embedding_sentences: Vec<Vec<f64>>,

    semantic_features: Map<Feature, f64>,

    label: Option<String>,
}

pub struct EmbeddingElement {
    pub article_id: i32,
    pub sentence_id: i32,
    pub embedding: Vec<f64>,
    pub label: TrueFalseLabel,
}

pub trait SimpleDistance {
    fn euclidean_distance(&self, other: &EmbeddingElement) -> f64;
    fn is_within_box_distance_or_whatever_it_is_called(&self, other: &EmbeddingElement, box_distance: &f64) -> bool;
}


impl SimpleDistance for EmbeddingElement {
    fn euclidean_distance(&self, other: &EmbeddingElement) -> f64 {
        let mut roll_sum: f64 = 0.0;
        for i in 0..self.embedding.len() {
            roll_sum += (self.embedding.get(i).unwrap() - other.embedding.get(i).unwrap()).powi(2);
        }
        return roll_sum.sqrt();
    }

    fn is_within_box_distance_or_whatever_it_is_called(&self, other: &EmbeddingElement, box_distance: &f64) -> bool {
        for i in 0..self.embedding.len() {
            let diff: f64 = self.embedding.get(i).unwrap() - other.embedding.get(i).unwrap();
            if diff.abs() > *box_distance {
                return false;
            }
        }
        return true;
    }
}

