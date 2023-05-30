use bytes::Buf;
use json::object;
use std::collections::{HashMap, HashSet};
use std::io;
use std::io::Write;
use std::iter::Map;
use std::ops::{Range, RangeInclusive};

use crate::representation::antibody::Antibody;
use crate::representation::antigen::AntiGen;
use rand::prelude::{IteratorRandom, SliceRandom};
use rand::Rng;
use rayon::prelude::*;
use statrs::statistics::Statistics;
use strum_macros::{Display, EnumString};

// pub struct ClassPrediction {
//     pub class: usize,
//     pub reg_count: usize,
//     pub valid_count: usize,
//     pub membership_sum: f64,
//     pub valid_membership_sum: f64,
// }
//
//
// pub struct Prediction {
//     pub predicted_class: usize,
//     pub class_predictions: Vec<ClassPrediction>,
// }

pub enum Prediction {
    ClassPredict(usize),
    NoPredict,
}

#[derive(Copy, Clone, Display,EnumString, Hash, Eq, PartialEq, Debug)]
pub enum EvaluationMethod {
    Count,
    Fraction,
    AffinitySum,
}

fn count_prediction(antigen: &AntiGen, registered_antibodies: Vec<&Antibody>) -> Prediction {
    if registered_antibodies.len() == 0 {
        return Prediction::NoPredict;
    }

    let labels: HashSet<usize> = registered_antibodies
        .iter()
        .map(|ab| ab.class_label)
        .collect();

    let mut max_count = 0;
    let mut max_count_label = 0;
    for label in labels {
        let label_count = registered_antibodies
            .iter()
            .filter(|ab| ab.class_label == label)
            .count();

        if label_count >= max_count {
            max_count_label = label.clone();
            max_count = label_count;
        }
    }

    return Prediction::ClassPredict(max_count_label);
}

fn fraction_prediction(antigen: &AntiGen, registered_antibodies: Vec<&Antibody>) -> Prediction {
    if registered_antibodies.len() == 0 {
        return Prediction::NoPredict;
    }

    let labels: HashSet<usize> = registered_antibodies
        .iter()
        .map(|ab| ab.class_label)
        .collect();

    let mut max_sum = 0.0;
    let mut max_sum_label = 0;
    for label in labels {
        let label_sum = registered_antibodies
            .iter()
            .filter(|ab| ab.class_label == label)
            .map(|ab| ab.final_train_label_membership.unwrap().0)
            .sum();

        if label_sum >= max_sum {
            max_sum_label = label.clone();
            max_sum = label_sum;
        }
    }

    return Prediction::ClassPredict(max_sum_label);
}




fn affinity_prediction(antigen: &AntiGen, registered_antibodies: Vec<&Antibody>) -> Prediction {
    if registered_antibodies.len() == 0 {
        return Prediction::NoPredict;
    }

    let labels: HashSet<usize> = registered_antibodies
        .iter()
        .map(|ab| ab.class_label)
        .collect();

    let mut max_sum = 0.0;
    let mut max_sum_label = 0;
    for label in labels {
        let label_sum = registered_antibodies
            .iter()
            .filter(|ab| ab.class_label == label)
            .map(|ab| ab.final_train_label_affinity.unwrap().0)
            .sum();

        if label_sum >= max_sum {
            max_sum_label = label.clone();
            max_sum = label_sum;
        }
    }

    return Prediction::ClassPredict(max_sum_label);
}


fn weighted_prediction(antigen: &AntiGen, registered_antibodies: Vec<&Antibody>, eval_method: &EvaluationMethod) -> Prediction {
    if registered_antibodies.len() == 0 {
        return Prediction::NoPredict;
    }

    let labels: HashSet<usize> = registered_antibodies
        .iter()
        .map(|ab| ab.class_label)
        .collect();

    let mut max_sum = 0.0;
    let mut max_sum_label = 0;


    let mut predicate = match eval_method {
        EvaluationMethod::Count => {|ab: &Antibody|-> f64 {1.0}}
        EvaluationMethod::Fraction => {|ab: &Antibody|-> f64 {ab.final_train_label_membership.unwrap().0}}
        EvaluationMethod::AffinitySum => {|ab: &Antibody|-> f64 {ab.final_train_label_affinity.unwrap().0}}
    };

    for label in labels {
        let label_sum = registered_antibodies
            .iter()
            .filter(|ab| ab.class_label == label)
            .map(|ab| predicate(ab) * ab.boosting_model_alpha)
            .sum();

        if label_sum >= max_sum {
            max_sum_label = label.clone();
            max_sum = label_sum;
        }
    }

    return Prediction::ClassPredict(max_sum_label);
}

pub fn make_prediction(
    antigen: &AntiGen,
    antibodies: &Vec<Antibody>,
    eval_method: &EvaluationMethod,
) -> Prediction {

    let registered_antibodies = antibodies
        .par_iter()
        .filter(|antibody| antibody.test_antigen(antigen))
        .collect::<Vec<_>>();


    return weighted_prediction(antigen,registered_antibodies,eval_method)
    // return match eval_method {
    //     EvaluationMethod::Count => count_prediction(antigen, registered_antibodies),
    //     EvaluationMethod::Fraction => fraction_prediction(antigen, registered_antibodies),
    //     EvaluationMethod::AffinitySum => affinity_prediction(antigen, registered_antibodies),
    // };
}
pub fn is_class_correct(
    antigen: &AntiGen,
    antibodies: &Vec<Antibody>,
    eval_method: &EvaluationMethod,
) -> Option<bool> {
    let pred = make_prediction(antigen, antibodies, eval_method);

    return if let Prediction::ClassPredict(label) = pred {
        Some(label == antigen.class_label)
    } else {
        None
    };
}
//
// pub fn is_class_correct_with_type(antigen: &AntiGen, antibodies: &Vec<Antibody>, eval_method: EvaluationMethod) -> Option<bool> {
//
//
// }
//
//
// pub fn is_class_correct(antigen: &AntiGen, antibodies: &Vec<Antibody>) -> Option<bool> {
//     let matching_cells = antibodies
//         .par_iter()
//         .filter(|antibody| antibody.test_antigen(antigen))
//         .collect::<Vec<_>>();
//
//     if matching_cells.len() == 0 {
//         return None;
//     }
//
//     let class_true = matching_cells
//         .iter()
//         .filter(|x| x.class_label == antigen.class_label)
//         .collect::<Vec<_>>();
//     let class_false = matching_cells
//         .iter()
//         .filter(|x| x.class_label != antigen.class_label)
//         .collect::<Vec<_>>();
//
//     if class_true.len() > class_false.len() {
//         return Some(true);
//     } else {
//         // println!("wrong match id {:?}, cor: {:?}  incor {:?}", antigen.id, class_true.len(), class_false.len());
//         return Some(false);
//     }
//
// }
//
// pub fn make_prediction(antigen: &AntiGen, antibodies: &Vec<Antibody>) -> Option<Prediction> {
//     let matching_cells = antibodies
//         .par_iter()
//         .filter(|antibody| antibody.test_antigen(antigen))
//         .collect::<Vec<_>>();
//
//     let matching_classes = matching_cells
//         .iter()
//         .map(|ab| ab.class_label)
//         .collect::<HashSet<usize>>();
//
//     let mut label_membership_values = matching_classes
//         .iter()
//         .map(|class| {
//             (
//                 class.clone(),
//                 ClassPrediction {
//                     class: class.clone(),
//                     reg_count: 0,
//                     membership_sum: 0.0,
//                     valid_count: 0,
//                     valid_membership_sum: 0.0,
//                 },
//             )
//         })
//         .collect::<HashMap<usize, ClassPrediction>>();
//
//     for cell in matching_cells {
//         let mut match_class_pred = label_membership_values.get_mut(&cell.class_label).unwrap();
//         match_class_pred.membership_sum += cell.final_train_label_membership.unwrap().0;
//         match_class_pred.reg_count += 1;
//
//         // if cell.final_train_label_membership.unwrap().0 >= 0.95{
//         if cell.final_train_label_membership.unwrap().0 >= 0.0 {
//             match_class_pred.valid_count += 1;
//             // match_class_pred.valid_membership_sum += cell.final_train_label_membership.unwrap().0 * cell.boosting_model_alpha;
//             match_class_pred.valid_membership_sum += 1.0 * cell.boosting_model_alpha;
//         }
//     }
//
//     let mut class_preds = Vec::new();
//
//     let mut best_class = None;
//     let mut best_score = 0.0;
//     for (label, label_membership_value) in label_membership_values {
//         if best_score < label_membership_value.valid_membership_sum {
//             best_class = Some(label);
//             best_score = label_membership_value.valid_membership_sum.clone()
//         }
//         class_preds.push(label_membership_value)
//     }
//
//     return Some(Prediction {
//         predicted_class: best_class?,
//         class_predictions: class_preds,
//     });
// }
//
// pub fn is_class_correct_with_membership(
//     antigen: &AntiGen,
//     antibodies: &Vec<Antibody>,
// ) -> Option<bool> {
//     let pred_value = make_prediction(antigen, antibodies)?;
//
//     if pred_value.predicted_class == antigen.class_label {
//         return Some(true);
//     } else {
//         // println!("wrong match id {:?}, cor: {:?}  incor {:?}", antigen.id, class_true.len(), class_false.len());
//         return Some(false);
//     }
// }
