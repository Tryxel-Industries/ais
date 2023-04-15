use std::cmp::min;
use std::collections::HashMap;

use rayon::prelude::*;

use crate::bucket_empire::{BucketKing, ValueRangeType};
use crate::representation::antibody::{Antibody, DimValueType};
use crate::representation::antigen::AntiGen;

#[derive(Clone, Debug)]
pub struct Evaluation {
    pub matched_ids: Vec<usize>,
    pub wrongly_matched: Vec<usize>,
}


pub fn evaluate_antibody(
    antigens: &Vec<AntiGen>,
    antibody: &Antibody,
) -> Evaluation {
    /*
    //todo: this is a mess that does not work if any of the dims are open, some workarounds are possible but probably firmly overoptimizing
    let dim_radus = antibody
        .dim_values
        .iter()
        .map(|dv| {
            return match dv.value_type {
                DimValueType::Disabled => ValueRangeType::Open,
                DimValueType::Open => ValueRangeType::Open,

                DimValueType::Circle => {
                    let value = antibody.radius_constant.sqrt() / dv.multiplier;
                    let flipped_offset = dv.offset;
                    return ValueRangeType::Symmetric((
                        flipped_offset - value,
                        flipped_offset + value,
                    ));
                }
            };
        })
        .collect();

    let mut idx_list = bk
        .get_potential_matches(&dim_radus)
        .unwrap();

    idx_list.sort();*/

    let registered_antigens = antigens
        .iter()
        // .filter(|ag| idx_list.binary_search(&ag.id).is_ok())
        .filter(|ag| antibody.test_antigen(ag))
        .collect::<Vec<_>>();

    if false {
        let test_a = antigens
            .iter()
            // .filter(|ag| idx_list.binary_search(&ag.id).is_ok())
            .filter(|ag| antibody.test_antigen(ag))
            .collect::<Vec<_>>();

        let test_b = antigens
            .iter()
            .filter(|ag| antibody.test_antigen(ag))
            .collect::<Vec<_>>();
        if test_a.len() != test_b.len() {
            println!("{:?} vs {:?}", test_a.len(), test_b.len());
            println!();
            println!(
                "cell vt    {:?}",
                antibody
                    .dim_values
                    .iter()
                    .map(|b| b.value_type.clone())
                    .collect::<Vec<_>>()
            );
            println!(
                "cell mp    {:?}",
                antibody
                    .dim_values
                    .iter()
                    .map(|b| b.multiplier.clone())
                    .collect::<Vec<_>>()
            );
            println!(
                "cell of    {:?}",
                antibody
                    .dim_values
                    .iter()
                    .map(|b| b.offset.clone())
                    .collect::<Vec<_>>()
            );
            println!("cell rad : {:?}", antibody.radius_constant);
            println!();
            // println!("dim rad: {:?}", dim_radus);
            println!("bk  res: {:?}", test_a);
            println!();
            println!("otr res: {:?}", test_b);
            panic!("bucket empire error")
        }

        // println!("a {:?} b {:?}", test_a.len(), test_b.len());
    }

    let mut corr_matched = Vec::with_capacity(registered_antigens.len());
    let mut wrong_matched = Vec::with_capacity(registered_antigens.len());

    for registered_antigen in registered_antigens {
        if registered_antigen.class_label == antibody.class_label {
            corr_matched.push(registered_antigen.id)
        } else {
            wrong_matched.push(registered_antigen.id)
        }
    }

    // let with_same_label = registered_antigens.iter().filter(|ag|ag.class_label==antibody.class_label ).collect::<Vec<_>>();
    // let num_wrong = registered_antigens.iter().filter(|ag|ag.class_label!=antibody.class_label ).collect::<Vec<_>>();
    // let num_wrong = registered_antigens.len()-with_same_label.len();

    // let score = (with_same_label.len()as f64) - ((num_wrong as f64/2.0));

    // println!("matched {:?}\n", corr_matched);
    let ret_evaluation = Evaluation {
        matched_ids: corr_matched,
        wrongly_matched: wrong_matched,
    };
    // println!("num reg {:?} same label {:?} other label {:?}",antigens.len(), with_same_label.len(), num_wrong);
    return ret_evaluation;
}

//
//  Match counter
//
#[derive(Clone, Debug)]
pub struct MatchCounter {
    max_id: usize,
    pub correct_match_counter: Vec<usize>,
    pub incorrect_match_counter: Vec<usize>,
}

impl MatchCounter {
    pub fn new(max_id: usize) -> MatchCounter{
        return MatchCounter{
            max_id,
            correct_match_counter: vec![0usize; max_id+1],
            incorrect_match_counter: vec![0usize; max_id+1]
        };

    }

    pub fn get_inversed_correct_match_counts(&self) -> Vec<usize>{
        let max_val = self.correct_match_counter.iter().max().unwrap();
        let mut inverse = self.correct_match_counter.clone();
        inverse = inverse.iter().map(|v| max_val - v).collect();
        return inverse;
    }

    pub fn get_inversed_incorrect_match_counts(&self) -> Vec<usize>{
        let max_val = self.incorrect_match_counter.iter().max().unwrap();
        let mut inverse = self.incorrect_match_counter.clone();
        inverse = inverse.iter().map(|v| max_val - v).collect();
        return inverse;
    }

    pub fn add_evaluations(&mut self, evaluations: Vec<&Evaluation> ){
        for evaluation in evaluations{
            for correct_match_id in &evaluation.matched_ids{
                if let Some(elem) = self.correct_match_counter.get_mut(*correct_match_id) {
                    *elem += 1;
                }else {
                    println!("match id {:?} arr len {:?}", correct_match_id, self.correct_match_counter.len());
                    panic!("match count error")
                }
            }

            for correct_match_id in &evaluation.wrongly_matched{
                if let Some(elem) = self.incorrect_match_counter.get_mut(*correct_match_id) {
                    *elem += 1;
                }else {
                    panic!("match count error")
                }
            }
        }

    }

    pub fn remove_evaluations(&mut self, evaluations: Vec<&Evaluation> ){
        for evaluation in evaluations{
            for correct_match_id in &evaluation.matched_ids{
                if let Some(elem) = self.correct_match_counter.get_mut(*correct_match_id) {
                    *elem -= 1;
                }else {
                    panic!("match count error")
                }
            }

            for correct_match_id in &evaluation.wrongly_matched{
                if let Some(elem) = self.incorrect_match_counter.get_mut(*correct_match_id) {
                    *elem -= 1;
                }else {
                    panic!("match count error")
                }
            }
        }

    }
}
