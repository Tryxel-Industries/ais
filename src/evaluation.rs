use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use rayon::prelude::*;

use crate::bucket_empire::{BucketKing, ValueRangeType};
use crate::params::Params;
use crate::representation::antibody::{Antibody, DimValueType};
use crate::representation::antigen::{AntiGen, AntiGenPop};

#[derive(Clone, Debug)]
pub struct Evaluation {
    pub matched_ids: Vec<usize>,
    pub wrongly_matched: Vec<usize>,
    pub membership_value: (f64, f64),
}

pub fn evaluate_antibody(
    params: &Params,
    antigen_pop: &AntiGenPop,
    antibody: &Antibody) -> Evaluation {
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

    let registered_antigens : Vec<_> = if params.gpu_accelerate{
        let mask = antigen_pop.get_registered_antigens(&antibody, &None);
        antigen_pop
            .antigens
            .iter()
            .zip(mask.iter())
            .filter(|(a, b)| **b)
            .map(|(a, b)| a)
            .collect()
    }else {

        antigen_pop.antigens
            .iter()
            // .filter(|ag| idx_list.binary_search(&ag.id).is_ok())
            .filter(|ag| antibody.test_antigen(ag))
            .collect::<Vec<_>>()
    };




    //
    // let registered_antigens = antigens
    //     .iter()
    //     // .filter(|ag| idx_list.binary_search(&ag.id).is_ok())
    //     .filter(|ag| antibody.test_antigen(ag))
    //     .collect::<Vec<_>>();

    if true {
        let gpu_start = Instant::now();
        let mask = antigen_pop.get_registered_antigens(&antibody, &None);

        let duration_gpu = gpu_start.elapsed();

        let test_a: Vec<_> = antigen_pop
            .antigens
            .iter()
            .zip(mask.iter())
            .filter(|(a, b)| **b)
            .map(|(a, b)| a)
            .collect();

        let cpu_start = Instant::now();
        let test_b: Vec<_> = antigen_pop.antigens
            .iter()
            .filter(|ag| antibody.test_antigen(ag))
            .collect::<Vec<_>>();
        let duration_cpu = cpu_start.elapsed();
        println!("cpu time {:?}, gpu time {:?}, matches {:?}", duration_cpu, duration_gpu, test_a.len());

        let is_eq = test_a.iter().zip(&test_b).all(|(a,b)| a.id == b.id);
        println!("is eq: {:?}", is_eq);

        if duration_gpu.as_millis() < 50{
            // println!("is eq: {:?}", antibody);
            // println!("gpu: {:?} vs cpu: {:?}", test_a.len(), test_b.len());
        }

        if !is_eq{//test_a.len() != test_b.len() {
            println!("gpu: {:?} vs cpu: {:?}", test_a.len(), test_b.len());
            if true{

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
            println!("gpu  res: {:?}", test_a);
            println!();
            println!("cpu res: {:?}", test_b);
            }

        }

        // println!("a {:?} b {:?}", test_a.len(), test_b.len());
    }

    let mut corr_matched = Vec::with_capacity(antigen_pop.antigens.len());
    let mut wrong_matched = Vec::with_capacity(antigen_pop.antigens.len());

    for registered_antigen in &registered_antigens {
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

    // -- membership value -- //
    let same_label_membership = corr_matched.len() as f64
        / (corr_matched.len() as f64 + wrong_matched.len() as f64).max(1.0);

    let ret_evaluation = Evaluation {
        matched_ids: corr_matched,
        wrongly_matched: wrong_matched,
        membership_value: (same_label_membership, 1.0 - same_label_membership),
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

    pub boosting_weight_values: Vec<f64>,
    pub class_labels: HashSet<usize>,
    pub frac_map: Vec<(usize, f64)>,
    pub count_map: HashMap<usize, usize>,
}

impl MatchCounter {
    pub fn new(antigens: &Vec<AntiGen>) -> MatchCounter {
        let max_id = antigens.iter().max_by_key(|ag| ag.id).unwrap().id;
        let mut ag_weight_map = vec![0.0; max_id + 1];
        antigens.iter().for_each(|ag| {
            if let Some(v) = ag_weight_map.get_mut(ag.id) {
                *v = ag.boosting_weight
            }
        });

        let class_labels = antigens
            .iter()
            .map(|x| x.class_label)
            .collect::<HashSet<_>>();

        let frac_map: Vec<(usize, f64)> = class_labels
            .iter()
            .map(|x| {
                (
                    x.clone(),
                    antigens
                        .iter()
                        .filter(|ag| ag.class_label == *x)
                        .collect::<Vec<&AntiGen>>()
                        .len() as f64
                        / antigens.len() as f64,
                )
            })
            .collect();
        let count_map: HashMap<usize, usize> = class_labels
            .iter()
            .map(|x| {
                (
                    x.clone(),
                    antigens
                        .iter()
                        .filter(|ag| ag.class_label == *x)
                        .collect::<Vec<&AntiGen>>()
                        .len(),
                )
            })
            .collect();

        // println!("count map tot counts: {:?}", count_map);

        return MatchCounter {
            max_id,
            correct_match_counter: vec![0usize; max_id + 1],
            incorrect_match_counter: vec![0usize; max_id + 1],
            boosting_weight_values: ag_weight_map,
            class_labels,
            frac_map,
            count_map,
        };
    }

    pub fn get_inversed_correct_match_counts(&self) -> Vec<usize> {
        let max_val = self.correct_match_counter.iter().max().unwrap();
        let mut inverse = self.correct_match_counter.clone();
        inverse = inverse.iter().map(|v| max_val - v).collect();
        return inverse;
    }

    pub fn get_inversed_incorrect_match_counts(&self) -> Vec<usize> {
        let max_val = self.incorrect_match_counter.iter().max().unwrap();
        let mut inverse = self.incorrect_match_counter.clone();
        inverse = inverse.iter().map(|v| max_val - v).collect();
        return inverse;
    }

    pub fn add_evaluations(&mut self, evaluations: Vec<&Evaluation>) {
        for evaluation in evaluations {
            for correct_match_id in &evaluation.matched_ids {
                if let Some(elem) = self.correct_match_counter.get_mut(*correct_match_id) {
                    *elem += 1;
                } else {
                    println!(
                        "match id {:?} arr len {:?}",
                        correct_match_id,
                        self.correct_match_counter.len()
                    );
                    panic!("match count error")
                }
            }

            for correct_match_id in &evaluation.wrongly_matched {
                if let Some(elem) = self.incorrect_match_counter.get_mut(*correct_match_id) {
                    *elem += 1;
                } else {
                    panic!("match count error")
                }
            }
        }
    }

    pub fn remove_evaluations(&mut self, evaluations: Vec<&Evaluation>) {
        for evaluation in evaluations {
            for correct_match_id in &evaluation.matched_ids {
                if let Some(elem) = self.correct_match_counter.get_mut(*correct_match_id) {
                    *elem -= 1;
                } else {
                    panic!("match count error")
                }
            }

            for correct_match_id in &evaluation.wrongly_matched {
                if let Some(elem) = self.incorrect_match_counter.get_mut(*correct_match_id) {
                    *elem -= 1;
                } else {
                    panic!("match count error")
                }
            }
        }
    }
}
