use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::f32::consts::E;
use std::ops::Deref;

use rayon::prelude::*;

use crate::bucket_empire::{BucketKing, ValueRangeType};
use crate::params::Params;
use crate::representation::antibody::{Antibody, DimValueType};
use crate::representation::antigen::AntiGen;
use crate::representation::evaluated_antibody::EvaluatedAntibody;
use crate::stupid_mutations::MutationOp;

#[derive(Clone, Debug)]
pub struct Evaluation {
    pub matched_ids: Vec<usize>,
    pub matched_afin: Vec<f64>,
    pub wrongly_matched: Vec<usize>,
    pub wrongly_matched_afin: Vec<f64>,
    pub membership_value: (f64, f64),
    pub affinity_weight: (f64, f64),
    pub affinity_ag_map: HashMap<usize, (f64, usize)>
}

impl Evaluation {

    pub fn new(antigens: &Vec<AntiGen>, antibody: &Antibody)-> Evaluation{

    let affinity_ag_map: HashMap<_, (_,_)> =   antigens
            // .iter()
            .par_iter()
            .map(|ag| (ag.id,(calculate_affinity_dist(antibody,ag), ag.class_label)))
            .collect::<HashMap<usize,(f64,usize)>>();

        return Evaluation{
            matched_ids: vec![],
            matched_afin: vec![],
            wrongly_matched: vec![],
            wrongly_matched_afin: vec![],
            membership_value: (0.0, 0.0),
            affinity_weight: (0.0, 0.0),
            affinity_ag_map,
        }

    }
    pub fn update_eval(&mut self, cur_label: usize){

        let  registered_antigens =  self.affinity_ag_map
            .iter()
            .filter(| (id,(afin,label))| is_afinity_registering(*afin))
            .map(|(id, (afin,label))| (id, afin, label))
            .collect::<Vec<_>>();
        // println!("registered {:?}", registered_antigens.len());



        let mut corr_matched = Vec::with_capacity(registered_antigens.len());
        let mut corr_matched_afin = Vec::with_capacity(registered_antigens.len());
    // let mut corr_matched_weighted_count = Vec::with_capacity(registered_antigens.len());

    let mut wrong_matched = Vec::with_capacity(registered_antigens.len());
    let mut wrong_matched_afin = Vec::with_capacity(registered_antigens.len());
    // let mut wrong_matched_weighted_count = Vec::with_capacity(registered_antigens.len());

    for (id, afin, label) in registered_antigens {
        if *label == cur_label {
            corr_matched.push(*id);
            corr_matched_afin.push(*afin);
        } else {
            wrong_matched.push(*id);
            wrong_matched_afin.push(*afin);
        }
    }

    let same_label_membership = corr_matched.len() as f64 / (corr_matched.len() as f64 + wrong_matched.len() as f64).max(1.0);

    let corr_match_afin_sum:f64 = corr_matched_afin.iter().sum::<f64>() * -1.0;
    let wrong_match_afin_sum:f64 = wrong_matched_afin.iter().sum::<f64>() * -1.0;

    let balance = corr_match_afin_sum/ (corr_match_afin_sum+wrong_match_afin_sum).max(1.0);


        self.matched_ids = corr_matched;
        self.matched_afin = corr_matched_afin;
        self.wrongly_matched = wrong_matched;
        self.wrongly_matched_afin = wrong_matched_afin;

        self.membership_value =  (same_label_membership, 1.0 - same_label_membership);
        self.affinity_weight = (balance, 1.0 - balance);

    }
}



fn calculate_affinity_dist(antibody: &Antibody, antigen: &AntiGen) -> f64 {
        let mut roll_sum: f64 = 0.0;
        for i in 0..antigen.values.len() {
            let b_dim = antibody.dim_values.get(i).unwrap();
            let antigen_dim_val = antigen.values.get(i).unwrap();
            roll_sum += match b_dim.value_type {
                DimValueType::Disabled => 0.0,
                DimValueType::Open => b_dim.multiplier * (antigen_dim_val - b_dim.offset),
                DimValueType::Circle => {
                    (b_dim.multiplier * (antigen_dim_val - b_dim.offset)).powi(2)
                }
            };
        }
        roll_sum -= antibody.radius_constant;
        return roll_sum;
}

fn is_afinity_registering(affinity: f64) -> bool{
    return if affinity <= 0.0 {
            true
        } else {
            false
        }
}



pub fn evaluate_antibody(antigens: &Vec<AntiGen>, antibody: &Antibody) -> Evaluation {
    let mut eval = Evaluation::new(antigens, antibody);
    eval.update_eval(antibody.class_label);
    return eval;

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
    pub ag_label_boost_sum_map: HashMap<usize,usize>,
    pub class_labels: HashSet<usize>,
    pub frac_map: Vec<(usize,f64)>,
    pub count_map: HashMap<usize,usize>,
}

impl MatchCounter {
    pub fn new(antigens: &Vec<AntiGen>) -> MatchCounter {
        let max_id = antigens.iter().max_by_key(|ag| ag.id).unwrap().id;
        let mut ag_weight_map = vec![0.0; max_id + 1];
        let mut ag_label_boost_sum_map: HashMap<usize,usize> = HashMap::new();
        antigens.iter().for_each(|ag| {
            if let Some(v) = ag_weight_map.get_mut(ag.id){
                *v = ag.boosting_weight
            }
            if let Some(v) = ag_label_boost_sum_map.get_mut(&ag.class_label){
                *v += 1;
            }else { 
                ag_label_boost_sum_map.insert(ag.class_label, 1);
            }
        } );


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
            ag_label_boost_sum_map,
            class_labels,
            frac_map,
            count_map

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

    pub fn add_evaluation(&mut self, evaluation: &Evaluation) {
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


    pub fn add_evaluations(&mut self, evaluations: &Vec<EvaluatedAntibody>) {
        for evaluated_ab in evaluations {
            self.add_evaluation(&evaluated_ab.evaluation);
        }
    }

    pub fn remove_evaluation(&mut self, evaluation: &Evaluation) {
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
    pub fn remove_evaluations(&mut self, evaluations: &Vec<EvaluatedAntibody>) {
        for evaluated_ab in evaluations {
            self.remove_evaluation(&evaluated_ab.evaluation);
        }
    }
}
