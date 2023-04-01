use std::cmp::min;
use std::collections::HashMap;
use crate::ais::Params;
use crate::bucket_empire::{BucketEmpireOfficialRangeNotationSystemClasses, BucketKing};
use crate::representation::{AntiGen, BCell, DimValueType};

use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Evaluation {
    pub matched_ids: Vec<usize>,
    pub wrongly_matched: Vec<usize>,
}

pub fn evaluate_b_cell(
    bk: &BucketKing<AntiGen>,
    antigens: &Vec<AntiGen>,
    b_cell: &BCell,
) -> Evaluation {
    // todo: flip offset mabye
    let dim_radus = b_cell
        .dim_values
        .iter()
        .map(|dv| {
            return match dv.value_type {
                DimValueType::Disabled => BucketEmpireOfficialRangeNotationSystemClasses::Open,
                DimValueType::Open => {
                    return BucketEmpireOfficialRangeNotationSystemClasses::Open;
                    // return  BucketEmpireOfficialRangeNotationSystemClasses::Open;
                    let value = (b_cell.radius_constant) / dv.multiplier;
                    if dv.multiplier > 0.0 {
                        return BucketEmpireOfficialRangeNotationSystemClasses::UpperBound(
                            value,
                        );
                    } else {
                        return BucketEmpireOfficialRangeNotationSystemClasses::LowerBound(
                            value,
                        );
                    }
                }
                DimValueType::Circle => {
                    let value = b_cell.radius_constant.sqrt() / dv.multiplier;
                    let flipped_offset = dv.offset;
                    return BucketEmpireOfficialRangeNotationSystemClasses::Symmetric((
                        flipped_offset - value,
                        flipped_offset + value,
                    ));
                }
            };
        })
        .collect();

    let mut idx_list = bk
        .get_potential_matches_indexes_with_raw_values(&dim_radus)
        .unwrap();

    idx_list.sort();

    let registered_antigens = antigens
        .iter()
        .filter(|ag| idx_list.binary_search(&ag.id).is_ok())
        .filter(|ag| b_cell.test_antigen(ag))
        .collect::<Vec<_>>();

    if true{
        let test_a = antigens
            .iter()
            .filter(|ag| idx_list.binary_search(&ag.id).is_ok())
            .filter(|ag| b_cell.test_antigen(ag))
            .collect::<Vec<_>>();


        let test_b = antigens
            .iter()
            .filter(|ag| b_cell.test_antigen(ag))
            .collect::<Vec<_>>();
        if test_a.len() != test_b.len(){
            println!();
            println!("cell vt    {:?}",b_cell.dim_values.iter().map(|b| b.value_type.clone()).collect::<Vec<_>>());
            println!("cell mp    {:?}",b_cell.dim_values.iter().map(|b| b.multiplier.clone()).collect::<Vec<_>>());
            println!("cell of    {:?}",b_cell.dim_values.iter().map(|b| b.offset.clone()).collect::<Vec<_>>());
            println!("cell rad : {:?}",b_cell.radius_constant);
            println!();
            println!("dim rad: {:?}",dim_radus);
            println!("bk  res: {:?}",test_a);
            println!();
            println!("otr res: {:?}",test_b);
            panic!("bucket empire error")

        }

        // println!("a {:?} b {:?}", test_a.len(), test_b.len());
    }

    let mut corr_matched = Vec::with_capacity(registered_antigens.len());
    let mut wrong_matched = Vec::with_capacity(registered_antigens.len());

    for registered_antigen in registered_antigens {
        if registered_antigen.class_label == b_cell.class_label {
            corr_matched.push(registered_antigen.id)
        } else {
            wrong_matched.push(registered_antigen.id)
        }
    }

    // let with_same_label = registered_antigens.iter().filter(|ag|ag.class_label==b_cell.class_label ).collect::<Vec<_>>();
    // let num_wrong = registered_antigens.iter().filter(|ag|ag.class_label!=b_cell.class_label ).collect::<Vec<_>>();
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
//  Merge mask
//

fn merge_evaluation_matches(evaluations: Vec<&Evaluation>, mask: Vec<usize>, is_error: bool) -> Vec<usize> {
    let merged = evaluations
        .iter()
        // .filter(|e| e.wrongly_matched.len() == 0)
        .map(|e| {
            if is_error{
                &e.wrongly_matched
            }else{
                &e.matched_ids
            }
        })
        .fold(mask, |mut acc, b| {
            b.iter().for_each(|v| {
                if let Some(elem) = acc.get_mut(*v) {
                    *elem += 1;
                }
            });
            return acc;
        });
    return merged;
}

fn _gen_merge_mask(scored_population: &Vec<(Evaluation, BCell)>, is_error: bool) -> Vec<usize> {
    let evaluations = scored_population
        .iter()
        .map(|(b, _)| b)
        .collect::<Vec<&Evaluation>>();

    let max_id = evaluations
        .iter()
        .map(|e| e.matched_ids.iter().max().unwrap_or(&0).max(e.wrongly_matched.iter().max().unwrap_or(&0)))
        .max()
        .unwrap()
        + 1;

    let mask = vec![0usize; max_id];

    return merge_evaluation_matches(evaluations, mask, is_error);
}

pub fn gen_merge_mask(scored_population: &Vec<(Evaluation, BCell)>) -> Vec<usize> {
    return  _gen_merge_mask(scored_population, false);
}

pub fn gen_error_merge_mask(scored_population: &Vec<(Evaluation, BCell)>) -> Vec<usize> {
    return  _gen_merge_mask(scored_population, true);
}

pub fn expand_merge_mask(
    scored_population: &Vec<(Evaluation, BCell)>,
    mask: Vec<usize>,
    is_error_mask: bool,
) -> Vec<usize> {
    let evaluations = scored_population
        .iter()
        .map(|(b, _)| b)
        .collect::<Vec<&Evaluation>>();
    return merge_evaluation_matches(evaluations, mask, is_error_mask);
}



pub fn score_b_cells(
    scored_population: Vec<(Evaluation, BCell)>,
    merged_mask: &Vec<usize>,
    error_merged_mask: &Vec<usize>,
    count_map: &HashMap<usize,usize>,
) -> Vec<(f64, Evaluation, BCell)> {
    // println!("{:?}", merged_mask);
    // println!("len {:?}", merged_mask.len());
    // println!("sum {:?}", merged_mask.iter().sum::<usize>());
    // println!("match -s: ");
    let scored = scored_population
        // .into_iter()
        .into_par_iter() // TODO: set paralell
        .map(|(eval, cell)| {


            let mut true_positives:f64 = 0.0;
            let mut false_positives :f64= 0.0;

            let mut discounted_match_score: f64 = 0.0;

            let mut unique_positives = 0;

            let mut shared_positive_weight = 0;
            let mut shared_error_weight = 0.0;

            let mut error_problem_magnitude: f64 = 0.0;


            for mid in &eval.matched_ids {
                true_positives += 1.0;
                let sharers = merged_mask.get(*mid).unwrap_or(&0);

                if *sharers > 1 {
                    let delta = (1.0 / *sharers as f64).max(0.0);
                    discounted_match_score += delta;
                    shared_positive_weight += *sharers;
                } else {
                    unique_positives += 1;
                }
            }

            for mid in &eval.wrongly_matched {
                false_positives += 1.0;
                let sharers = *error_merged_mask.get(*mid).unwrap_or(&0) as f64;
                let cor_shares = *merged_mask.get(*mid).unwrap_or(&0) as f64;

                // when the amount of wrong predictivness goes up the value goes down
                error_problem_magnitude += cor_shares/ (cor_shares+sharers).max(1.0);

                if sharers > 1.0 {
                    // bonus_error += 1.0-(1.0 / *sharers as f64);

                    shared_error_weight += sharers;
                } else {
                }
            }
            // ##################### F1 stuff #####################

            let label_tot_count = *count_map.get(&cell.class_label).unwrap() as f64;
            let tot_elements = count_map.values().sum::<usize>() as f64;

            let pred_pos = true_positives + false_positives;
            let pred_neg = tot_elements - pred_pos;

            let positives = label_tot_count;
            let negatives = tot_elements - positives;

            let false_negatives = positives - true_positives;
            let true_negatives = negatives - false_positives;

            let beta : f64 = 0.5f64.powi(2);

            let f1_divisor = (1.0+ beta) * true_positives + beta*false_negatives + false_positives;
            let f1_top = (1.0+beta)*true_positives;
            let f1 = f1_top/f1_divisor;


            // the precession
            let positive_predictive_value = true_positives / pred_pos.max(1.0);

            let pos_coverage = true_positives / positives;
            let neg_coverage = pred_neg / negatives;


            // how shared are the values for the predictor
            let purity = true_positives/shared_positive_weight as f64;


            let penalty =  1.0 - (error_problem_magnitude/ false_positives.max(1.0));
            let mut score = 0.0;
            // positive_predictive_value + pos_coverage * discounted_match_score/true_positives.max(1.0) + penalty;

            // add ppv (0-1) indicating the accuracy of the prediction
            // score += positive_predictive_value;

            // add the pos coverage (0-1) indicating how big of a fraction of the space label space is covered.
            // score += pos_coverage/4.0;
            score += f1;
            // score += (true_positives-false_positives)/(true_positives+false_positives).max(1.0);

            // score +=  discounted_match_score/true_positives.max(1.0);




            // let mut score = positive_predictive_value + pos_coverage + discounted_match_score/true_positives.max(1.0);


            // let precession = if positive_predictive_value.is_finite(){positive_predictive_value} else { 0.0 };
            // let purity = if purity.is_finite(){purity } else {0.0};
            // let score = precession + purity;

            // let score = true_positives / ((shared_positive_weight + shared_error_weight) as f64).max(1.0);
            // let score = crowdedness+purity+accuracy ;


            // let score = f1 / (shared_error_weight as f64).max(1.0);

            // let score = pred_pos / (false_positives  as f64).max(1.0);

            // let score = ((base_sum - n_wrong)+(bonus_sum*0.5))*accuracy ;
            // let score = ((discounted_sum) - (n_wrong).powi(2)) + bonus_sum  - bonus_error;

            // let score = ((true_positives) - (true_negatives).powi(2)) + unique_positives as f64 * 5.0;

            // let mut divisor = (n_wrong + bonus_error).powi(2) + n_shared_error as f64;
            // divisor = if (divisor.is_finite()) | (divisor != 0.0) {divisor} else { 1.0 };

            // let score = ((n_right + (bonus_sum+1.0).powi(1)) / (divisor+1.0) );

            // let score = precession + discounted_match_score/true_positives.max(1.0);


            // let score = ((true_positives + discounted_match_score) - (false_positives).powi(2)) + unique_positives as f64 * 5.0  - (shared_error_weight as f64/2.0);

            // let score = ((discounted_sum) - (bonus_error)) + bonus_sum * 5.0;
            // let score = (matched_sum - n_wrong).max(0.0) ;


            // println!("###########################");
            // println!("match:        {:?}", eval.matched_ids);
            // println!("wrong match:  {:?}", eval.wrongly_matched);
            // println!("ppv        {:?}", positive_predictive_value);
            // println!("pos cov    {:?}", pos_coverage);
            // println!("penalty    {:?}", penalty);
            // println!("bonus       {:?}", bonus_sum);
            // println!("divisor       {:?}", divisor);
            // println!("final score {:?}", score);

            if pred_pos == tot_elements{
                score = -5.0;
            }


            return (score, eval, cell);
        })
        .collect();

    return scored;
}

// pub fn score_with_binary(scored_population: Vec<(Evaluation, BCell)>, merged_mask: &Vec<usize>) -> Vec<(f64, Evaluation, BCell)>{
//
//     // println!("{:?}", merged_mask);
//     // println!("len {:?}", merged_mask.len());
//     // println!("sum {:?}", merged_mask.iter().sum::<usize>());
//     // println!("match -s: ");
//     let scored = scored_population
//         // .into_iter()
//         .into_par_iter() // TODO: set paralell
//         .map(|(eval, cell)| {
//             let mut bonus_sum: f64 = 0.0;
//             let mut base_sum: f64 = 0.0;
//
//             let mut n_shared = 0;
//             let mut roll_shared = 0;
//
//             for mid in &eval.matched_ids {
//                 let sharers = merged_mask.get(*mid).unwrap_or(&0);
//                 if *sharers > 1 {
//                     n_shared += 1 as usize;
//                     roll_shared += sharers;
//                     // matched_sum += (0.5+(1.0 / ((*sharers as f64) / 2.0))).min(1.0)
//                     // matched_sum += (1.0 / ((*sharers as f64) / 2.0)).max(0.5)
//
//                     let delta =  (1.0/ *sharers as f64).min(0.0);
//                     // let delta =  (1.0 / ((*sharers as f64) / 2.0)).max(0.5);
//
//
//                     bonus_sum += delta;
//                 } else {
//                     bonus_sum += 5.0;
//                 }
//                 base_sum += 1.0;
//             }
//             let n_wrong = eval.wrongly_matched.len() as f64;
//             let n_right = eval.matched_ids.len() as f64;
//
//             let mut purity = n_right / n_wrong;
//             let mut accuracy = n_right / (n_wrong + n_right);
//             let mut crowdedness = 1.0 / (roll_shared as f64 / n_shared as f64);
//
//             if !purity.is_finite() {
//                 purity = 0.0;
//             }
//             if !accuracy.is_finite() {
//                 accuracy = 0.0;
//             }
//             if !crowdedness.is_finite() {
//                 crowdedness = 0.0;
//             }
//
//             // let score = crowdedness+purity+accuracy ;
//             // let score = ((base_sum - n_wrong)+(bonus_sum*0.5))*accuracy ;
//             // let score = ((bonus_sum) / (n_wrong + 1.0));
//             // let score = (matched_sum - n_wrong).max(0.0) ;
//
//
//             let score = ((base_sum) / (n_wrong + 1.0));
//
//             // println!("\n score {:?}", score);
//             return (score, eval, cell);
//         })
//         .collect();
//
//     return scored;
// }
