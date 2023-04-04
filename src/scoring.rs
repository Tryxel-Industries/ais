use std::cmp::min;
use std::collections::HashMap;

use rayon::prelude::*;

use crate::bucket_empire::{BucketKing, ValueRangeType};
use crate::evaluation::{Evaluation, MatchCounter};
use crate::representation::antibody::{Antibody, DimValueType};
use crate::representation::antigen::AntiGen;

pub fn score_antibodies(
    evaluated_population: Vec<(Evaluation, Antibody)>,
    count_map: &HashMap<usize, usize>,
    match_counter: &MatchCounter
) -> Vec<(f64, Evaluation, Antibody)> {

    let merged_mask: &Vec<usize> = match_counter.correct_match_counter.as_ref();
    let error_merged_mask: &Vec<usize> = match_counter.incorrect_match_counter.as_ref();
    // println!("{:?}", merged_mask);
    // println!("len {:?}", merged_mask.len());
    // println!("sum {:?}", merged_mask.iter().sum::<usize>());
    // println!("match -s: ");
    let scored = evaluated_population
        // .into_iter()
        .into_par_iter() // TODO: set paralell
        .map(|(eval, cell)| {
            let mut true_positives: f64 = 0.0;
            let mut false_positives: f64 = 0.0;

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
                error_problem_magnitude += cor_shares / (cor_shares + sharers).max(1.0);

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

            let beta: f64 = 0.5f64.powi(2);

            let f1_divisor =
                (1.0 + beta) * true_positives + beta * false_negatives + false_positives;
            let f1_top = (1.0 + beta) * true_positives;
            let f1 = f1_top / f1_divisor;

            // the precession
            let positive_predictive_value = true_positives / pred_pos.max(1.0);

            let pos_coverage = true_positives / positives;
            let neg_coverage = pred_neg / negatives;

            // how shared are the values for the predictor
            let purity = true_positives / shared_positive_weight as f64;

            let penalty = 1.0 - (error_problem_magnitude / false_positives.max(1.0));
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

            if pred_pos == tot_elements {
                score = -5.0;
            }

            return (score, eval, cell);
        })
        .collect();

    return scored;
}
