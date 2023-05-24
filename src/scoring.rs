use nalgebra::RealField;
use std::cmp::min;
use std::collections::HashMap;

use rayon::prelude::*;

use crate::bucket_empire::{BucketKing, ValueRangeType};
use crate::evaluation::{Evaluation, MatchCounter};
use crate::params::Params;
use crate::representation::antibody::{Antibody, DimValueType};
use crate::representation::antigen::AntiGen;
use crate::representation::evaluated_antibody::EvaluatedAntibody;

fn mod_sigmoid(num: f64) -> f64 {
    return 1.0 - (1.0 / (1.0 + f64::e().powf(num)));
}

pub fn score_antibody(
    eval_ab: &EvaluatedAntibody,
    params: &Params,
    match_counter: &MatchCounter,
) -> f64 {
    let eval = &eval_ab.evaluation;
    let cell = &eval_ab.antibody;

    let count_map = &match_counter.count_map;
    let merged_mask: &Vec<usize> = match_counter.correct_match_counter.as_ref();
    let error_merged_mask: &Vec<usize> = match_counter.incorrect_match_counter.as_ref();

    // the lover the deeper in side the zone
    let mut corr_affinity_heuristic: f64 = 0.0;
    let mut err_affinity_heuristic: f64 = 0.0;

    let mut true_positives: f64 = 0.0;
    let mut false_positives: f64 = 0.0;

    let mut discounted_match_score: f64 = 0.0;

    let mut unique_positives = 0;

    let mut shared_positive_weight = 0;
    let mut shared_error_weight = 0.0;

    let mut error_problem_magnitude: f64 = 0.0;

    let mut pos_relevance = 0.0;
    let mut neg_relevance = 0.0;

    for (mid, afin) in eval.matched_ids.iter().zip(eval.matched_afin.iter()) {
        let relevance = match_counter.boosting_weight_values.get(*mid).unwrap();
        pos_relevance += relevance;

        corr_affinity_heuristic += (mod_sigmoid(-1.0 * afin) * relevance);

        // println!("mod sig {:?}", mod_sigmoid(-1.0*afin));
        true_positives += 1.0;
        let sharers = merged_mask.get(*mid).unwrap_or(&0);

        if *sharers > 1 {
            let delta = (1.0 / *sharers as f64).max(0.0);
            discounted_match_score += delta * relevance;
            shared_positive_weight += *sharers;
        } else {
            unique_positives += 1;
        }
    }

    for (mid, afin) in eval
        .wrongly_matched
        .iter()
        .zip(eval.wrongly_matched_afin.iter())
    {
        let relevance = match_counter.boosting_weight_values.get(*mid).unwrap();
        neg_relevance += relevance;

        err_affinity_heuristic += (mod_sigmoid(-1.0 * afin) * relevance);

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

    // ##################### relevance ajust #####################
    // println!("");
    // println!(
    //     "true pos {:?} false pos {:?}",
    //     true_positives, false_positives
    // );
    pos_relevance /= true_positives.max(1.0);
    neg_relevance /= false_positives.max(1.0);

    corr_affinity_heuristic /= true_positives.max(1.0);
    err_affinity_heuristic /= false_positives.max(1.0);

    true_positives *= pos_relevance;
    false_positives *= neg_relevance;

    // println!("true pos {:?} false pos {:?}", true_positives, false_positives);

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

    let f1_divisor = (1.0 + beta) * true_positives + beta * false_negatives + false_positives;
    let f1_top = (1.0 + beta) * true_positives;
    let f1 = f1_top / f1_divisor;

    // the precession
    let positive_predictive_value = true_positives / pred_pos.max(1.0);

    let pos_coverage = true_positives / positives;
    let neg_coverage = pred_neg / negatives;

    // how shared are the values for the predictor
    let purity = true_positives / shared_positive_weight.max(1) as f64;

    let penalty = 1.0 - (error_problem_magnitude / false_positives.max(1.0));

    // positive_predictive_value + pos_coverage * discounted_match_score/true_positives.max(1.0) + penalty;

    // add ppv (0-1) indicating the accuracy of the prediction
    // score += positive_predictive_value;

    // add the pos coverage (0-1) indicating how big of a fraction of the space label space is covered.
    // score += pos_coverage/4.0;
    // score += f1;

    // add a value from 1 -> -1 indicating the fraction of correctness
    let correctness =
        (true_positives - (false_positives * 2.0)) / (true_positives + false_positives).max(1.0);

    // add a value from (0 - 1) * a indicating the coverage of the label space
    let coverage = pos_coverage * pos_relevance;

    let uniqueness = (discounted_match_score / (true_positives).max(1.0));

    let good_affinity = corr_affinity_heuristic;
    let bad_affinity = -1.0 * err_affinity_heuristic;
    // let mut score = good_affinity - bad_affinity*0.5+ coverage*0.5;

    // println!("score {:?}  good {:?} bad {:?}", score, good_affinity, bad_affinity );

    let mut score = correctness * params.correctness_weight
        + coverage * params.coverage_weight
        + uniqueness * params.uniqueness_weight
        + good_affinity * params.good_afin_weight
        + bad_affinity * params.bad_afin_weight;

    if pred_pos == tot_elements {
        score = -5.0;
    }
    return score;
}

pub fn score_antibodies(
    params: &Params,
    evaluated_population: Vec<EvaluatedAntibody>,
    match_counter: &MatchCounter,
) -> Vec<(f64, EvaluatedAntibody)> {
    // println!("{:?}", merged_mask);
    // println!("len {:?}", merged_mask.len());
    // println!("sum {:?}", merged_mask.iter().sum::<usize>());
    // println!("match -s: ");
    let scored = evaluated_population
        // .into_iter()
        .into_par_iter() // TODO: set paralell
        .map(|eab| {
            let score = score_antibody(&eab, params, match_counter);

            return (score, eab);
        })
        .collect();

    return scored;
}
