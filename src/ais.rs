use itertools::multiunzip;
use json::object;
use std::collections::{HashMap, HashSet};
use std::io;
use std::io::Write;
use std::iter::Map;
use std::ops::{Range, RangeInclusive};
use std::time::Instant;

use rand::prelude::{IteratorRandom, SliceRandom};
use rand::Rng;
use rayon::prelude::*;
use statrs::statistics::Statistics;

use crate::bucket_empire::BucketKing;
use crate::display::eval_display;
use crate::evaluation::{evaluate_antibody, Evaluation, MatchCounter};
use crate::experiment_logger::LoggedValue::{CorWrongNoReg, SingleValue, TrainTest};
use crate::experiment_logger::{ExperimentLogger, ExperimentProperty, LoggedValue};
use crate::params::{MutationType, Params, PopSizeType, ReplaceFractionType, VerbosityParams};
use crate::params::MutationType::ValueType;
use crate::prediction::{is_class_correct, make_prediction, EvaluationMethod, Prediction};
use crate::representation::antibody::{Antibody, DimValueType, InitType};
use crate::representation::antibody_factory::AntibodyFactory;
use crate::representation::antigen::AntiGen;
use crate::representation::evaluated_antibody::EvaluatedAntibody;
use crate::representation::news_article_mapper::NewsArticleAntigenTranslator;
use crate::representation::{evaluate_population, expand_antibody_radius_until_hit};
use crate::scoring::{score_antibodies, score_antibody};
use crate::selection::{elitism_selection, kill_by_mask_yo, labeled_tournament_pick, pick_best_n, remove_strictly_worse, replace_if_better, replace_if_better_per_cat, replace_worst_n_per_cat, snip_worst_n, tournament_pick};
use crate::stupid_mutations::mutate_clone_transform;
use crate::util::get_pop_acc;

/*
1. select n parents
2. clone -> mutate parents n times
 */
pub struct ArtificialImmuneSystem {
    pub antibodies: Vec<Antibody>,
}

//
//  AIS
//

fn gen_initial_population(
    antigens: &Vec<AntiGen>,
    params: &Params,
    cell_factory: &AntibodyFactory,
    pop_size: &usize,
    match_counter: &MatchCounter,
) -> Vec<Antibody> {
    let mut rng = rand::thread_rng();
    return if *pop_size == antigens.len() {
        antigens
            .par_iter()
            .map(|ag| cell_factory.generate_from_antigen(ag))
            .map(|cell| {
                if params.antibody_init_expand_radius {
                    expand_antibody_radius_until_hit(cell, &antigens)
                } else {
                    cell
                }
            })
            .collect()
    } else {
        match_counter
            .frac_map
            .iter()
            .flat_map(|(label, frac)| {
                let replace_count_for_label = (*pop_size as f64 * frac).ceil() as usize;

                let filtered: Vec<_> = antigens
                    .iter()
                    .filter(|ag| ag.class_label == *label)
                    .collect();

                return (0..replace_count_for_label)
                    .par_bridge()
                    .map(|_| {
                        cell_factory.generate_from_antigen(
                            filtered.choose(&mut rand::thread_rng()).unwrap(),
                        )
                    })
                    .map(|cell| {
                        if params.antibody_init_expand_radius {
                            expand_antibody_radius_until_hit(cell, &antigens)
                        } else {
                            cell
                        }
                    })
                    .collect::<Vec<_>>();
            })
            .collect()
    };
}

// fn evaluate_print_population(
//     antigens: &Vec<AntiGen>,
//     antibodies: &Vec<Antibody>,
//     eval_method: &EvaluationMethod,
//
//     logger: &mut ExperimentLogger,
// ) {
//     let mut test_n_corr = 0;
//     let mut test_n_wrong = 0;
//
//     let mut membership_test_n_corr = 0;
//     let mut membership_test_n_wrong = 0;
//
//     let mut test_per_class_corr = HashMap::new();
//     let mut test_n_no_detect = 0;
//     for antigen in antigens {
//         let pred_class = is_class_correct_with_membership(&antigen, antibodies);
//
//         if let Some(v) = pred_class {
//             if v {
//                 membership_test_n_corr += 1;
//             } else {
//                 membership_test_n_wrong += 1
//             }
//         }
//         let pred_class = is_class_correct(&antigen, antibodies, eval_method);
//         if let Some(v) = pred_class {
//             if v {
//                 test_n_corr += 1;
//                 let class_count = test_per_class_corr.get(&antigen.class_label).unwrap_or(&0);
//                 test_per_class_corr.insert(antigen.class_label, *class_count + 1);
//             } else {
//                 test_n_wrong += 1
//             }
//         } else {
//             test_n_no_detect += 1
//         }
//     }
//
//     let test_acc = test_n_corr as f64 / (antigens.len() as f64);
//     let test_precession = test_n_corr as f64 / (test_n_wrong as f64 + test_n_corr as f64).max(1.0);
//     let membership_test_acc = membership_test_n_corr as f64 / (antigens.len() as f64);
//     let membership_test_precession = membership_test_n_corr as f64
//         / (membership_test_n_wrong as f64 + membership_test_n_corr as f64).max(1.0);
//
//     println!(
//         "without membership: corr {:>2?}, false {:>3?}, no_detect {:>3?}, precession: {:>2.3?}, frac: {:2.3?}",
//         test_n_corr, test_n_wrong, test_n_no_detect,test_precession, test_acc
//     );
//     println!(
//         "with membership:    corr {:>2?}, false {:>3?}, no_detect {:>3?}, precession: {:>2.3?}, frac: {:2.3?}",
//         membership_test_n_corr, membership_test_n_wrong, antigens.len()-(membership_test_n_corr+membership_test_n_wrong), membership_test_precession, membership_test_acc
//     );
//
// }

impl ArtificialImmuneSystem {
    pub fn new() -> ArtificialImmuneSystem {
        return Self {
            antibodies: Vec::new(),
        };
    }

    pub fn train(
        &mut self,
        antigens: &Vec<AntiGen>,
        params: &Params,
        verbosity_params: &VerbosityParams,
        logger: &mut ExperimentLogger,
        highly_dubious_practices: &Vec<AntiGen>,
    ) -> (Vec<f64>, Vec<f64>, Vec<(f64, Evaluation, Antibody)>) {
        let pop_size = match params.antigen_pop_size {
            PopSizeType::Fraction(fraction) => (antigens.len() as f64 * fraction) as usize,
            PopSizeType::Number(n) => n,
            PopSizeType::BoostingFixed(n) => n,
        };

        let (train_acc_hist, train_score_hist, scored_pop) =
            self.train_ab_set(antigens, params, verbosity_params, pop_size, logger, highly_dubious_practices);
        let mut save_antibodies = Vec::new();
        scored_pop
            .iter()
            .for_each(|(a, b, cell)| save_antibodies.push(cell.clone()));
        self.antibodies = save_antibodies;
        return (train_acc_hist, train_score_hist, scored_pop);
    }

    pub fn train_immunobosting(
        &mut self,
        input_antigens: &Vec<AntiGen>,
        params: &Params,
        verbosity_params: &VerbosityParams,
        boosting_rounds: usize,
        highly_dubious_practices: &Vec<AntiGen>,
        translator: &NewsArticleAntigenTranslator,
        logger: &mut ExperimentLogger,
    ) -> (Vec<f64>, Vec<f64>, Vec<(f64, Evaluation, Antibody)>) {
        let mut antigens = input_antigens.clone();

        let train_per_round = if let PopSizeType::BoostingFixed(count) = params.antigen_pop_size {
            count
        } else {
            let pop_size = match params.antigen_pop_size {
                PopSizeType::Fraction(fraction) => {
                    (antigens.len() as f64 * fraction).ceil() as usize
                }
                PopSizeType::Number(n) => n,
                _ => {
                    panic!("error")
                }
            };
            ((pop_size as f64) * 1.0 / boosting_rounds as f64).ceil() as usize
        };

        if verbosity_params.print_boost_info {
            println!(
                "pop size, final num ab {:?} gained through {:?} rounds with {:?}",
                train_per_round * boosting_rounds,
                boosting_rounds,
                train_per_round
            );
        }

        let mut train_boost_hist = Vec::new();
        let mut test_boost_hist = Vec::new();

        let mut ab_pool: Vec<Antibody> = Vec::new();

        let max_retries = 5;
        let mut current_retries = 0;

        let mut current_boost_round = 0;

        let agl = antigens.len() as f64;
        antigens
            .iter_mut()
            .for_each(|ag| ag.boosting_weight = (1.0 / agl));
        loop {
            if current_retries >= max_retries {
                current_boost_round += 1;
                current_retries = 0;
            }
            if current_boost_round >= boosting_rounds {
                break;
            }

            let start = Instant::now();

            // using https://towardsdatascience.com/boosting-algorithms-explained-d38f56ef3f30

            // norm weights

            let ag_weights: Vec<f64> = antigens.iter().map(|ag| ag.boosting_weight).collect();
            // println!("boost weights pre norm {:?}", ag_weights);

            let max_w = (&ag_weights).max();
            let min_w = (&ag_weights).min();
            let max_min = max_w - min_w;

            let mean_w = (&ag_weights).mean();
            let std_w = (&ag_weights).std_dev();

            let soft_m_sum: f64 = antigens.iter().map(|ag| ag.boosting_weight).sum();

            // println!("max w {:?}, min_w {:?}", max_w, min_w);
            // let ag_weight_sum: f64 = antigens.iter().map(|ag| ag.boosting_weight).sum();
            // println!("sum is {:?},", antigens.iter().map(|ag| ag.boosting_weight).sum::<f64>());
            if max_min != 0.0 {
                antigens
                    .iter_mut()
                    // .for_each(|ag| ag.boosting_weight = (ag.boosting_weight -mean_w));
                    // .for_each(|ag| ag.boosting_weight = (ag.boosting_weight/ soft_m_sum));
                    .for_each(|ag| {
                        ag.boosting_weight = ((ag.boosting_weight - min_w) / max_min) + 0.1
                    });
                // .for_each(|ag| ag.boosting_weight = ((ag.boosting_weight - mean_w)/ std_w) + 1.0);
            } else {
                println!("Not norming boost round")
            }

            let ag_weights: Vec<f64> = antigens.iter().map(|ag| ag.boosting_weight).collect();
            // println!("boost weights post norm {:?}", ag_weights);

            // println!("sum is {:?},", antigens.iter().map(|ag| ag.boosting_weight).sum::<f64>());

            //   println!();
            //     let mut match_counter = MatchCounter::new(&antigens);
            //     println!(
            //     "########## boost mask \n{:?}",
            //     match_counter.boosting_weight_values
            // );

            // train predictor
            let (train_acc_hist, train_score_hist, scored_pop) =
                self.train_ab_set(&antigens, params, verbosity_params, train_per_round, logger, highly_dubious_practices);
            let (_, eval, mut antibodies): (Vec<f64>, Vec<Evaluation>, Vec<Antibody>) =
                multiunzip(scored_pop);

            ////////////////////////////////////////// TEST
            /*

                        let weighted_registered_antigens: Vec<_> = antigens
                            .par_iter()
                            .filter_map(|ag| {
                                if let Some(correct) = is_class_correct(&ag, &antibodies, &params.eval_method) {
                                    Some((correct, ag))
                                } else {
                                    None
                                }
                            })
                            .collect();

                        let weighted_error_count: f64 = weighted_registered_antigens
                            .iter()
                            .filter(|(is_cor, ag)| !(*is_cor))
                            .map(|(is_cor, ag)| ag.boosting_weight)
                            .sum();

                        let full_weight_sum: f64 = weighted_registered_antigens
                            .iter()
                            .map(|(is_cor, ag)| ag.boosting_weight)
                            .sum();
                        let full_error = weighted_error_count / full_weight_sum;

                        // calculate alpha weight for model
                        let full_alpha_m = ((1.0 - full_error) / full_error).ln();

                        // println!("Alpha is ({:.2?})", alpha_m);

                        let ab_count = antibodies.len();
                        let mut update_map: HashMap<usize, (Vec<f64>, Vec<f64>)> = HashMap::new();
                        let mut alpha_update_map: HashMap<usize, Vec<f64>> = HashMap::new();

                        antibodies
                            .iter_mut()
                            .zip(eval.into_iter())
                            .for_each(|(ab, eval)| {
                                let cor_ag: Vec<_> = eval
                                    .matched_ids
                                    .iter()
                                    .map(|id| antigens.iter().find(|ag| ag.id == *id).unwrap())
                                    .collect();
                                let wrong_ag: Vec<_> = eval
                                    .wrongly_matched
                                    .iter()
                                    .map(|id| antigens.iter().find(|ag| ag.id == *id).unwrap())
                                    .collect();

                                let corr_weight_sum: f64 = cor_ag.iter().map(|ag| ag.boosting_weight).sum();
                                let wrong_weight_sum: f64 = wrong_ag.iter().map(|ag| ag.boosting_weight).sum();

                                let error = wrong_weight_sum / corr_weight_sum;

                                // calculate alpha weight for the ag
                                let alpha_m = ((1.0 - error) / error).ln();
                                if alpha_m <= 0.0 {
                                    println!(
                                        "Alpha is negative ({:.2?}) error: {:.2?} discarding round",
                                        alpha_m, error
                                    );
                                }
                                ab.boosting_model_alpha = alpha_m;

                                wrong_ag.iter().for_each(|ag| {
                                    if let Some(alpha_list) = alpha_update_map.get_mut(&ag.id) {
                                        alpha_list.push(ag.boosting_weight * alpha_m.exp())
                                    } else {
                                        alpha_update_map
                                            .insert(ag.id, vec![ag.boosting_weight * alpha_m.exp()]);
                                    }
                                    if let Some((cor, wrong)) = update_map.get_mut(&ag.id) {
                                        cor.push(corr_weight_sum);
                                        wrong.push(wrong_weight_sum);
                                    } else {
                                        let cor = vec![corr_weight_sum];
                                        let wrong = vec![wrong_weight_sum];
                                        update_map.insert(ag.id, (cor, wrong));
                                    }
                                });
                            });
                        ab_pool.extend(antibodies);

                        //
                        let mut n_corr = 0;
                        let mut n_wrong = 0;
                        let mut n_no_reg = 0;
                        /*             for antigen in &mut antigens {
                            let optn = alpha_update_map.get(&antigen.id);
                            if optn.is_none(){
                                continue
                            }
                            let updates = optn.unwrap();

                               let scaled_updates_sum: f64 = updates.iter().map(|upd| upd * (1.0/ab_count as f64)).sum();

                            antigen.boosting_weight = antigen.boosting_weight + scaled_updates_sum  ;
                        }*/

                        // update weights
                        for antigen in &mut antigens {
                            if full_alpha_m <= 0.0 {
                                println!(
                                    "Alpha is negative ({:.2?}) error: {:.2?} discarding round",
                                    full_alpha_m, full_error
                                );
                            }

                            let pred = is_class_correct(&antigen, &ab_pool, &params.eval_method);

                            let val: f64 = if let Some(is_corr) = pred {
                                if is_corr {
                                    0.0
                                } else {
                                    // full_alpha_m
                                    1.3
                                    // 1.0
                                }
                            } else {
                                // 0.0
                                1.1
                                // full_alpha_m
                            };

                            antigen.boosting_weight = antigen.boosting_weight * val.exp();
                        }
            */
            ////////////////////////////////////////// TEST END
            // get predictor error

            let mut reg_w_sum = 0.0;
            let weighted_error_count: f64 = antigens
                .iter()
                .map(|antigen| {
                    let pred = is_class_correct(&antigen, &antibodies, &params.eval_method);
                    return if let Some(is_corr) = pred {
                        reg_w_sum += antigen.boosting_weight;
                        if is_corr {
                            0.0
                        } else {
                            1.0 * antigen.boosting_weight
                        }
                    } else {
                        // 0.1 * antigen.boosting_weight
                        // 0.0 * antigen.boosting_weight
                        0.0
                    };
                })
                .sum();

            let weight_sum: f64 = antigens.iter().map(|ag| ag.boosting_weight).sum();
            println!(
                "Error count {:?} reg_w sum {:?}, full_w sum {:?}",
                weighted_error_count, reg_w_sum, weight_sum
            );
            let mut error = weighted_error_count / reg_w_sum;

            // calculate alpha weight for model
            let alpha_m = if !error.is_finite() || weighted_error_count == 0.0 {
                println!("hit!");
                1.0
            } else {
                ((1.0 - error) / error).ln()
            };

            println!(
                "Error is {:?}, Alpha is {:?}, exp alpha is {:}",
                error,
                alpha_m,
                alpha_m.exp()
            );

            if alpha_m <= 0.0 {
                println!(
                    "Alpha is negative ({:.2?}) error: {:.2?} discarding round {:} retry {:}",
                    alpha_m, error, current_boost_round, current_retries
                );
                current_retries += 1;
                continue;
            }

            antibodies
                .iter_mut()
                .for_each(|ab| ab.boosting_model_alpha = ab.final_train_label_affinity.unwrap().0);
            // .for_each(|ab| ab.boosting_model_alpha = alpha_m);

            ab_pool.extend(antibodies);

            let mut n_corr = 0;
            let mut n_wrong = 0;
            let mut n_no_reg = 0;

            // // update weights
            for antigen in &mut antigens {
                let pred = is_class_correct(&antigen, &ab_pool, &params.eval_method);

                // let is_corr = is_class_correct(&antigen, &ab_pool).unwrap_or(false);
                let val: f64 = if let Some(is_corr) = pred {
                    if is_corr {
                        n_corr += 1;
                        0.0
                    } else {
                        n_wrong += 1;
                        // 1.5
                        // 1.0

                        alpha_m
                    }
                } else {
                    n_no_reg += 1;
                    // 0.0
                    // 1.1
                    alpha_m
                    // alpha_m
                };

                antigen.boosting_weight = antigen.boosting_weight * val.exp();
            }

            let boost_weight_avg = antigens.iter().map(|ag| ag.boosting_weight).mean();
            println!("Bosting weight mean: {:?}", boost_weight_avg);

            if logger.should_run(ExperimentProperty::BoostAccuracy) {
                logger.log_prop(
                    ExperimentProperty::BoostAccuracy,
                    LoggedValue::gen_corr_wrong_no_reg(n_corr, n_wrong, n_no_reg),
                )
            }

            if logger.should_run(ExperimentProperty::BoostAccuracyTest) {

                let (n_cor, n_wrong, n_no_detect) = get_pop_acc(highly_dubious_practices, &ab_pool, params);

                logger.log_prop(
                    ExperimentProperty::BoostAccuracyTest,
                    LoggedValue::gen_corr_wrong_no_reg(n_corr, n_wrong, n_no_reg),
                )
            }

            // save the antibodies

            if verbosity_params.print_boost_info {
                println!("\n#\n# for boost round {:?}:\n#", current_boost_round);
                println!("current ab pool s {:?}:", ab_pool.len());

                // println!("last model error: {:<3.3?} alpha {:?}:", error, alpha_m);

                let mut _ais = ArtificialImmuneSystem::new();
                _ais.antibodies = ab_pool.clone();
                let train_acc = eval_display(
                    &antigens,
                    &_ais,
                    &translator,
                    "TRAIN".to_string(),
                    false,
                    Some(&params.eval_method),
                );
                let test_acc = eval_display(
                    &highly_dubious_practices,
                    &_ais,
                    &translator,
                    "TEST".to_string(),
                    false,
                    Some(&params.eval_method),
                );

                train_boost_hist.push(train_acc);
                test_boost_hist.push(test_acc);
                let duration = start.elapsed();
                // println!(
                //     "Total runtime: {:?}, \nPer iteration: {:?}",
                //     duration,
                //     duration.as_nanos() / params.generations as u128
                // );

                /*    println!("train");
                evaluate_print_population(&antigens, &ab_pool, &params.eval_method,logger);

                println!("test");
                evaluate_print_population(&highly_dubious_practices, &ab_pool,&params.eval_method, logger);

                println!("articles");
                let translator_formatted = input_antigens.iter().chain(highly_dubious_practices).map(|ag| {
                    let pred_class = is_class_correct_with_membership(&ag, &ab_pool);
                    return if let Some(v) = pred_class {
                        if v {
                            (Some(true), ag)
                        } else {
                            (Some(false), ag)
                        }
                    } else {
                        (None, ag)
                    }
                }).collect();

                translator.get_show_ag_acc(translator_formatted, false);*/
            }

            current_boost_round += 1;
        }
        self.antibodies = ab_pool.clone();

        let mut match_counter = MatchCounter::new(&antigens);
        let evaluated_pop = evaluate_population(params, ab_pool, &antigens);
        match_counter.add_evaluations(&evaluated_pop);

        let scored_pop = score_antibodies(params, evaluated_pop, &match_counter);

        // println!(
        //     "########## boost mask \n{:?}",
        //     match_counter.boosting_weight_values
        // );

        let out_pop = scored_pop
            .into_iter()
            .map(|(a, b)| (a, b.evaluation, b.antibody))
            .collect();
        return (train_boost_hist, test_boost_hist, out_pop);
    }

    fn train_ab_set(
        &self,
        antigens: &Vec<AntiGen>,
        params: &Params,
        verbosity_params: &VerbosityParams,
        pop_size: usize,
        logger: &mut ExperimentLogger,
        highly_dubious_practices: &Vec<AntiGen>
    ) -> (Vec<f64>, Vec<f64>, Vec<(f64, Evaluation, Antibody)>) {
        logger.init_train();
        let t0 = Instant::now();
        // =======  init misc training params  ======= //

        let leak_size = (antigens.len() as f64 * params.leak_fraction) as usize;

        let mut rng = rand::thread_rng();

        // check dims and classes
        let n_dims = antigens.get(0).unwrap().values.len();

        // keeps track of the matches
        let max_ag_id = antigens.iter().max_by_key(|ag| ag.id).unwrap().id;
        let mut match_counter = MatchCounter::new(antigens);

        // init hist and watchers
        let mut best_run: Vec<(f64, Evaluation, Antibody)> = Vec::new();
        let mut best_score = 0.0;
        let mut train_score_hist = Vec::new();
        let mut train_acc_hist = Vec::new();

        // make the cell factory
        let cell_factory = AntibodyFactory::new(
            params,
            n_dims,
            Vec::from_iter(match_counter.class_labels.clone().into_iter()),
        );

        // =======  set up population  ======= //
        let initial_population: Vec<Antibody> =
            gen_initial_population(antigens, params, &cell_factory, &pop_size, &match_counter);


        // the evaluated pop is the population where the ab -> ag matches has been calculated but not scored
        let mut evaluated_pop: Vec<EvaluatedAntibody> = Vec::with_capacity(pop_size);

        // the scored pop is the population where the ab -> ag matches and score has been calculated
        let mut scored_pop: Vec<(f64, EvaluatedAntibody)> = Vec::with_capacity(pop_size);

        evaluated_pop = evaluate_population(params, initial_population, antigens);
        match_counter.add_evaluations(&evaluated_pop);

        scored_pop = score_antibodies(params, evaluated_pop, &match_counter);

        //
        if verbosity_params.show_initial_pop_info {
            println!("initial");
            match_counter
                .class_labels
                .clone()
                .into_iter()
                .for_each(|cl| {
                    let filtered: Vec<usize> = scored_pop
                        .iter()
                        .filter(|(_, c)| c.antibody.class_label == cl)
                        .map(|_| 1usize)
                        .collect();
                    print!("num with {:?} is {:?} ", cl, filtered.len())
                });
            println!("\ninitial end");
        }

        for i in 0..params.generations {
            // =======  tracking and logging   ======= //
            let max_score = scored_pop
                .iter()
                .map(|(score, _)| score)
                .max_by(|a, b| a.total_cmp(b))
                .unwrap();

            let avg_score =
                scored_pop.iter().map(|(score, _)| score).sum::<f64>() / scored_pop.len() as f64;

            if let Some(n) = verbosity_params.iter_info_interval {
                if i % n == 0 {
                    println!(
                        "iter: {:<5} avg score {:.6}, max score {:.6}, last acc {:.6}",
                        i,
                        avg_score,
                        max_score,
                        train_acc_hist.last().unwrap_or(&0.0)
                    );
                    println!("pop size {:} ", scored_pop.len());
                }
            } else {
                print!("iter: {:<5} of {:<5}\r", i, params.generations,);
                io::stdout().flush().unwrap();
            }

            train_score_hist.push(avg_score.clone());


            if logger.should_run(ExperimentProperty::AvgTrainScore) {
                logger.log_prop(
                    ExperimentProperty::AvgTrainScore,
                    LoggedValue::SingleValue(avg_score),
                )
            }



            if logger.should_run(ExperimentProperty::PopLabelMemberships) {
                let label_membership_vals: HashMap<usize, usize> = match_counter
                    .class_labels
                    .clone()
                    .into_iter()
                    .map(|cl| {
                        let filtered: Vec<usize> = scored_pop
                            .iter()
                            .filter(|(a, b)| b.antibody.class_label == cl)
                            .map(|(a, b)| 1usize)
                            .collect();
                        // print!("num with {:?} is {:?} ", cl, filtered.len());
                        (cl, filtered.len())
                    })
                    .collect();

                logger.log_prop(
                    ExperimentProperty::PopLabelMemberships,
                    LoggedValue::LabelMembership(label_membership_vals),
                )
            }

            if logger.should_run(ExperimentProperty::PopDimTypeMemberships){
                let mut dim_type_map: HashMap<DimValueType, f64> = HashMap::from([(DimValueType::Open,0.0), (DimValueType::Disabled,0.0), (DimValueType::Circle,0.0)]);
                scored_pop.iter().for_each(|(_,ab)| {
                    for dim_value in &ab.antibody.dim_values {
                        if let Some(v) = dim_type_map.get_mut(&dim_value.value_type) {
                            *v += 1.0/n_dims as f64;
                        }
                    }
                }
                );

                dim_type_map.iter_mut().for_each(|(k,v)|{
                    *v /= scored_pop.len() as f64;
                });

                logger.log_prop(ExperimentProperty::PopDimTypeMemberships,LoggedValue::DimTypeMembership(dim_type_map) )
            }

            // =======  parent selection  ======= //
            let mut n_to_replace = match &params.replace_frac_type {
                ReplaceFractionType::Linear(range) => {
                    let range_span = range.end - range.start;
                    let round_frac = i as f64 / params.generations as f64;
                    (pop_size as f64 * (range.start + (range_span * round_frac))).ceil() as usize
                }
                ReplaceFractionType::MaxRepFrac(max_frac) => {
                    let replace_exponent =
                        (3.0 / 2.0) * (((i as f64) + 1.0) / params.generations as f64);
                    let replace_frac = max_frac * (2.0 / pop_size as f64).powf(replace_exponent);
                    (pop_size as f64 * replace_frac).ceil() as usize
                }
            };

            // params.max_replacment_frac * (2.0 / pop_size as f64).powf(replace_exponent) + 0.05;

            // =======  clone -> mut -> eval  ======= //
            let mut new_gen: Vec<EvaluatedAntibody> = Vec::new();

            let mut parent_idx_vec: Vec<usize> = Vec::new();

            let pre = scored_pop.iter().map(|(a, b)| a).sum::<f64>();


            // println!("scored pre {:?}", scored_pop.iter().map(|(score, _)| score).sum::<f64>());
            // calculate and preform the replacement selection/mutation for each ab label separately to maintain the label ratios
            for (label, fraction) in &match_counter.frac_map.clone() {
                let replace_count_for_label = (n_to_replace as f64 * fraction).ceil() as usize;

                if replace_count_for_label <= 0
                // || (*match_counter.count_map.get(label).unwrap() < replace_count_for_label)
                {
                    continue;
                }

                let parents = labeled_tournament_pick(
                    &scored_pop,
                    &replace_count_for_label,
                    &params.tournament_size,
                    Some(label),
                );

                parent_idx_vec.extend(parents.clone());

                let parents_eval: Vec<_> = parents
                    .iter()
                    .map(|idx| {
                        let (score, eval_ab) = scored_pop.get(*idx).unwrap();
                        eval_ab.evaluation.clone()
                    })
                    .collect();

                let label_gen: Vec<_> = parents
                    .clone()
                    .into_par_iter() // TODO: set parallel
                    .map(|idx| {
                        let (score, mut eval_ab) = scored_pop.get(idx).unwrap().clone();

                        let frac_of_max = (score / max_score).max(0.2).min(1.0);
                        let n_clones =
                            ((params.n_parents_mutations as f64 * frac_of_max) as usize).max(1);
                        let fitness_scaler = 1.0 - frac_of_max;

                        // println!();
                        // println!("score pre {:}", score_antibody(&eval_ab, &params, &match_counter));

                        let best_score = mutate_clone_transform(
                            &mut eval_ab,
                            score,
                            params,
                            fitness_scaler,
                            antigens,
                            &match_counter,
                            n_clones,
                        );

                        if best_score < score {
                            println!("\nISSUE")
                        }

                        // println!("score post {:}",  score_antibody(&eval_ab, &params, &match_counter));
                        return eval_ab;
                    })
                    .collect();

                parents_eval
                    .iter()
                    .for_each(|ev| match_counter.remove_evaluation(&ev));
                match_counter.add_evaluations(&label_gen);

                new_gen.extend(label_gen)
            }

            let new_gen_scored = score_antibodies(params, new_gen, &match_counter);

            // filter to the n best new antigens
            // let mut to_add = pick_best_n(new_gen_scored, n_to_replace);
            let mut to_add = new_gen_scored;

            // =======  selection  ======= //
            let mut from_sum = 0.0;
            let mut to_sum = 0.0;

            // let mut id_vec = parent_idx_vec.clone();
            // id_vec.sort();
            // let l1 = id_vec.len();
            // id_vec.dedup();
            // let l2 = id_vec.len();
            // println!("pre dedupe {:?}, post dedupe {:?}", l1, l2);

            // let mut from_list: Vec<_> = Vec::new();
            // let mut to_list: Vec<_> = Vec::new();
            // println!("scored post {:?}", scored_pop.iter().map(|(score, _)| score).sum::<f64>());
            for idx in parent_idx_vec {
                // let mut parent_value = scored_pop.get_mut(idx).unwrap();
                let (p_score, p_eab) = scored_pop.get_mut(idx).unwrap();
                let (c_score, c_eab) = to_add.pop().unwrap();

 /*               from_list.push((p_score.clone(),p_eab.clone()));
                to_list.push((c_score.clone(),c_eab.clone()));
                from_sum += *p_score;
                to_sum += c_score;*/
                std::mem::replace(p_score, c_score);
                std::mem::replace(p_eab, c_eab);
            }
      /*      println!("from sum: {:?},  to sum: {:?}", from_sum, to_sum);
            if  from_sum > to_sum{
                println!("FROM LIST:");

                from_list.iter().for_each(|(s,eab)|{
                    println!("label {:?}, num cor {:?}  num err {:?} score {:?}, dims {:?}", eab.antibody.class_label, eab.evaluation.matched_ids.len(), eab.evaluation.wrongly_matched.len(), s, eab.antibody.dim_values);
                });

                println!("TO LIST:");
                to_list.iter().for_each(|(s,eab)|{
                    println!("label {:?}, num cor {:?}  num err {:?} score {:?}, dims {:?}", eab.antibody.class_label, eab.evaluation.matched_ids.len(), eab.evaluation.wrongly_matched.len(), s, eab.antibody.dim_values);
                });
                // println!("from ag: {:?}", from_list.iter().map(|eag| eag.antibody.clone()).collect::<Vec<_>>());
                // println!("to ag:   {:?}", to_list.iter().map(|eag| eag.antibody.clone()).collect::<Vec<_>>());
            }*/

            /*  let post = scored_pop.iter().map(|(a,b)| a).sum::<f64>();
            if pre > post{

                println!("pre  {:?}", pre);
                println!("post {:?}", post);
            }*/

            //
            // =======  leak new  ======= //
            // let is_strip_round = i % 100 == 0;
            let is_strip_round = false;

            let n_to_leak = ((n_to_replace as f64 * params.leak_fraction) as usize);
            let mut new_leaked: Vec<(f64, EvaluatedAntibody)> = Vec::new();

            let n_to_gen_map = {
                let mut replace_map = HashMap::new();

                for (label, fraction) in &match_counter.frac_map {
                    let replace_count_for_label = (n_to_leak as f64 * fraction).ceil() as usize;
                    replace_map.insert(label.clone(), replace_count_for_label);
                }
                replace_map
            };

            if params.leak_fraction > 0.0 {
                let inversed = match_counter.get_inversed_correct_match_counts();

                let mut local_match_counter = match_counter.clone();

                for (label, count) in &n_to_gen_map {

                    let mut rng = rand::thread_rng();


                    if *count <= 0 {
                        continue;
                    }

                    let filtered: Vec<_> = if params.ratio_lock{
                        antigens
                            .iter()
                            .filter(|ag| ag.class_label == *label)
                            .map(|ag| (ag, inversed.get(ag.id).unwrap_or(&0).clone() as i32)) // todo: validate that this unwrap or does not hide a bug
                            .collect()
                    } else {
                        antigens
                            .iter()
                            .map(|ag| (ag, inversed.get(ag.id).unwrap_or(&0).clone() as i32)) // todo: validate that this unwrap or does not hide a bug
                            .collect()
                    };

                    if filtered.len() == 0{
                        println!("\n\nno elm in label continuing");
                        continue
                    }

                    let mut new_pop_pop: Vec<_> = (0..*count)
                        .into_par_iter()
                        .map(|_| {
                            let mut rng = rand::thread_rng();
                            let new_ag = filtered
                                .choose_weighted(&mut rng, |v| v.1 + 1)
                                .unwrap()
                                .0
                                .clone();
                            let mut new_antibody = if rng.gen_bool(1.0 - params.leak_rand_prob) {
                                cell_factory.generate_from_antigen(&new_ag)
                            } else {
                                cell_factory.generate_random_genome_with_label(new_ag.class_label)
                                // let mut new_ab = cell_factory.generate_random_genome_with_label(label);
                                //     expand_antibody_radius_until_hit(new_ab, &antigens)
                            };

                            if params.antibody_init_expand_radius {
                                new_antibody =
                                    expand_antibody_radius_until_hit(new_antibody, &antigens)
                            }
                            let eval = evaluate_antibody(antigens, &new_antibody);
                            return EvaluatedAntibody {
                                evaluation: eval,
                                antibody: new_antibody,
                            };
                        })
                        .collect();

                    local_match_counter.add_evaluations(&new_pop_pop);
                    let leaked_to_add = score_antibodies(params, new_pop_pop, &local_match_counter);

                    // shit but works
                    leaked_to_add.iter().for_each(|(a, b)| {
                        local_match_counter.remove_evaluation(&b.evaluation);
                    });

                    new_leaked.extend(leaked_to_add);
                }
            }

            if new_leaked.len() > 0 {
                if params.ratio_lock {
                    scored_pop = replace_if_better_per_cat(
                        scored_pop,
                        new_leaked,
                        n_to_gen_map,
                        &mut match_counter,
                    );
                } else {
                    scored_pop = replace_if_better(
                        scored_pop,
                        new_leaked,
                        n_to_gen_map,
                        &mut match_counter,
                    );
                }
            }

            // =======  next gen cleanup  ======= //

            let evaluated_pop: Vec<EvaluatedAntibody> =
                scored_pop.into_iter().map(|(a, b)| b).collect();

            // let mut match_counter_2 = MatchCounter::new(max_ag_id);
            // match_counter_2.add_evaluations(
            //     evaluated_pop
            //         .iter()
            //         .map(|(evaluation, _)| evaluation)
            //         .collect::<Vec<_>>(),
            // );
            //
            // let neg_ok = match_counter
            //     .incorrect_match_counter
            //     .eq(&match_counter_2.incorrect_match_counter);
            // let pos_ok = match_counter
            //     .correct_match_counter
            //     .eq(&match_counter_2.correct_match_counter);
            //
            // if !(neg_ok & pos_ok) {
            //     panic!("error with match counter updates ")
            // }

            scored_pop = score_antibodies(params, evaluated_pop, &match_counter);



            if logger.should_run(ExperimentProperty::TestAccuracy) | logger.should_run(ExperimentProperty::TrainAccuracy) {

                let eval_pop: Vec<_> = scored_pop
                    .iter()
                    .map(|(score, evaluated_ab)| {
                        let mut save_ab = evaluated_ab.antibody.clone();
                        save_ab.final_train_label_membership =
                            Some(evaluated_ab.evaluation.membership_value);
                        save_ab.final_train_label_affinity =
                            Some(evaluated_ab.evaluation.affinity_weight);
                        return (*score, evaluated_ab.evaluation.clone(), save_ab);
                    })
                    .filter(|(score, _, cell)| {
                        cell.final_train_label_membership.unwrap().0 >= params.membership_required
                    })
                    .collect();
                let antibodies: Vec<Antibody> =
                    eval_pop.iter().map(|(a, b, c)| c.clone()).collect();

                if logger.should_run(ExperimentProperty::TrainAccuracy){
                    let (n_corr, n_wrong, n_no_detect) = get_pop_acc(antigens, &antibodies, params);
                    logger.log_prop(
                        ExperimentProperty::TrainAccuracy,
                        LoggedValue::gen_corr_wrong_no_reg(n_corr, n_wrong, n_no_detect)
                    )
                };
                if logger.should_run(ExperimentProperty::TestAccuracy) {
                    let (n_corr, n_wrong, n_no_detect) = get_pop_acc(highly_dubious_practices, &antibodies, params);
                    logger.log_prop(
                        ExperimentProperty::TestAccuracy,
                        LoggedValue::gen_corr_wrong_no_reg(n_corr, n_wrong, n_no_detect)
                    )

                }
            }


            if let Some(n) = verbosity_params.full_pop_acc_interval {
                let eval_pop: Vec<_> = scored_pop
                    .iter()
                    .map(|(score, evaluated_ab)| {
                        let mut save_ab = evaluated_ab.antibody.clone();
                        save_ab.final_train_label_membership =
                            Some(evaluated_ab.evaluation.membership_value);
                        save_ab.final_train_label_affinity =
                            Some(evaluated_ab.evaluation.affinity_weight);
                        return (*score, evaluated_ab.evaluation.clone(), save_ab);
                    })
                    .filter(|(score, _, cell)| {
                        cell.final_train_label_membership.unwrap().0 >= params.membership_required
                    })
                    .collect();

                if i % n == 0 {
                    let antibodies: Vec<Antibody> =
                        eval_pop.iter().map(|(a, b, c)| c.clone()).collect();

        /*            let mut n_corr = 0;
                    let mut n_wrong = 0;
                    let mut n_no_detect = 0;
                    for antigen in antigens {
                        let pred = is_class_correct(&antigen, &antibodies, &params.eval_method);
                        if let Some(v) = pred {
                            if v {
                                n_corr += 1
                            } else {
                                n_wrong += 1
                            }
                        } else {
                            n_no_detect += 1
                        }
                    }*/

                    let (n_corr, n_wrong, n_no_detect) = get_pop_acc(antigens, &antibodies, params);


                    let avg_acc = n_corr as f64 / antigens.len() as f64;
                    // let avg_score = scored_pop.iter().map(|(a, _b, _)| a).sum::<f64>() / scored_pop.len() as f64;
                    if avg_acc >= best_score {
                        best_score = avg_acc;
                        best_run = eval_pop.clone();
                    }
                    train_acc_hist.push(avg_acc);


                } else {
                    let last = train_acc_hist.last().unwrap_or(&0.0);
                    train_acc_hist.push(last.clone());
                }
            } else {
                let last = train_acc_hist.last().unwrap_or(&0.0);
                train_acc_hist.push(last.clone());
            }

            if verbosity_params.show_class_info {
                println!("replacing {:} leaking {:}", n_to_replace, n_to_leak);
                match_counter
                    .class_labels
                    .clone()
                    .into_iter()
                    .for_each(|cl| {
                        let filtered: Vec<usize> = scored_pop
                            .iter()
                            .filter(|(a, b)| b.antibody.class_label == cl)
                            .map(|(a, b)| 1usize)
                            .collect();
                        print!("num with {:?} is {:?} ", cl, filtered.len())
                    });
                println!();
            }

            logger.iter_step();
        }

        // println!("########## error mask \n{:?}", error_match_mask);
        // println!("########## match mask \n{:?}", match_mask);
        // scored_pop = best_run;

        // scored_pop = snip_worst_n(scored_pop, 10);
        // let (scored_pop, _drained) = elitism_selection(scored_pop, &100);

        if false {
            println!(
                "########## match mask \n{:?}",
                match_counter.correct_match_counter
            );
            // let (removed_map, stripped_pop) =
            // remove_strictly_worse(scored_pop, &mut match_mask, &mut error_match_mask, None);
            // scored_pop = stripped_pop;
            // println!("stripped count {:?}", removed_map);
            println!(
                "########## match mask \n{:?}",
                match_counter.correct_match_counter
            );
        }

        let splitted = scored_pop
            .into_iter()
            // .filter(|(score, _, _)| *score >= 2.0)
            // .filter(|(score, _, _)| *score > 0.0)
            .map(|(score, evaluated_ab)| {
                let mut save_ab = evaluated_ab.antibody.clone();
                save_ab.final_train_label_membership =
                    Some(evaluated_ab.evaluation.membership_value);
                save_ab.final_train_label_affinity = Some(evaluated_ab.evaluation.affinity_weight);
                return (score, evaluated_ab.evaluation.clone(), save_ab);
            })
            .filter(|(score, _, cell)| {
                cell.final_train_label_membership.unwrap().0 >= params.membership_required
            })
            .collect();

        logger.end_train();

        let duration = t0.elapsed();
        if logger.should_run(ExperimentProperty::Runtime){
            logger.log_prop(ExperimentProperty::Runtime, LoggedValue::SingleValue(duration.as_secs_f64()))
        }
        return (train_acc_hist, train_score_hist, splitted);
    }

    pub fn print_ab_mut_info(&self) {
        let mut clone_count = 0;

        let mut rand_init = 0;
        let mut ag_init = 0;

        let mut mut_map: HashMap<MutationType, usize> = HashMap::new();
        self.antibodies.iter().for_each(|ab| {
            clone_count += ab.clone_count;
            match ab.init_type {
                InitType::Random => rand_init += 1,
                InitType::Antibody => ag_init += 1,
                InitType::NA => {}
            };
            for (k, count) in &ab.mutation_counter {
                if let Some(v) = mut_map.get_mut(k) {
                    *v += count;
                } else {
                    mut_map.insert(k.clone(), *count);
                };
            }
        });

        let clone_count_mean = clone_count as f64 / self.antibodies.len() as f64;
        println!("avg clone count: {:.4?}", clone_count_mean);
        println!("avg rand inits : {:.4?}", rand_init);
        println!("avg ag inits   : {:.4?}", ag_init);

        println!("\nMut types:");
        println!("{:?}", mut_map);
        mut_map.iter().for_each(|(mut_t, cnt)| {
            println!(
                "mut type {:<5} count: {:.2?}",
                mut_t.to_string(),
                *cnt as f64 / self.antibodies.len() as f64
            );
        });
    }

    pub fn is_class_correct(
        &self,
        antigen: &AntiGen,
        eval_method: &EvaluationMethod,
    ) -> Option<bool> {
        return is_class_correct(antigen, &self.antibodies, eval_method);
    }

    pub fn make_prediction(&self, antigen: &AntiGen, eval_method: &EvaluationMethod) -> Prediction {
        return make_prediction(antigen, &self.antibodies, eval_method);
    }

    pub fn print_current_ab(&self) {
        let mut ab = self.antibodies.clone();
        ab.sort_by(|a, b| {
            a.dim_values
                .first()
                .unwrap()
                .multiplier
                .total_cmp(&b.dim_values.first().unwrap().multiplier)
        });
        ab.iter().for_each(|ag| {
            println!("{:?}", ag);
        });
    }
}
