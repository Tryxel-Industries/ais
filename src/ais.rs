use json::object;
use std::collections::{HashMap, HashSet};
use std::iter::Map;
use std::ops::{Range, RangeInclusive};
use std::io;
use std::io::Write;

use rand::prelude::{IteratorRandom, SliceRandom};
use rand::Rng;
use rayon::prelude::*;
use statrs::statistics::Statistics;

use crate::bucket_empire::BucketKing;
use crate::evaluation::{evaluate_antibody, Evaluation, MatchCounter};
use crate::mutate;
use crate::params::{Params, PopSizeType, ReplaceFractionType, VerbosityParams};
use crate::representation::antibody::Antibody;
use crate::representation::antibody_factory::AntibodyFactory;
use crate::representation::antigen::AntiGen;
use crate::representation::expand_antibody_radius_until_hit;
use crate::representation::news_article_mapper::NewsArticleAntigenTranslator;
use crate::scoring::score_antibodies;
use crate::selection::{
    elitism_selection, kill_by_mask_yo, labeled_tournament_pick, pick_best_n,
    remove_strictly_worse, replace_if_better_per_cat, replace_worst_n_per_cat, snip_worst_n,
    tournament_pick,
};

/*
1. select n parents
2. clone -> mutate parents n times
 */
pub struct ArtificialImmuneSystem {
    pub antibodies: Vec<Antibody>,
}

pub struct ClassPrediction {
    pub class: usize,
    pub reg_count: usize,
    pub valid_count: usize,
    pub membership_sum: f64,
    pub valid_membership_sum: f64,
}
pub struct Prediction {
    pub predicted_class: usize,
    pub class_predictions: Vec<ClassPrediction>,
}

//
//  AIS
//

pub fn evaluate_population(
    params: &Params,
    population: Vec<Antibody>,
    antigens: &Vec<AntiGen>,
) -> Vec<(Evaluation, Antibody)> {
    return population
        .into_par_iter() // TODO: set paralell
        // .into_iter()
        .map(|antibody| {
            // evaluate antibodies
            let score = evaluate_antibody(antigens, &antibody);
            return (score, antibody);
        })
        .collect();
}

fn gen_initial_population(
    bucket_king: &BucketKing<AntiGen>,
    antigens: &Vec<AntiGen>,
    params: &Params,
    cell_factory: &AntibodyFactory,
    pop_size: &usize,
    match_counter: &MatchCounter,
) -> Vec<Antibody> {
    let mut rng = rand::thread_rng();
    return if (*pop_size == antigens.len()) {
        antigens
            .par_iter()
            .map(|ag| cell_factory.generate_from_antigen(ag))
            .map(|cell| {
                if params.antibody_init_expand_radius {
                    expand_antibody_radius_until_hit(cell, &bucket_king, &antigens)
                } else {
                    cell
                }
            })
            .collect()
    } else {
        match_counter.frac_map.iter().flat_map(|(label,frac)| {
            let replace_count_for_label = (*pop_size as f64 * frac).ceil() as usize;

            let filtered: Vec<_> = antigens
                .iter()
                .filter(|ag| ag.class_label == *label)
                .collect();

            return (0..*pop_size)
            .par_bridge()
            .map(|_| cell_factory.generate_from_antigen(filtered.choose(&mut rand::thread_rng()).unwrap()))
            .map(|cell| {
                if params.antibody_init_expand_radius {
                    expand_antibody_radius_until_hit(cell, &bucket_king, &antigens)
                } else {
                    cell
                }
            })
            .collect::<Vec<_>>()
        }).collect()


        // (0..*pop_size)
        //     .par_bridge()
        //     .map(|_| cell_factory.generate_from_antigen(antigens.choose(&mut rand::thread_rng()).unwrap()))
        //     .map(|cell| {
        //         if params.antibody_init_expand_radius {
        //             expand_antibody_radius_until_hit(cell, &bucket_king, &antigens)
        //         } else {
        //             cell
        //         }
        //     })
        //     .collect()
    };
}

pub fn is_class_correct(antigen: &AntiGen, antibodies: &Vec<Antibody>) -> Option<bool> {
    let matching_cells = antibodies
        .par_iter()
        .filter(|antibody| antibody.test_antigen(antigen))
        .collect::<Vec<_>>();

    if matching_cells.len() == 0 {
        return None;
    }

    let class_true = matching_cells
        .iter()
        .filter(|x| x.class_label == antigen.class_label)
        .collect::<Vec<_>>();
    let class_false = matching_cells
        .iter()
        .filter(|x| x.class_label != antigen.class_label)
        .collect::<Vec<_>>();

    if class_true.len() > class_false.len() {
        return Some(true);
    } else {
        // println!("wrong match id {:?}, cor: {:?}  incor {:?}", antigen.id, class_true.len(), class_false.len());
        return Some(false);
    }
}

pub fn make_prediction(antigen: &AntiGen, antibodies: &Vec<Antibody>) -> Option<Prediction> {
    let matching_cells = antibodies
        .par_iter()
        .filter(|antibody| antibody.test_antigen(antigen))
        .collect::<Vec<_>>();

    let matching_classes = matching_cells
        .iter()
        .map(|ab| ab.class_label)
        .collect::<HashSet<usize>>();

    let mut label_membership_values = matching_classes
        .iter()
        .map(|class| {
            (
                class.clone(),
                ClassPrediction {
                    class: class.clone(),
                    reg_count: 0,
                    membership_sum: 0.0,
                    valid_count: 0,
                    valid_membership_sum: 0.0,
                },
            )
        })
        .collect::<HashMap<usize, ClassPrediction>>();

    for cell in matching_cells {
        let mut match_class_pred = label_membership_values.get_mut(&cell.class_label).unwrap();
        match_class_pred.membership_sum += cell.final_train_label_membership.unwrap().0;
        match_class_pred.reg_count += 1;

        // if cell.final_train_label_membership.unwrap().0 >= 0.95{
        if cell.final_train_label_membership.unwrap().0 >= 0.0 {
            match_class_pred.valid_count += 1;
            // match_class_pred.valid_membership_sum += cell.final_train_label_membership.unwrap().0 * cell.boosting_model_alpha;
            match_class_pred.valid_membership_sum += 1.0 * cell.boosting_model_alpha;
        }
    }

    let mut class_preds = Vec::new();

    let mut best_class = None;
    let mut best_score = 0.0;
    for (label, label_membership_value) in label_membership_values {
        if best_score < label_membership_value.valid_membership_sum {
            best_class = Some(label);
            best_score = label_membership_value.valid_membership_sum.clone()
        }
        class_preds.push(label_membership_value)
    }

    return Some(Prediction {
        predicted_class: best_class?,
        class_predictions: class_preds,
    });
}

pub fn is_class_correct_with_membership(
    antigen: &AntiGen,
    antibodies: &Vec<Antibody>,
) -> Option<bool> {
    let pred_value = make_prediction(antigen, antibodies)?;

    if pred_value.predicted_class == antigen.class_label {
        return Some(true);
    } else {
        // println!("wrong match id {:?}, cor: {:?}  incor {:?}", antigen.id, class_true.len(), class_false.len());
        return Some(false);
    }
}

fn evaluate_print_population(antigens: &Vec<AntiGen>, antibodies: &Vec<Antibody>) {
    let mut test_n_corr = 0;
    let mut test_n_wrong = 0;

    let mut membership_test_n_corr = 0;
    let mut membership_test_n_wrong = 0;

    let mut test_per_class_corr = HashMap::new();
    let mut test_n_no_detect = 0;
    for antigen in antigens {
        let pred_class = is_class_correct_with_membership(&antigen, antibodies);

        if let Some(v) = pred_class {
            if v {
                membership_test_n_corr += 1;
            } else {
                membership_test_n_wrong += 1
            }
        }
        let pred_class = is_class_correct(&antigen, antibodies);
        if let Some(v) = pred_class {
            if v {
                test_n_corr += 1;
                let class_count = test_per_class_corr.get(&antigen.class_label).unwrap_or(&0);
                test_per_class_corr.insert(antigen.class_label, *class_count + 1);
            } else {
                test_n_wrong += 1
            }
        } else {
            test_n_no_detect += 1
        }
    }

    let test_acc = test_n_corr as f64 / (antigens.len() as f64);
    let test_precession = test_n_corr as f64 / (test_n_wrong as f64 + test_n_corr as f64).max(1.0);
    let membership_test_acc = membership_test_n_corr as f64 / (antigens.len() as f64);
    let membership_test_precession = membership_test_n_corr as f64
        / (membership_test_n_wrong as f64 + membership_test_n_corr as f64).max(1.0);

    println!(
        "without membership: corr {:>2?}, false {:>3?}, no_detect {:>3?}, presission: {:>2.3?}, frac: {:2.3?}",
        test_n_corr, test_n_wrong, test_n_no_detect,test_precession, test_acc
    );
    println!(
        "with membership:    corr {:>2?}, false {:>3?}, no_detect {:>3?}, presission: {:>2.3?}, frac: {:2.3?}",
        membership_test_n_corr, membership_test_n_wrong, antigens.len()-(membership_test_n_corr+membership_test_n_wrong), membership_test_precession, membership_test_acc
    );

}
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
    ) -> (Vec<f64>, Vec<f64>, Vec<(f64, Evaluation, Antibody)>) {

        let pop_size = match params.antigen_pop_size {
            PopSizeType::Fraction(fraction ) => { (antigens.len() as f64 * fraction) as usize}
            PopSizeType::Number(n) => {n}
        };

        let (train_acc_hist, train_score_hist, scored_pop) =
            self.train_ab_set(antigens, params, verbosity_params, pop_size);
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
        translator: &NewsArticleAntigenTranslator
    ) -> (Vec<f64>, Vec<f64>, Vec<(f64, Evaluation, Antibody)>) {
        let mut antigens = input_antigens.clone();


        let pop_size = match params.antigen_pop_size {
            PopSizeType::Fraction(fraction ) => { (antigens.len() as f64 * fraction) as usize}
            PopSizeType::Number(n) => {n}
        };

        let train_per_round = ((pop_size as f64) * 1.0 / boosting_rounds as f64) as usize;

        println!(
            "pop size, final num ab {:?} gained throgh {:?} rounds with {:?}",
            pop_size, boosting_rounds, train_per_round
        );



        let mut ab_pool: Vec<Antibody> = Vec::new();

        for n in 0..boosting_rounds {
            // using https://towardsdatascience.com/boosting-algorithms-explained-d38f56ef3f30

            // norm weights
            let ag_weight_sum: f64 = antigens.iter().map(|ag| ag.boosting_weight).sum();
            // println!("sum is {:?},", antigens.iter().map(|ag| ag.boosting_weight).sum::<f64>());
            antigens
                .iter_mut()
                .for_each(|ag| ag.boosting_weight = ag.boosting_weight / ag_weight_sum);

            // println!("sum is {:?},", antigens.iter().map(|ag| ag.boosting_weight).sum::<f64>());

        //   println!();
        //     let mut match_counter = MatchCounter::new(&antigens);
        //     println!(
        //     "########## boost mask \n{:?}",
        //     match_counter.boosting_weight_values
        // );

            // train predictor
            let (train_acc_hist, train_score_hist, scored_pop) =
                self.train_ab_set(&antigens, params, verbosity_params, train_per_round);
            let mut antibodies: Vec<_> = scored_pop.into_iter().map(|x| x.2).collect();

            // get predictor error

            let weighted_error_count: f64 = antigens
                .iter()
                .map(|antigen| {
                    if let Some(is_corr) = is_class_correct(&antigen, &antibodies){
                        if is_corr {
                            return 0.0;
                        } else {
                            return 1.0 * antigen.boosting_weight;
                        }
                    }else {
                        return 0.5 * antigen.boosting_weight;
                    }
                })
                .sum();

            let weight_sum: f64 = antigens.iter().map(|ag| ag.boosting_weight).sum();
            let error = weighted_error_count / weight_sum;

            // calculate alpha weight for model
            let alpha_m = ((1.0 - error) / error).ln();

            // update weights
            for antigen in &mut antigens {
                let is_corr = is_class_correct(&antigen, &antibodies).unwrap_or(false);
                let val = if is_corr { 0.0 } else { alpha_m };

                antigen.boosting_weight = antigen.boosting_weight * val.exp();
            }

            // save the antibodies
            antibodies
                .iter_mut()
                .for_each(|ab| ab.boosting_model_alpha = alpha_m.clone());
            ab_pool.extend(antibodies);

            println!("\n#\n# for boost round {:?}:\n#", n);
            println!("current ab pool s {:?}:", ab_pool.len());


            println!("train");
            evaluate_print_population(&antigens, &ab_pool);

            println!("test");
            evaluate_print_population(&highly_dubious_practices, &ab_pool);

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

            translator.get_show_ag_acc(translator_formatted);
        }
        self.antibodies = ab_pool.clone();

        let mut match_counter = MatchCounter::new(&antigens);
        let evaluated_pop = evaluate_population(params, ab_pool, &antigens);
        match_counter.add_evaluations(
            evaluated_pop
                .iter()
                .map(|(evaluation, _)| evaluation)
                .collect::<Vec<_>>(),
        );


        let scored_pop = score_antibodies(params,evaluated_pop, &match_counter);


        // println!(
        //     "########## boost mask \n{:?}",
        //     match_counter.boosting_weight_values
        // );
        return (Vec::new(), Vec::new(), scored_pop);
    }

    fn train_ab_set(
        &self,
        antigens: &Vec<AntiGen>,
        params: &Params,
        verbosity_params: &VerbosityParams,
        pop_size: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<(f64, Evaluation, Antibody)>) {
        // =======  init misc training params  ======= //

        let leak_size = (antigens.len() as f64 * params.leak_fraction) as usize;

        let mut rng = rand::thread_rng();
        // check dims and classes
        let n_dims = antigens.get(0).unwrap().values.len();

        // build ag index
        let mut bucket_king: BucketKing<AntiGen> =
            BucketKing::new(n_dims, (0.0, 1.0), 10, |ag| ag.id, |ag| &ag.values);
        bucket_king.add_values_to_index(antigens);

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
            gen_initial_population(&bucket_king, antigens, params, &cell_factory, &pop_size, &match_counter);

        // the evaluated pop is the population where the ab -> ag matches has been calculated but not scored
        let mut evaluated_pop: Vec<(Evaluation, Antibody)> = Vec::with_capacity(pop_size);

        // the scored pop is the population where the ab -> ag matches and score has been calculated
        let mut scored_pop: Vec<(f64, Evaluation, Antibody)> = Vec::with_capacity(pop_size);

        evaluated_pop = evaluate_population(params, initial_population, antigens);
        match_counter.add_evaluations(
            evaluated_pop
                .iter()
                .map(|(evaluation, _)| evaluation)
                .collect::<Vec<_>>(),
        );

        scored_pop = score_antibodies(params,evaluated_pop,  &match_counter);

        //
        if verbosity_params.show_initial_pop_info {
            println!("initial");
            match_counter.class_labels.clone().into_iter().for_each(|cl| {
                let filtered: Vec<usize> = scored_pop
                    .iter()
                    .filter(|(_, _, c)| c.class_label == cl)
                    .map(|_| 1usize)
                    .collect();
                print!("num with {:?} is {:?} ", cl, filtered.len())
            });
            println!("\ninital end");
        }

        for i in 0..params.generations {
            // =======  tracking and logging   ======= //
            let max_score = scored_pop
                .iter()
                .map(|(score, _, _)| score)
                .max_by(|a, b| a.total_cmp(b))
                .unwrap();

            let avg_score =
                scored_pop.iter().map(|(score, _, _)| score).sum::<f64>() / scored_pop.len() as f64;

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
            }else {
                print!(
                        "iter: {:<5} of {:<5}\r",
                        i,
                        params.generations,
                    );
                io::stdout().flush().unwrap();
            }

            train_score_hist.push(avg_score);

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
            let mut new_gen: Vec<(Evaluation, Antibody)> = Vec::new();

            let mut parent_idx_vec: Vec<usize> = Vec::new();

            // calculate and preform the replacement selection/mutation for each ab label separately to maintain the label ratios
            for (label, fraction) in &match_counter.frac_map.clone() {
                let replace_count_for_label = (n_to_replace as f64 * fraction).ceil() as usize;
                if replace_count_for_label <= 0 || (*match_counter.count_map.get(label).unwrap() < replace_count_for_label) {
                    continue;
                }

                let parents = labeled_tournament_pick(
                    &scored_pop,
                    &replace_count_for_label,
                    &params.tournament_size,
                    Some(label),
                );

                parent_idx_vec.extend(parents.clone());

                // let (label_gen, match_updates): (Vec<(Evaluation, Antibody)>, Option<(Evaluation,Evaluation)>) = parents
                let (label_gen, match_updates): (
                    Vec<(Evaluation, Antibody)>,
                    Vec<Option<(Evaluation, Evaluation)>>,
                ) = parents
                    // .clone()
                    .into_par_iter() // TODO: set paralell
                    // .into_iter()
                    .map(|idx| scored_pop.get(idx).unwrap().clone())
                    .map(|(parent_score, parent_eval, parent_antibody)| {
                        // find the fraction of max score of the current ab, this is used for the cloning and mutation rates
                        let frac_of_max = (parent_score / max_score).max(0.2);
                        let n_clones =
                            ((params.n_parents_mutations as f64 * frac_of_max) as usize).max(1);

                        // generate the calculated amount of clones evaluate them
                        let children = (0..n_clones)
                            // .into_iter()
                            .into_par_iter() // TODO: set paralell
                            .map(|_| {
                                let mut mutated = mutate(
                                    params,
                                    (1.0 - frac_of_max),
                                    parent_antibody.clone(),
                                    antigens,
                                );
                                mutated.clone_count += 1;
                                let eval = evaluate_antibody(antigens, &mutated);
                                return (eval, mutated);
                            })
                            .collect::<Vec<(Evaluation, Antibody)>>(); //.into_iter();//.collect::<Vec<(Evaluation, Antibody)>>()

                        let child_evals = children
                            .iter()
                            .map(|(evaluation, _)| evaluation)
                            .collect::<Vec<_>>();

                        // add the child evals to the match counter so we can apply fitness sharing among the new clones
                        let mut local_match_counter = match_counter.clone();
                        local_match_counter.add_evaluations(child_evals);

                        let new_gen_scored = score_antibodies(params,children, &match_counter);
                        // rescore the parent ab with the new match counter vals to correctly fitness share the score
                        let (daddy_score, daddy_eval, daddy_antibody) = score_antibodies(
                            params,vec![(parent_eval, parent_antibody)],

                            &match_counter,
                        )
                        .pop()
                        .unwrap();

                        // println!("scores {:?}", new_gen_scored.iter().map(|(a,b,c)| a).collect::<Vec<_>>());

                        // find the best score among the children
                        let (best_s, best_eval, best_cell) = new_gen_scored
                            .into_iter()
                            .max_by(|(a, _, _), (b, _, _)| a.total_cmp(b))
                            .unwrap();
                        // println!("best s {:?} parent s {:?}, dady s: {:?}", best_s, parent_score,daddy_score);

                        // if the best child score beats the parent use the childs score
                        if (best_s > daddy_score) {
                            return (
                                (best_eval.clone(), best_cell),
                                Some((daddy_eval, best_eval)),
                            );
                        } else {
                            return ((daddy_eval, daddy_antibody), None);
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .unzip();

                let (removed_evals, added_evals): (Vec<Evaluation>, Vec<Evaluation>) =
                    match_updates.into_iter().filter_map(|a| a).unzip();

                match_counter.remove_evaluations(removed_evals.iter().collect());
                match_counter.add_evaluations(added_evals.iter().collect());
                new_gen.extend(label_gen)
            }

            let new_gen_scored = score_antibodies(params,new_gen, &match_counter);

            // filter to the n best new antigens
            // let mut to_add = pick_best_n(new_gen_scored, n_to_replace);
            let mut to_add = new_gen_scored;

            // =======  selection  ======= //

            for idx in parent_idx_vec {
                // let mut parent_value = scored_pop.get_mut(idx).unwrap();
                let (p_score, p_eval, p_cell) = scored_pop.get_mut(idx).unwrap();
                let (c_score, c_eval, c_cell) = to_add.pop().unwrap();
                std::mem::replace(p_score, c_score);
                std::mem::replace(p_eval, c_eval);
                std::mem::replace(p_cell, c_cell);
            }

            // =======  leak new  ======= //
            // let is_strip_round = i % 100 == 0;
            let is_strip_round = false;

            let n_to_leak = ((n_to_replace as f64 * params.leak_fraction) as usize);
            let mut new_leaked: Vec<(f64, Evaluation, Antibody)> = Vec::new();

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
                    if *count <= 0 {
                        continue;
                    }

                    let filtered: Vec<_> = antigens
                        .iter()
                        .filter(|ag| ag.class_label == *label)
                        .map(|ag| (ag, inversed.get(ag.id).unwrap_or(&0).clone() as i32)) // todo: validate that this unwrap or does not hide a bug
                        .collect();

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
                            };

                            if params.antibody_init_expand_radius {
                                new_antibody = expand_antibody_radius_until_hit(
                                    new_antibody,
                                    &bucket_king,
                                    &antigens,
                                )
                            }
                            let eval = evaluate_antibody(antigens, &new_antibody);
                            return (eval, new_antibody);
                        })
                        .collect();

                    let new_evals: Vec<_> = new_pop_pop.iter().map(|(e, _)| e).collect();
                    local_match_counter.add_evaluations(new_evals);
                    let leaked_to_add =
                        score_antibodies(params,new_pop_pop,  &local_match_counter);

                    let cleanup_evals: Vec<_> = leaked_to_add.iter().map(|(_, e, _)| e).collect();
                    local_match_counter.remove_evaluations(cleanup_evals);

                    new_leaked.extend(leaked_to_add);
                }
            }

            if new_leaked.len() > 0 {
                if is_strip_round {
                    scored_pop.extend(new_leaked);
                } else {
                    // scored_pop = replace_worst_n_per_cat(scored_pop, new_leaked, n_to_gen_map);
                    scored_pop = replace_if_better_per_cat(
                        scored_pop,
                        new_leaked,
                        n_to_gen_map,
                        &mut match_counter,
                    );
                }
            }

            // =======  next gen cleanup  ======= //

            let evaluated_pop: Vec<(Evaluation, Antibody)> =
                scored_pop.into_iter().map(|(a, b, c)| (b, c)).collect();

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

            scored_pop = score_antibodies(params,evaluated_pop,  &match_counter);

            if let Some(n) = verbosity_params.full_pop_acc_interval {
                if i % n == 0 {
                    let antibody: Vec<Antibody> =
                    scored_pop.iter().map(|(a, b, c)| c.clone()).collect();

                let mut n_corr = 0;
                let mut n_wrong = 0;
                let mut n_no_detect = 0;
                for antigen in antigens {
                    let pred_class = is_class_correct(&antigen, &antibody);
                    if let Some(v) = pred_class {
                        if v {
                            n_corr += 1
                        } else {
                            n_wrong += 1
                        }
                    } else {
                        n_no_detect += 1
                    }
                }

                let avg_acc = n_corr as f64 / antigens.len() as f64;
                // let avg_score = scored_pop.iter().map(|(a, _b, _)| a).sum::<f64>() / scored_pop.len() as f64;
                if avg_acc >= best_score {
                    best_score = avg_acc;
                    best_run = scored_pop.clone();
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
                match_counter.class_labels.clone().into_iter().for_each(|cl| {
                    let filtered: Vec<usize> = scored_pop
                        .iter()
                        .inspect(|(a, b, c)| {})
                        .filter(|(a, b, c)| c.class_label == cl)
                        .map(|(a, b, c)| 1usize)
                        .collect();
                    print!("num with {:?} is {:?} ", cl, filtered.len())
                });
                println!();
            }
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

        scored_pop = scored_pop
            .into_iter()
            // .filter(|(score, _, _)| *score >= 2.0)
            // .filter(|(score, _, _)| *score > 0.0)
            .map(|(score, evaluation, cell)| {
                let mut save_ab = cell.clone();
                save_ab.final_train_label_membership = Some(evaluation.membership_value);
                return (score, evaluation, save_ab);
            })

            .filter(|(score, _, cell)| cell.final_train_label_membership.unwrap().0 > params.membership_required)
            .collect();

        return (train_acc_hist, train_score_hist, scored_pop);
    }

    pub fn is_class_correct(&self, antigen: &AntiGen) -> Option<bool> {
        return is_class_correct(antigen, &self.antibodies);
    }

    pub fn make_prediction(&self, antigen: &AntiGen) -> Option<Prediction> {
        return make_prediction(antigen, &self.antibodies);
    }

    pub fn is_class_correct_with_membership(&self, antigen: &AntiGen) -> Option<bool> {
        return is_class_correct_with_membership(antigen, &self.antibodies);
    }
}
