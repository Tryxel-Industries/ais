use std::collections::{HashMap, HashSet};
use std::iter::Map;
use std::ops::{Range, RangeInclusive};

use rand::prelude::{IteratorRandom, SliceRandom};
use rand::Rng;
use rayon::prelude::*;
use statrs::statistics::Statistics;

use crate::bucket_empire::BucketKing;
use crate::evaluation::{evaluate_antibody, Evaluation, MatchCounter};
use crate::mutate;
use crate::params::{Params, VerbosityParams};
use crate::representation::antibody::Antibody;
use crate::representation::antibody_factory::AntibodyFactory;
use crate::representation::antigen::AntiGen;
use crate::representation::expand_antibody_radius_until_hit;
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

//
//  AIS
//

pub fn evaluate_population(
    bk: &BucketKing<AntiGen>,
    params: &Params,
    population: Vec<Antibody>,
    antigens: &Vec<AntiGen>,
) -> Vec<(Evaluation, Antibody)> {
    return population
        .into_par_iter() // TODO: set paralell
        // .into_iter()
        .map(|antibody| {
            // evaluate antibodies
            let score = evaluate_antibody(bk, antigens, &antibody);
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
) -> Vec<Antibody> {
    let mut rng = rand::thread_rng();
    return if params.antigen_pop_fraction == 1.0 {
        antigens
            .iter()
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
        (0..*pop_size)
            .map(|_| cell_factory.generate_from_antigen(antigens.choose(&mut rng).unwrap()))
            .map(|cell| {
                if params.antibody_init_expand_radius {
                    expand_antibody_radius_until_hit(cell, &bucket_king, &antigens)
                } else {
                    cell
                }
            })
            .collect()
    };
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
        // =======  init misc training params  ======= //

        let pop_size = (antigens.len() as f64 * params.antigen_pop_fraction) as usize;
        let leak_size = (antigens.len() as f64 * params.leak_fraction) as usize;

        let mut rng = rand::thread_rng();
        // check dims and classes
        let n_dims = antigens.get(0).unwrap().values.len();
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

        // build ag index
        let mut bucket_king: BucketKing<AntiGen> =
            BucketKing::new(n_dims, (0.0, 1.0), 10, |ag| ag.id, |ag| &ag.values);
        bucket_king.add_values_to_index(antigens);

        // keeps track of the matches
        let max_ag_id = antigens.iter().max_by_key(|ag| ag.id).unwrap().id;
        let mut match_counter = MatchCounter::new(max_ag_id);

        // init hist and watchers
        let mut best_run: Vec<(f64, Evaluation, Antibody)> = Vec::new();
        let mut best_score = 0.0;
        let mut train_score_hist = Vec::new();
        let mut train_acc_hist = Vec::new();

        // make the cell factory
        let cell_factory = AntibodyFactory::new(
            n_dims,
            params.antibody_ag_init_multiplier_range.clone(),
            params.antibody_ag_init_range_range.clone(),
            params.antibody_ag_init_value_types.clone(),
            params.antibody_rand_init_multiplier_range.clone(),
            params.antibody_rand_init_offset_range.clone(),
            params.antibody_rand_init_range_range.clone(),
            params.antibody_rand_init_value_types.clone(),
            Vec::from_iter(class_labels.clone().into_iter()),
        );

        // =======  set up population  ======= //
        let initial_population: Vec<Antibody> =
            gen_initial_population(&bucket_king, antigens, params, &cell_factory, &pop_size);

        // the evaluated pop is the population where the ab -> ag matches has been calculated but not scored
        let mut evaluated_pop: Vec<(Evaluation, Antibody)> = Vec::with_capacity(pop_size);

        // the scored pop is the population where the ab -> ag matches and score has been calculated
        let mut scored_pop: Vec<(f64, Evaluation, Antibody)> = Vec::with_capacity(pop_size);

        evaluated_pop = evaluate_population(&bucket_king, params, initial_population, antigens);
        match_counter.add_evaluations(
            evaluated_pop
                .iter()
                .map(|(evaluation, _)| evaluation)
                .collect::<Vec<_>>(),
        );

        scored_pop = score_antibodies(evaluated_pop, &count_map, &match_counter);

        //
        if verbosity_params.show_initial_pop_info {
            println!("initial");
            class_labels.clone().into_iter().for_each(|cl| {
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
            }

            train_score_hist.push(avg_score);

            // =======  parent selection  ======= //
            let replace_exponent = (3.0 / 2.0) * (((i as f64) + 1.0) / params.generations as f64);
            let replace_frac =
                params.max_replacment_frac * (2.0 / pop_size as f64).powf(replace_exponent);
            // params.max_replacment_frac * (2.0 / pop_size as f64).powf(replace_exponent) + 0.05;
            let mut n_to_replace = (pop_size as f64 * replace_frac).ceil() as usize;

            // =======  clone -> mut -> eval  ======= //
            let mut new_gen: Vec<(Evaluation, Antibody)> = Vec::new();

            let mut parent_idx_vec: Vec<usize> = Vec::new();

            // calculate and preform the replacement selection/mutation for each ab label separately to maintain the label ratios
            for (label, fraction) in &frac_map {
                let replace_count_for_label = (n_to_replace as f64 * fraction).ceil() as usize;
                if replace_count_for_label <= 0 {
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
                    .clone()
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
                                let mutated = mutate(params, frac_of_max, parent_antibody.clone());
                                let eval = evaluate_antibody(&bucket_king, antigens, &mutated);
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

                        let new_gen_scored = score_antibodies(children, &count_map, &match_counter);
                        // rescore the parent ab with the new match counter vals to correctly fitness share the score
                        let (daddy_score, daddy_eval, daddy_antibody) = score_antibodies(
                            vec![(parent_eval, parent_antibody)],
                            &count_map,
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

            let new_gen_scored = score_antibodies(new_gen, &count_map, &match_counter);

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

                for (label, fraction) in &frac_map {
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
                            let eval = evaluate_antibody(&bucket_king, antigens, &new_antibody);
                            return (eval, new_antibody);
                        })
                        .collect();

                    let new_evals: Vec<_> = new_pop_pop.iter().map(|(e,_)|e).collect();
                    local_match_counter.add_evaluations(new_evals);
                    let leaked_to_add = score_antibodies(new_pop_pop, &count_map, &local_match_counter);

                    let cleanup_evals: Vec<_> = leaked_to_add.iter().map(|(_,e,_)|e).collect();
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

            scored_pop = score_antibodies(evaluated_pop, &count_map, &match_counter);

            if let Some(n) = verbosity_params.full_pop_acc_interval {
                let antibody: Vec<Antibody> =
                    scored_pop.iter().map(|(a, b, c)| c.clone()).collect();

                self.antibodies = antibody;
                let mut n_corr = 0;
                let mut n_wrong = 0;
                let mut n_no_detect = 0;
                for antigen in antigens {
                    let pred_class = self.is_class_correct(&antigen);
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

            if verbosity_params.show_class_info {
                println!("replacing {:} leaking {:}", n_to_replace, n_to_leak);
                class_labels.clone().into_iter().for_each(|cl| {
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

        self.antibodies = scored_pop
            .iter()
            // .filter(|(score, _, _)| *score >= 2.0)
            .filter(|(score, _, _)| *score > 0.0)
            .map(|(_score, _ev, cell)| cell.clone())
            .collect();
        return (train_acc_hist, train_score_hist, scored_pop);
    }

    pub fn is_class_correct(&self, antigen: &AntiGen) -> Option<bool> {
        let matching_cells = self
            .antibodies
            .iter()
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
}
