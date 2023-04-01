use crate::bucket_empire::BucketKing;
use crate::evaluation::{
    evaluate_b_cell, expand_merge_mask, gen_error_merge_mask, gen_merge_mask, score_b_cells,
    Evaluation,
};
use crate::mutate;
use crate::representation::{
    expand_b_cell_radius_until_hit, AntiGen, BCell, BCellFactory, DimValueType,
};
use crate::selection::{elitism_selection, kill_by_mask_yo, labeled_tournament_pick, pick_best_n, replace_if_better_per_cat, replace_worst_n_per_cat, snip_worst_n, tournament_pick};
use rand::prelude::{IteratorRandom, SliceRandom};
use rand::Rng;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::iter::Map;
use std::ops::{Range, RangeInclusive};
use statrs::statistics::Statistics;


#[derive(Clone)]
pub enum MutationType {
    Offset,
    Multiplier,
    ValueType,
    Radius,
    Label,
}

pub struct Params {
    // -- train params -- //
    pub antigen_pop_fraction: f64,
    pub leak_fraction: f64,
    pub leak_rand_prob: f64,
    pub generations: usize,

    pub mutation_offset_weight: usize,
    pub mutation_multiplier_weight: usize,
    pub mutation_radius_weight: usize,
    pub mutation_value_type_weight: usize,
    pub mutation_label_weight: usize,

    pub offset_mutation_multiplier_range: RangeInclusive<f64>,
    pub multiplier_mutation_multiplier_range: RangeInclusive<f64>,
    pub radius_mutation_multiplier_range: RangeInclusive<f64>,
    pub value_type_valid_mutations: Vec<DimValueType>,
    pub label_valid_mutations: Vec<usize>,

    // selection
    pub max_replacment_frac: f64,
    pub tournament_size: usize,

    pub n_parents_mutations: usize,

    pub b_cell_init_expand_radius: bool,

    // -- B-cell from antigen initialization -- //
    pub b_cell_ag_init_multiplier_range: RangeInclusive<f64>,
    pub b_cell_ag_init_value_types: Vec<DimValueType>,
    pub b_cell_ag_init_range_range: RangeInclusive<f64>,

    // -- B-cell from random initialization -- //
    pub b_cell_rand_init_offset_range: RangeInclusive<f64>,
    pub b_cell_rand_init_multiplier_range: RangeInclusive<f64>,
    pub b_cell_rand_init_value_types: Vec<DimValueType>,
    pub b_cell_rand_init_range_range: RangeInclusive<f64>,
}

impl Params {
    pub fn roll_mutation_type(&self) -> MutationType {
        let weighted = vec![
            (MutationType::Offset, self.mutation_offset_weight),
            (MutationType::Multiplier, self.mutation_multiplier_weight),
            (MutationType::ValueType, self.mutation_value_type_weight),
            (MutationType::Radius, self.mutation_radius_weight),
            (MutationType::Label, self.mutation_label_weight),
        ];

        let mut rng = rand::thread_rng();
        return weighted
            .choose_weighted(&mut rng, |v| v.1)
            .unwrap()
            .0
            .clone();
    }
}

/*
1. select n parents
2. clone -> mutate parents n times
 */
pub struct ArtificialImmuneSystem {
    pub b_cells: Vec<BCell>,
}

//
//  AIS
//

fn inverse_match_mask(mask: &Vec<usize>) -> Vec<usize> {
    // let max_val = mask.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let max_val = mask.iter().max().unwrap();
    let mut inversed = mask.clone();
    inversed = inversed.iter().map(|v| max_val - v).collect();
    return inversed;
}
pub fn evaluate_population(
    bk: &BucketKing<AntiGen>,
    params: &Params,
    population: Vec<BCell>,
    antigens: &Vec<AntiGen>,
) -> Vec<(Evaluation, BCell)> {
    return population
        .into_par_iter()// TODO: set paralell
        // .into_iter()

        .map(|b_cell| {
            // evaluate b_cells
            let score = evaluate_b_cell(bk, antigens, &b_cell);
            return (score, b_cell);
        })
        .collect();
}

fn remove_strictly_worse(
    mut scored_pop: Vec<(f64,Evaluation, BCell)>,
    match_mask: &mut Vec<usize>,
    error_match_mask: &mut Vec<usize>,
    max_rm: Option<usize>,
) -> (HashMap<usize, usize>, Vec<(f64,Evaluation, BCell)>) {
    let mut removed_tracker: HashMap<usize, usize> = HashMap::new();
    // let mut out_vec:Vec<(Evaluation,BCell)> = Vec::with_capacity(evaluated_pop.len());

    scored_pop.sort_by(|(_,eval_a, _), (_, eval_b, _)| {
        let errors_a = eval_a.wrongly_matched.len();
        let errors_b = eval_b.wrongly_matched.len();

        if errors_a != errors_b {
            return eval_b.matched_ids.len().cmp(&eval_a.matched_ids.len());
        } else {
            return errors_b.cmp(&errors_a);
        }
    });

    let out_vec: Vec<(f64,Evaluation, BCell)> = scored_pop
        .into_iter()
        // .filter(|(a,b)|b.class_label == *label)
        .filter_map(|(s,a, b)| {
            // println!("match mask: {:?}", match_mask);
            let removed_count = removed_tracker.get(&b.class_label).unwrap_or(&0);
                if let Some(max_v) = max_rm{
                    if *removed_count > max_v{
                        return Some((s,a, b));
                    }
                }

            let mut strictly_worse = true;
            for id in &a.matched_ids {
                let sharers = match_mask.get(*id).unwrap();
                let errors = error_match_mask.get(*id).unwrap();

                if sharers -1 <= *errors{
                    // to avoid snipping that results in acc loss
                    strictly_worse = false;
                    break;
                }else if *sharers <= 1 {
                    strictly_worse = false;
                    break;
                }
            }

            // println!("is sw {:?}", strictly_worse);

            if strictly_worse {



                a.matched_ids
                    .iter()
                    .for_each(|v| *match_mask.get_mut(*v).unwrap() -= 1);
                a.wrongly_matched
                    .iter()
                    .for_each(|v| *error_match_mask.get_mut(*v).unwrap() -= 1);
                removed_tracker.insert(b.class_label, removed_count + 1);
                return None;
            } else {
                return Some((s,a, b));
            }
        })
        .collect();

    return (removed_tracker, out_vec);
}

enum BCellStates {
    New(BCell),
    Evaluated(Evaluation, BCell),
    Scored(f64, Evaluation, BCell),
}
impl ArtificialImmuneSystem {
    pub fn new() -> ArtificialImmuneSystem {
        return Self {
            b_cells: Vec::new(),
        };
    }

    pub fn train(
        &mut self,
        antigens: &Vec<AntiGen>,
        params: &Params,
        verbose: bool,
    ) -> (Vec<f64>, Vec<f64>, Vec<(f64, Evaluation, BCell)>) {
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
        let count_map: HashMap<usize,usize> = class_labels
            .iter()
            .map(|x| {
                (
                    x.clone(),
                    antigens
                        .iter()
                        .filter(|ag| ag.class_label == *x)
                        .collect::<Vec<&AntiGen>>()
                        .len()
                )
            })
            .collect();

        // build ag index
        let mut bk: BucketKing<AntiGen> =
            BucketKing::new(n_dims, (0.0, 1.0), 10, |ag| ag.id, |ag| &ag.values);
        bk.add_values_to_index(antigens);

        // init hist and watchers
        let mut best_run: Vec<(f64, Evaluation, BCell)> = Vec::new();
        let mut best_score = 0.0;
        let mut train_score_hist = Vec::new();
        let mut train_acc_hist = Vec::new();

        // make the cell factory
        let cell_factory = BCellFactory::new(
            n_dims,
            params.b_cell_ag_init_multiplier_range.clone(),
            params.b_cell_ag_init_range_range.clone(),
            params.b_cell_ag_init_value_types.clone(),
            params.b_cell_rand_init_multiplier_range.clone(),
            params.b_cell_rand_init_offset_range.clone(),
            params.b_cell_rand_init_range_range.clone(),
            params.b_cell_rand_init_value_types.clone(),
            Vec::from_iter(class_labels.clone().into_iter()),
        );

        // =======  set up population  ======= //
        /*
        evaluated population -> population with meta info about what correct and incorrect matches the b-cell has
        scored population -> evaluated pop with aditional info about the current b-cell score
        match_mask -> a vector of equal size to the number of antigen samples, indicating how many matches the ag with an id equal to the vec index has
         */

        let mut evaluated_pop: Vec<(Evaluation, BCell)> = Vec::with_capacity(pop_size);
        let mut scored_pop: Vec<(f64, Evaluation, BCell)> = Vec::with_capacity(pop_size);
        let mut match_mask: Vec<usize> = Vec::new();
        let mut error_match_mask: Vec<usize> = Vec::new();

        // gen init pop
        let initial_population: Vec<BCell> = if params.antigen_pop_fraction == 1.0 {
            antigens
                .iter()
                .map(|ag| cell_factory.generate_from_antigen(ag))
                .map(|cell| {
                    if params.b_cell_init_expand_radius {
                        expand_b_cell_radius_until_hit(cell, &bk, &antigens)
                    } else {
                        cell
                    }
                })
                .collect()
        } else {
            (0..pop_size)
                .map(|_| cell_factory.generate_from_antigen(antigens.choose(&mut rng).unwrap()))
                .map(|cell| {
                    if params.b_cell_init_expand_radius {
                        expand_b_cell_radius_until_hit(cell, &bk, &antigens)
                    } else {
                        cell
                    }
                })
                .collect()
        };
        // let mut initial_population: Vec<BCell> = Vec::with_capacity(pop_size);

        // antigens
        //     .iter()
        //     .for_each(|ag| initial_population.push(cell_factory.generate_from_antigen(ag)));

        evaluated_pop = evaluate_population(&bk, params, initial_population, antigens);
        match_mask = gen_merge_mask(&evaluated_pop);
        error_match_mask = gen_error_merge_mask(&evaluated_pop);
        scored_pop = score_b_cells(evaluated_pop, &match_mask, &error_match_mask, &count_map);
        if verbose{
            println!("initial");
            class_labels.clone().into_iter().for_each(|cl| {
                let filtered: Vec<usize> = scored_pop
                    .iter()
                    .inspect(|(a, b, c)| {})
                    .filter(|(a, b, c)| c.class_label == cl)
                    .map(|(a, b, c)| 1usize)
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
                scored_pop.iter().map(|(a, _b, _)| a).sum::<f64>() / scored_pop.len() as f64;

            if verbose{
                println!(
                    "iter: {:<5} avg score {:.6}, max score {:.6}, last acc {:.6}",
                    i,
                    avg_score,
                    max_score,
                    train_acc_hist.last().unwrap_or(&0.0)
                );
                println!("pop size {:} ", scored_pop.len());
            }

            train_score_hist.push(avg_score);


            if i % 500 == 0{
                println!(
                    "iter: {:<5} avg score {:.6}, max score {:.6}, last acc {:.6}",
                    i,
                    avg_score,
                    max_score,
                    train_acc_hist.last().unwrap_or(&0.0)
                );
            }


            // =======  parent selection  ======= //
            let replace_exponent = (3.0 / 2.0) * (((i as f64) + 1.0) / params.generations as f64);
            let replace_frac =
                params.max_replacment_frac * (2.0 / pop_size as f64).powf(replace_exponent);
            params.max_replacment_frac * (2.0 / pop_size as f64).powf(replace_exponent) + 0.05;
            let mut n_to_replace = (pop_size as f64 * replace_frac).ceil() as usize;

            // =======  clone -> mut -> eval  ======= //
            let mut new_gen: Vec<(Evaluation, BCell)> = Vec::new();

            let mut parent_idx_vec: Vec<usize> = Vec::new();

            let mut clone_count: Vec<f64> = Vec::new();

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

                let label_gen: Vec<(Evaluation, BCell)> = parents
                    .clone()
                    .into_par_iter() // TODO: set paralell
                    // .into_iter()
                    .map(|idx| scored_pop.get(idx).unwrap().clone())
                    .map(|(parent_score, parent_eval, parent_b_cell)| {
                        // println!("############################");
                        let frac_of_max = (parent_score/max_score).max(0.2);

                        // println!("max score {:?} parent score {:?}",max_score, parent_score);
                        let n_clones = ((params.n_parents_mutations as f64 * frac_of_max) as usize).max(1);

                        // clone_count.push(n_clones.clone() as f64);

                        // println!("n clones {:?}",n_clones);
                        let children = (0..n_clones)
                            // .into_iter()
                            .into_par_iter() // TODO: set paralell
                            .map(|_| {
                                let mutated = mutate(params, frac_of_max, parent_b_cell.clone());
                                let eval = evaluate_b_cell(&bk, antigens, &mutated);
                                return (eval, mutated);
                            })
                            .collect::<Vec<(Evaluation, BCell)>>(); //.into_iter();//.collect::<Vec<(Evaluation, BCell)>>()
                        let mut new_local_match_mask =
                            expand_merge_mask(&children, match_mask.clone(), false);
                        let mut new_local_error_match_mask =
                            expand_merge_mask(&children, error_match_mask.clone(), true);


                        let new_gen_scored = score_b_cells(
                            children,
                            &new_local_match_mask,
                            &new_local_error_match_mask, &count_map
                        );
                        let (daddy_score, daddy_eval, daddy_bcell) = score_b_cells(
                            vec![(parent_eval, parent_b_cell)],
                            &new_local_match_mask,
                            &error_match_mask, &count_map
                        )
                        .pop()
                        .unwrap();

                        // println!("scores {:?}", new_gen_scored.iter().map(|(a,b,c)| a).collect::<Vec<_>>());
                        let (best_s, best_eval, best_cell) = new_gen_scored
                            .into_iter()
                            .max_by(|(a, _, _), (b, _, _)| a.total_cmp(b))
                            .unwrap();
                        // println!("best s {:?} parent s {:?}, dady s: {:?}", best_s, parent_score,daddy_score);

                        if (best_s > daddy_score) {
                            // println!("\ninc {:?}", best_s -daddy_score);
                            // if best_cell.class_label != daddy_bcell.class_label{
                            //     print!("\n\n\n\n\n\n")
                            // }
                            return (best_eval, best_cell);
                        } else {
                            return (daddy_eval, daddy_bcell);
                        }
                    })
                    .collect();

                new_gen.extend(label_gen)
            }

            // println!("avg clone count {:?}", clone_count.mean());
            // println!("{:?}", parent_idx_vec);

            // gen a new match mask for the added values
            let new_gen_match_mask = expand_merge_mask(&new_gen, match_mask.clone(), false);
            let new_gen_error_match_mask =
                expand_merge_mask(&new_gen, error_match_mask.clone(), true);
            let new_gen_scored =
                score_b_cells(new_gen, &new_gen_match_mask, &new_gen_error_match_mask, &count_map);

            // filter to the n best new antigens
            // let mut to_add = pick_best_n(new_gen_scored, n_to_replace);
            let mut to_add = new_gen_scored;

            // =======  selection  ======= //

            if true {
                for idx in parent_idx_vec {
                    // let mut parent_value = scored_pop.get_mut(idx).unwrap();
                    let (p_score, p_eval, p_cell) = scored_pop.get_mut(idx).unwrap();
                    let (c_score, c_eval, c_cell) = to_add.pop().unwrap();
                    std::mem::replace(p_score, c_score);
                    std::mem::replace(p_eval, c_eval);
                    std::mem::replace(p_cell, c_cell);
                }
            }

            // =======  leak new  ======= //
            // let is_strip_round = i % 100 == 0;
            let is_strip_round = false;

            let n_to_leak = ((n_to_replace as f64 * params.leak_fraction) as usize);
            let mut new_leaked: Vec<(f64, Evaluation, BCell)> = Vec::new();

            let n_to_gen_map = if is_strip_round {
                let (removed_map, stripped_pop) =
                    remove_strictly_worse(scored_pop, &mut match_mask, &mut error_match_mask, Some(5));
                scored_pop = stripped_pop;
                println!("\n\nstrip round stripping map {:?}\n\n", removed_map);
                removed_map
            } else {
                let mut replace_map = HashMap::new();

                for (label, fraction) in &frac_map {
                    let replace_count_for_label = (n_to_leak as f64 * fraction).ceil() as usize;
                    replace_map.insert(label.clone(), replace_count_for_label);
                }
                replace_map
            };

            if params.leak_fraction > 0.0 {
                let inversed = inverse_match_mask(&match_mask);

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
                        let mut new_b_cell = if rng.gen_bool(1.0 - params.leak_rand_prob) {
                            cell_factory.generate_from_antigen(&new_ag)
                        } else {
                            cell_factory.generate_random_genome_with_label(new_ag.class_label)
                        };

                        if params.b_cell_init_expand_radius {
                            new_b_cell = expand_b_cell_radius_until_hit(new_b_cell, &bk, &antigens)
                        }
                        let eval = evaluate_b_cell(&bk, antigens, &new_b_cell);
                        return (eval, new_b_cell);
                        }).collect();
                 /*
                    let mut new_pop_pop = Vec::new();
                    for n in 0..*count {
                        let new_ag = filtered
                            .choose_weighted(&mut rng, |v| v.1 + 1)
                            .unwrap()
                            .0
                            .clone();
                        let mut new_b_cell = if rng.gen_bool(1.0 - params.leak_rand_prob) {
                            cell_factory.generate_from_antigen(&new_ag)
                        } else {
                            cell_factory.generate_random_genome_with_label(new_ag.class_label)
                        };

                        if params.b_cell_init_expand_radius {
                            new_b_cell = expand_b_cell_radius_until_hit(new_b_cell, &bk, &antigens)
                        }
                        let eval = evaluate_b_cell(&bk, antigens, &new_b_cell);
                        new_pop_pop.push((eval, new_b_cell))
                    }
*/
                    // let new_pop_pop = antigens
                    //     .iter()
                    //     .filter(|ag| ag.class_label == *label)
                    //     .choose_multiple(&mut rng, replace_count_for_label)
                    //     .iter()
                    //     .map(|ag| {
                    //         if rng.gen_bool(1.0 - params.leak_rand_prob){
                    //             cell_factory.generate_from_antigen(ag)
                    //         }else {
                    //             cell_factory.generate_random_genome_with_label(ag.class_label)
                    //         }
                    //     })
                    //     .map(|cell| {
                    //         if params.b_cell_init_expand_radius{
                    //             expand_b_cell_radius_until_hit(cell, &bk, &antigens)
                    //         }else {
                    //             cell
                    //         }
                    //     })
                    //     .map(|cell| (evaluate_b_cell(&bk,  antigens, &cell), cell))
                    //     .collect::<Vec<(Evaluation, BCell)>>();

                    let leaked_to_add =
                        score_b_cells(new_pop_pop, &new_gen_match_mask, &error_match_mask, &count_map);
                    new_leaked.extend(leaked_to_add);
                }
            }

            if new_leaked.len() > 0{
                        if is_strip_round{
                scored_pop.extend(new_leaked);
            }else {
                // scored_pop = replace_worst_n_per_cat(scored_pop, new_leaked, n_to_gen_map);
                    scored_pop = replace_if_better_per_cat(scored_pop, new_leaked, n_to_gen_map);
            }
            }


            // scored_pop = snip_worst_n(scored_pop, n_to_leak);
            // scored_pop.extend(new_leaked);

            // =======  next gen cleanup  ======= //

            // let pop = scored_pop.into_iter().map(|(a, b, c)| c).collect();
            // evaluated_pop = evaluate_population(&bk, &params, pop, &antigens);

            let evaluated_pop = scored_pop.into_iter().map(|(a, b, c)| (b, c)).collect();

            match_mask = gen_merge_mask(&evaluated_pop);
            error_match_mask = gen_error_merge_mask(&evaluated_pop);

            scored_pop = score_b_cells(evaluated_pop, &match_mask, &error_match_mask, &count_map);

            if true {
                let b_cell: Vec<BCell> = scored_pop.iter().map(|(a, b, c)| c.clone()).collect();

                self.b_cells = b_cell;
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
                train_acc_hist.push(0.0);
            }

            if verbose{
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

        if false{

            println!("########## match mask \n{:?}", match_mask);
            let (removed_map, stripped_pop) =
                remove_strictly_worse(scored_pop, &mut match_mask, &mut error_match_mask, None);
            scored_pop = stripped_pop;
            println!("stripped count {:?}", removed_map);
            println!("########## match mask \n{:?}", match_mask);

        }

        self.b_cells = scored_pop
            .iter()
            // .filter(|(score, _, _)| *score >= 2.0)
            .filter(|(score, _, _)| *score > 0.0)
            .map(|(_score, _ev, cell)| cell.clone())
            .collect();
        return (train_acc_hist, train_score_hist, scored_pop);
    }

    pub fn is_class_correct(&self, antigen: &AntiGen) -> Option<bool> {
        let matching_cells = self
            .b_cells
            .iter()
            .filter(|b_cell| b_cell.test_antigen(antigen))
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
