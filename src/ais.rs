use crate::bucket_empire::BucketKing;
use crate::evaluation::{evaluate_b_cell, score_b_cells, Evaluation, gen_merge_mask, expand_merge_mask};
use crate::representation::{AntiGen, BCell, BCellFactory, DimValueType};
use crate::selection::{elitism_selection, kill_by_mask_yo, pick_best_n, snip_worst_n, tournament_pick};
use rayon::prelude::*;
use std::collections::HashSet;
use std::ops::{Range, RangeInclusive};

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
            (MutationType::Multiplier,self.mutation_multiplier_weight),
             (MutationType::Radius,self.mutation_radius_weight),
              (MutationType::Label,self.mutation_label_weight)];

        let mut rng = rand::thread_rng();
        return  weighted.choose_weighted(&mut rng ,|v|v.1).unwrap().0.clone();


;
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

fn evaluate_population(
    bk: &BucketKing<AntiGen>,
    params: &Params,
    population: Vec<BCell>,
    antigens: &Vec<AntiGen>,
) -> Vec<(Evaluation, BCell)> {
    return population
        .into_par_iter()
        .map(|b_cell| {
            // evaluate b_cells
            let score = evaluate_b_cell(bk, params, antigens, &b_cell);
            return (score, b_cell);
        })
        .collect();
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
    ) -> (Vec<f64>, Vec<(f64, Evaluation, BCell)>) {
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

        // build ag index
        let mut bk: BucketKing<AntiGen> =
            BucketKing::new(n_dims, (0.0, 10.0), 10, |ag| ag.id, |ag| &ag.values);
        bk.add_values_to_index(antigens);

        // init hist and watchers
        let mut best_run: Vec<(f64, Evaluation, BCell)> = Vec::new();
        let mut best_score = 0.0;
        let mut train_hist = Vec::new();

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

        // gen init pop
        let mut initial_population: Vec<BCell> = (0..pop_size)
            .map(|_| cell_factory.generate_from_antigen(antigens.choose(&mut rng).unwrap()))
            .collect();
        // let mut initial_population: Vec<BCell> = Vec::with_capacity(pop_size);

        // antigens
        //     .iter()
        //     .for_each(|ag| initial_population.push(cell_factory.generate_from_antigen(ag)));

        let initial_population: Vec<BCell> = antigens
            .iter()
            .map(|ag| cell_factory.generate_from_antigen(ag))
            .collect();

        evaluated_pop = evaluate_population(&bk, params, initial_population, antigens);
        match_mask = gen_merge_mask(&evaluated_pop);
        scored_pop = score_b_cells(evaluated_pop, &match_mask);

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


        for i in 0..params.generations {
            // =======  tracking and logging   ======= //
            let max_score = scored_pop
                .iter()
                .map(|(score, _, _)| score)
                .max_by(|a, b| a.total_cmp(b))
                .unwrap();

            println!(
                "iter: {:<5} avg score {:.6}, max score {:.6}, last score {:.6}",
                i,
                scored_pop.iter().map(|(a, _b, _)| a).sum::<f64>() / scored_pop.len() as f64,
                max_score,
                train_hist.last().unwrap_or(&0.0)
            );
            println!("pop size {:} ",scored_pop.len());

            // =======  parent selection  ======= //
            let replace_exponent = (3.0/2.0)*(((i as f64)+1.0)/params.generations as f64);
            let replace_frac = params.max_replacment_frac * (2.0/pop_size as f64).powf(replace_exponent);
            let mut n_to_replace = (pop_size as f64 * replace_frac).ceil() as usize;

            let parents = tournament_pick(&scored_pop, &n_to_replace, &params.tournament_size);



            // =======  clone -> mut -> eval  ======= //

            let new_gen: Vec<(Evaluation, BCell)> = parents.clone()
                .into_par_iter() // TODO: set paralell
                // .into_iter()
                .map(|idx| scored_pop.get(idx).unwrap().clone())
                .map(|(parent_score, parent_eval, parent_b_cell)|{
                    let  children = (0..params.n_parents_mutations).into_iter().map(|_|{
                        let mutated = mutate(params, parent_score,parent_b_cell.clone());
                        let eval = evaluate_b_cell(&bk, params, antigens, &parent_b_cell);
                        return (eval, mutated)
                    }).collect::<Vec<(Evaluation, BCell)>>();//.into_iter();//.collect::<Vec<(Evaluation, BCell)>>()
                    let mut new_local_match_mask = expand_merge_mask(&children, match_mask.clone());

                    let new_gen_scored = score_b_cells(children, &new_local_match_mask);
                    let (daddy_score,daddy_eval, daddy_bcell) = score_b_cells(vec![(parent_eval, parent_b_cell)], &new_local_match_mask).pop().unwrap();

                    let (best_s, best_eval, best_cell) = new_gen_scored
                        .into_iter()
                        .max_by(|(a,_,_), (b,_,_)| a.total_cmp(b)).unwrap();

                    if (best_s >= daddy_score){
                        return (best_eval, best_cell)
                    }else {
                        return (daddy_eval, daddy_bcell)
                    }


                })
                // .flatten()
                // .flatten_iter()
                .collect();

            // gen a new match mask for the added values
            let new_gen_match_mask = expand_merge_mask(&new_gen, match_mask.clone());
            let new_gen_scored = score_b_cells(new_gen, &new_gen_match_mask);

            // filter to the n best new antigens
            // let mut to_add = pick_best_n(new_gen_scored, n_to_replace);
            let mut to_add = new_gen_scored;

            // =======  leak new  ======= //
            let n_to_leak = (n_to_replace as f64 * params.leak_fraction) as usize;

            let mut new_leaked: Vec<(f64,Evaluation, BCell)> = Vec::new();
            if params.leak_fraction > 0.0{

                let new_pop_pop = antigens.choose_multiple(&mut rng,n_to_leak)
                    .map(|ag| cell_factory.generate_from_antigen(ag))
                    .map(|cell| (evaluate_b_cell(&bk, params, antigens, &cell), cell))
                    .collect::<Vec<(Evaluation, BCell)>>();

                let leaked_to_add = score_b_cells(new_pop_pop, &new_gen_match_mask);
                new_leaked.extend(leaked_to_add);
            }
            // =======  selection  ======= //
            // remove the n worst b-cells
            // scored_pop = snip_worst_n(scored_pop, n_to_replace);
            // scored_pop.extend(to_add.into_iter());
            if true{
                for idx in parents{
                    // let mut parent_value = scored_pop.get_mut(idx).unwrap();
                    let (p_score,p_eval,p_cell) = scored_pop.get_mut(idx).unwrap();
                    let  (c_score,c_eval,c_cell) = to_add.pop().unwrap();
                    std::mem::replace(p_score, c_score);
                    std::mem::replace(p_eval, c_eval);
                    std::mem::replace(p_cell, c_cell);
                }
            }

            scored_pop = snip_worst_n(scored_pop, n_to_leak);
            scored_pop.extend(new_leaked);

            // =======  next gen cleanup  ======= //
            evaluated_pop = scored_pop.into_iter().map(|(a,b,c)|(b,c)).collect();

            match_mask = gen_merge_mask(&evaluated_pop);
            scored_pop =  score_b_cells(evaluated_pop, &match_mask);




            let b_cell: Vec<BCell> = scored_pop.iter().map(|(a,b,c)|c.clone()).collect();

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


            let avg_score = n_corr as f64/antigens.len()as f64;
            // let avg_score = scored_pop.iter().map(|(a, _b, _)| a).sum::<f64>() / scored_pop.len() as f64;
            if avg_score >= best_score{
                best_score = avg_score;
                best_run = scored_pop.clone();
            }

            train_hist.push(avg_score);



            println!("replacing {:} leaking {:}",n_to_replace, n_to_leak);
            class_labels.clone().into_iter().for_each(|cl| {
                let filtered: Vec<usize> = scored_pop.iter().inspect(|(a,b,c)|{
                }).filter(|(a,b,c)| c.class_label == cl ).map(|(a,b,c)| 1usize).collect();
                print!("num with {:?} is {:?} ", cl,filtered.len() )
            });
            println!();
//
//             // find the matches for the new B-cells
//             //
//
//
//
//             // let new_scored: Vec<(f64, BCell)> = scored_pop.clone().into_par_iter().map(|(score ,b_cell)| {
//    /*         let new_evaluated: Vec<(Evaluation, BCell)> = scored_pop
//                 .clone()
//                 // .into_par_iter() // TODO: set paralell
//                 .into_iter()
//                 .map(|(parent_score, parent_evaluation, b_cell)| {
//                     let frac_score = parent_score / max_score;
//
//                     let mutated: Vec<(Evaluation, BCell)> = (0..5).map(|_|{
//                          let mutated = mutate_fn(params, frac_score, b_cell.clone());
//                          let evaluation = evaluate_b_cell(&bk, params, antigens, &mutated);
//                         return (evaluation, mutated)
//                      }).filter(|(ev, cell)| {
//                         let more_matches = ev.matched_ids.len() > parent_evaluation.matched_ids.len();
//                         let not_more_error = ev.wrongly_matched.len() <= parent_evaluation.wrongly_matched.len();
//                         return (more_matches & not_more_error) || (not_more_error)
//                     }).collect();
//
//
//                     if mutated.len() == 0{
//                         return vec![(parent_evaluation, b_cell)];
//                     } else{
//                         return mutated;
//                     }
//
//                 })
//                 .flat_map(|v| v.into_iter())
//                 .collect();
//
// */
//             let new_evaluated: Vec<(Evaluation, BCell)> = scored_pop
//                 .clone()
//                 // .into_par_iter() // TODO: set paralell
//                 .into_iter()
//                 .map(|(parent_score, parent_evaluation, b_cell)| {
//                     let frac_score = parent_score / max_score;
//
//                     for n in (0..5){
//                         let mutated = mutate_fn(params, frac_score, b_cell.clone());
//                         let evaluation = evaluate_b_cell(&bk, params, antigens, &mutated);
//
//                         let more_matches = evaluation.matched_ids.len() > parent_evaluation.matched_ids.len();
//                         let not_more_error = evaluation.wrongly_matched.len() <= parent_evaluation.wrongly_matched.len();
//
//
//                         let mut res_v = score_b_cells(vec![(evaluation, mutated)], &match_mask);
//                         let (new_score, new_eval, new_b_cell) = res_v.pop().unwrap();
//                         if new_score > parent_score{
//                             return (new_eval, new_b_cell)
//
//                         }
//                         // if (more_matches & not_more_error) || (not_more_error){
//                         //     return (evaluation,mutated)
//                         // }
//                     }
//                     return (parent_evaluation, b_cell)
//                 })
//                 .collect();
//
//             evaluated_pop = scored_pop.into_iter().map(|(_a, b, c)| (b, c)).collect();
//             // evaluated_pop.extend(new_evaluated);
//             evaluated_pop = new_evaluated;
//
//
//             // let new_scored = score_population(params, mutated, antigens, eval_fn);
//
//             // tweak by overmatching factor
//             match_mask = gen_merge_mask(&evaluated_pop);
//             scored_pop = score_b_cells(evaluated_pop, &match_mask);
//
//
//
//             println!("init len {:?}", scored_pop.len());
//             // scored_pop = kill_by_mask_yo(scored_pop, &mut match_mask);
//             println!("dedup len {:?}", scored_pop.len());
//
//             if params.b_cell_pop_size < scored_pop.len() {
//                 // select a subset of the pop
//                 // scored_pop = selector_fn(params, scored_pop, &mut match_mask);
//             }
//
//             let avg_score = scored_pop.iter().map(|(a, _b, _)| a).sum::<f64>() / scored_pop.len() as f64;
//             if avg_score > best_score{
//                 best_score = avg_score;
//                 best_run = scored_pop.clone();
//             }
//
//             train_hist.push(avg_score);
//
//             // gen new b_cells to fill the hole
//             if params.b_cell_pop_size > scored_pop.len() {
//                 let missing = params.b_cell_pop_size - scored_pop.len();
//                 let num_new_rand = (missing as f64 * params.new_val_rand_frac) as usize;
//                 let num_new_pop = missing - num_new_rand;
//
//                 // println!("num rand {:<3?} num new pop {:<3?}", num_new_rand, num_new_pop);
//                 let new_pop_rand = (0..num_new_rand)
//                     .map(|_| cell_factory.generate_random_genome())
//                     .map(|cell| (evaluate_b_cell(&bk, params, antigens, &cell), cell))
//                     .collect::<Vec<(Evaluation, BCell)>>();
//
//                 let new_pop_pop = antigens.choose_multiple(&mut rng,num_new_pop)
//                     .map(|ag| cell_factory.generate_from_antigen(ag))
//                     .map(|cell| (evaluate_b_cell(&bk, params, antigens, &cell), cell))
//                     .collect::<Vec<(Evaluation, BCell)>>();
//
//                 evaluated_pop = scored_pop.into_iter().map(|(_a, b, c)| (b, c)).collect();
//                 evaluated_pop.extend(new_pop_rand);
//                 evaluated_pop.extend(new_pop_pop);
//
//
//
//                 match_mask = gen_merge_mask(&evaluated_pop);
//                 scored_pop = score_b_cells(evaluated_pop, &match_mask);
//
//             }
        }

        // scored_pop = best_run;

        // scored_pop = snip_worst_n(scored_pop, 10);
        // let (scored_pop, _drained) = elitism_selection(scored_pop, &100);
        self.b_cells = scored_pop
            .iter()
            // .filter(|(score, _, _)| *score > 5.0)
            .map(|(_score, _ev, cell)| cell.clone())

            .collect();
        return (train_hist, scored_pop);
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
            return Some(false);
        }
    }
}
