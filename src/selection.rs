use std::collections::HashMap;

use rand::prelude::SliceRandom;
use rand::{distributions::Distribution, thread_rng, Rng};

use crate::evaluation::{Evaluation, MatchCounter};
use crate::params::Params;
use crate::representation::antibody::Antibody;
use crate::util::pick_n_random;

pub fn kill_by_mask_yo(
    mut population: Vec<(f64, Evaluation, Antibody)>,
    match_mask: &mut Vec<usize>,
) -> Vec<(f64, Evaluation, Antibody)> {
    let mut survivors: Vec<(f64, Evaluation, Antibody)> = Vec::new();

    population.shuffle(&mut thread_rng());

    for (score, eval, cell) in population.into_iter() {
        let mut keep = false;
        for idx in &eval.matched_ids {
            if let Some(cnt) = match_mask.get(*idx) {
                if *cnt <= 1 {
                    keep = true;
                    break;
                }
            }
        }

        if keep {
            survivors.push((score, eval, cell))
        } else {
            // all the covered vars are covered elsewhere
            for idx in &eval.matched_ids {
                if let Some(cnt) = match_mask.get_mut(*idx) {
                    *cnt -= 1
                }
            }
        }
    }
    return survivors;
}

/*


           let n_to_gen_map = if is_strip_round {
               let (removed_map, stripped_pop) = remove_strictly_worse(
                   scored_pop,
                   &mut match_mask,
                   &mut error_match_mask,
                   Some(5),
               );
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
*/
pub fn remove_strictly_worse(
    mut scored_pop: Vec<(f64, Evaluation, Antibody)>,
    match_mask: &mut Vec<usize>,
    error_match_mask: &mut Vec<usize>,
    max_rm: Option<usize>,
) -> (HashMap<usize, usize>, Vec<(f64, Evaluation, Antibody)>) {
    //todo: clean up if ends up using it
    let mut removed_tracker: HashMap<usize, usize> = HashMap::new();
    // let mut out_vec:Vec<(Evaluation,Antibody)> = Vec::with_capacity(evaluated_pop.len());

    scored_pop.sort_by(|(_, eval_a, _), (_, eval_b, _)| {
        let errors_a = eval_a.wrongly_matched.len();
        let errors_b = eval_b.wrongly_matched.len();

        if errors_a != errors_b {
            return eval_b.matched_ids.len().cmp(&eval_a.matched_ids.len());
        } else {
            return errors_b.cmp(&errors_a);
        }
    });

    let out_vec: Vec<(f64, Evaluation, Antibody)> = scored_pop
        .into_iter()
        // .filter(|(a,b)|b.class_label == *label)
        .filter_map(|(s, a, b)| {
            // println!("match mask: {:?}", match_mask);
            let removed_count = removed_tracker.get(&b.class_label).unwrap_or(&0);
            if let Some(max_v) = max_rm {
                if *removed_count > max_v {
                    return Some((s, a, b));
                }
            }

            let mut strictly_worse = true;
            for id in &a.matched_ids {
                let sharers = match_mask.get(*id).unwrap();
                let errors = error_match_mask.get(*id).unwrap();

                if sharers - 1 <= *errors {
                    // to avoid snipping that results in acc loss
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
                return Some((s, a, b));
            }
        })
        .collect();

    return (removed_tracker, out_vec);
}
pub fn snip_worst_n(
    mut population: Vec<(f64, Evaluation, Antibody)>,
    num_to_snip: usize,
) -> Vec<(f64, Evaluation, Antibody)> {
    population.sort_by(|(score_a, _, _), (score_b, _, _)| score_a.total_cmp(score_b));

    // println!("pre {:?}", population.iter().map(|(a,b,c)| a).collect::<Vec<&f64>>());
    let survivors = population.split_off(num_to_snip);
    // println!("post {:?}", survivors.iter().map(|(a,b,c)| a).collect::<Vec<&f64>>());
    // let snipped: Vec<(f64, Evaluation, Antibody)> = population
    //     .drain(0..num_to_snip)
    //     .collect();

    return survivors;
}

pub fn replace_worst_n_per_cat(
    mut population: Vec<(f64, Evaluation, Antibody)>,
    mut replacements: Vec<(f64, Evaluation, Antibody)>,
    mut snip_list: HashMap<usize, usize>,
) -> Vec<(f64, Evaluation, Antibody)> {
    population.sort_by(|(score_a, _, _), (score_b, _, _)| score_a.total_cmp(score_b));

    let mut kill_list: Vec<_> = Vec::new();
    let mut cur_idx = 0;
    while snip_list.keys().len() > 0 {
        if let Some((_, _, cell)) = population.get(cur_idx) {
            if let Some(remaining_count) = snip_list.get_mut(&cell.class_label) {
                kill_list.push(cur_idx.clone());
                *remaining_count -= 1;
                if *remaining_count <= 0 {
                    snip_list.remove(&cell.class_label);
                }
            }
        } else {
            break;
        }
        cur_idx += 1;
    }

    for idx in kill_list {
        // let mut parent_value = scored_pop.get_mut(idx).unwrap();
        let (p_score, p_eval, p_cell) = population.get_mut(idx).unwrap();
        let (c_score, c_eval, c_cell) = replacements.pop().unwrap();
        // std::mem::replace(p_score, c_score);
        // std::mem::replace(p_eval, c_eval);
        // std::mem::replace(p_cell, c_cell);
        *p_score = c_score;
        *p_eval = c_eval;
        *p_cell = c_cell;
    }
    // println!("{:?}", replacements.len());

    return population;
}

pub fn replace_if_better_per_cat(
    mut population: Vec<(f64, Evaluation, Antibody)>,
    mut replacements: Vec<(f64, Evaluation, Antibody)>,
    mut snip_list: HashMap<usize, usize>,
    match_counter: &mut MatchCounter,
) -> Vec<(f64, Evaluation, Antibody)> {
    population.sort_by(|(score_a, _, _), (score_b, _, _)| score_a.total_cmp(score_b));

    replacements.sort_by(|(score_a, _, _), (score_b, _, _)| score_a.total_cmp(score_b));

    let mut rep_map: Vec<(usize, usize)> = Vec::new();

    for (label, snip_n) in snip_list {
        let label_pop: Vec<_> = population
            .iter()
            .enumerate()
            .filter(|(_, (_, _, c))| c.class_label == label)
            .collect();
        let label_rep: Vec<_> = replacements
            .iter()
            .enumerate()
            .filter(|(_, (_, _, c))| c.class_label == label)
            .collect();

        let mut pop_cur_idx = 0;
        let mut rep_cur_idx = 0;
        loop {
            let pop_option = label_pop.get(pop_cur_idx);
            let rep_option = label_rep.get(rep_cur_idx);

            if pop_option.is_none() | rep_option.is_none() {
                break;
            }

            let (pop_idx, (pop_score, _, pop_cell)) = pop_option.unwrap();
            let (rep_idx, (rep_score, _, rep_cell)) = rep_option.unwrap();

            if rep_score > pop_score {
                rep_map.push((pop_idx.clone(), rep_idx.clone()));
                pop_cur_idx += 1;
                rep_cur_idx += 1;
            } else {
                rep_cur_idx += 1;
            }
        }
    }

    for (pop_idx, rep_idx) in rep_map {
        // let mut parent_value = scored_pop.get_mut(idx).unwrap();
        let (p_score, p_eval, p_cell) = population.get_mut(pop_idx).unwrap();
        let (c_score, c_eval, c_cell) = replacements.get(rep_idx).unwrap();

        match_counter.remove_evaluations(vec![&p_eval]);
        match_counter.add_evaluations(vec![&c_eval]);
        // std::mem::replace(p_score, c_score);
        // std::mem::replace(p_eval, c_eval);
        // std::mem::replace(p_cell, c_cell);
        *p_score = c_score.clone();
        *p_eval = c_eval.clone();
        *p_cell = c_cell.clone();
    }

    return population;
}

pub fn pick_best_n(
    mut population: Vec<(f64, Evaluation, Antibody)>,
    num_to_pick: usize,
) -> Vec<(f64, Evaluation, Antibody)> {
    population.sort_by(|(score_a, _, _), (score_b, _, _)| score_b.total_cmp(score_a));

    // println!("pre {:?}", population.iter().map(|(a,b,c)| a).collect::<Vec<&f64>>());
    let _picked = population.split_off(num_to_pick);
    // println!("post {:?}", population.iter().map(|(a,b,c)| a).collect::<Vec<&f64>>());

    return population;
}

pub fn selection(
    _params: &Params,
    population: Vec<(f64, Evaluation, Antibody)>,
    _match_mask: &mut Vec<usize>,
) -> Vec<(f64, Evaluation, Antibody)> {
    let (selected, _drained) = elitism_selection(population, &150);
    // selected.extend(pick_n_random(drained, 70).into_iter());
    return selected;
}

pub fn elitism_selection(
    mut population: Vec<(f64, Evaluation, Antibody)>,
    num: &usize,
) -> (
    Vec<(f64, Evaluation, Antibody)>,
    Vec<(f64, Evaluation, Antibody)>,
) {
    population.sort_by(|(score_a, _, _), (score_b, _, _)| score_a.partial_cmp(score_b).unwrap());

    // let mut res_cells  = population.into_iter().map(|(a, b)| b).collect::<Vec<_>>();
    let mut res_cells = population.into_iter().collect::<Vec<_>>();

    // println!("{:?}", res_cells.get(res_cells.len()-1).unwrap());
    // return res_cells.drain(..num).collect();

    let select: Vec<(f64, Evaluation, Antibody)> = res_cells
        .drain((res_cells.len() - num)..res_cells.len())
        .collect();
    if false {
        println!("\n selected: ");
        select.iter().for_each(|(x, _, y)| {
            println!(
                "score {:.5}, genome dim value {:?}",
                x,
                y.dim_values
                    .iter()
                    .map(|v| v.multiplier)
                    .collect::<Vec<_>>()
            );
        });
        println!("\n discard: ");
        res_cells.iter().for_each(|(x, _, y)| {
            println!(
                "score {:.5}, genome dim value {:?}",
                x,
                y.dim_values
                    .iter()
                    .map(|v| v.multiplier)
                    .collect::<Vec<_>>()
            );
        });
        println!();
    }
    return (select, res_cells);
}

pub fn labeled_tournament_pick(
    population: &Vec<(f64, Evaluation, Antibody)>,
    num_to_pick: &usize,
    tournament_size: &usize,
    label_to_filter: Option<&usize>,
) -> Vec<usize> {
    let mut rng = rand::thread_rng();

    let pop_s: usize; // = population.len();
                      // idx map for convenience
    let idx_list: Vec<(usize, f64)>; // = population.iter().enumerate().map(|(idx, (score,_,_))| (idx, score.clone())).collect();

    if let Some(v) = label_to_filter {
        let filtered: Vec<_> = population
            .iter()
            .enumerate()
            .filter(|(_idx, (_score, _eval, antibody))| antibody.class_label == *v)
            .collect();

        let _index_vali: Vec<_> = filtered
            .iter()
            .map(|(_, (_a, _b, c))| c.class_label)
            .collect();

        // println!("for label {:?}", v);
        // println!("{:?}", index_vali);

        pop_s = filtered.len();
        idx_list = filtered
            .iter()
            .map(|(idx, (score, _, _))| (idx.clone(), score.clone()))
            .collect();
    } else {
        idx_list = population
            .iter()
            .enumerate()
            .map(|(idx, (score, _, _))| (idx, score.clone()))
            .collect();
        pop_s = population.len();
    }
    let mut picks: Vec<usize> = Vec::with_capacity(*num_to_pick);

    for _ in 0..*num_to_pick {
        // make pool
        let mut pool: Vec<&(usize, f64)> = (0..*tournament_size)
            .into_iter()
            .map(|_| idx_list.get(rng.gen_range(0..pop_s)).unwrap())
            .collect();
        pool.sort_by(|(_, score_a), (_, score_b)| score_a.total_cmp(score_b));

        let mut cur_fetch_idx: usize = 0;
        //todo this may crash but sort of ok if it does because of issues elsewhere
        while cur_fetch_idx < pool.len() {
            // println!("trial {:?} pool {:?}",cur_fetch_idx, pool);
            // println!("picks {:?}", picks);
            let (cand_idx, _s) = pool.get(cur_fetch_idx).unwrap();
            if picks.contains(cand_idx) {
                cur_fetch_idx += 1;
            } else {
                picks.push(cand_idx.clone());
                break;
            }
        }
    }
    /*
        let index_vali: Vec<_> = parents.iter()
            .map(|idx| scored_pop.get(*idx).unwrap())
            .map(|(a,b,c)|c.class_label)
            .collect();

        println!("{:?}", parents);
        println!("{:?}", index_vali);
        println!();
    */

    return picks;
}

pub fn tournament_pick(
    population: &Vec<(f64, Evaluation, Antibody)>,
    num_to_pick: &usize,
    tournament_size: &usize,
) -> Vec<usize> {
    return labeled_tournament_pick(population, num_to_pick, tournament_size, None);
}
