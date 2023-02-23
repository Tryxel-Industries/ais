use rand::{distributions::Distribution, Rng, thread_rng};
use rand::prelude::SliceRandom;
use crate::ais::Params;

use crate::evaluation::Evaluation;
use crate::representation::BCell;

fn pick_n_random<T>(vec: Vec<T>, n: usize) -> Vec<T> {
    let mut rng = rand::thread_rng();

    let mut idx_list = Vec::new();
    while idx_list.len() < n {
        let idx = rng.gen_range(0..vec.len());
        if !idx_list.contains(&idx) {
            idx_list.push(idx.clone());
        }
    }

    idx_list.sort();

    let picks = vec
        .into_iter()
        .enumerate()
        .filter(|(idx, _v)| idx_list.contains(idx))
        .map(|(_a, b)| b)
        .collect();
    return picks;
}

pub fn kill_by_mask_yo(mut population: Vec<(f64, Evaluation, BCell)>, match_mask: &mut Vec<usize>) -> Vec<(f64, Evaluation, BCell)>{
    let mut survivors: Vec<(f64, Evaluation, BCell)> = Vec::new();


    population.shuffle(&mut thread_rng());

    for (score, eval, cell) in population.into_iter(){
        let mut keep = false;
        for idx in &eval.matched_ids{
            if let Some(cnt) =  match_mask.get(*idx){
                if *cnt <= 1{
                    keep = true;
                    break;
                }
            }
        }


        if keep{
            survivors.push((score, eval, cell))
        }else {
            // all the covered vars are covered elsewhere
            for idx in &eval.matched_ids{
            if let Some( cnt) =  match_mask.get_mut(*idx){
                *cnt -= 1
            }
        }
        }


    }

    return survivors;
}

pub fn snip_worst_n(
    mut population: Vec<(f64, Evaluation, BCell)>,
    num_to_snip: usize,
) -> Vec<(f64, Evaluation, BCell)>{
    population.sort_by(|(score_a, _, _) ,(score_b,_,_)| score_a.total_cmp(score_b));


    // println!("pre {:?}", population.iter().map(|(a,b,c)| a).collect::<Vec<&f64>>());
    let survivors = population.split_off(num_to_snip);
    // println!("post {:?}", survivors.iter().map(|(a,b,c)| a).collect::<Vec<&f64>>());
    // let snipped: Vec<(f64, Evaluation, BCell)> = population
    //     .drain(0..num_to_snip)
    //     .collect();

    return survivors;
}


pub fn pick_best_n(
    mut population: Vec<(f64, Evaluation, BCell)>,
    num_to_pick: usize,
) -> Vec<(f64, Evaluation, BCell)>{
    population.sort_by(|(score_a, _, _) ,(score_b,_,_)| score_b.total_cmp(score_a));

    // println!("pre {:?}", population.iter().map(|(a,b,c)| a).collect::<Vec<&f64>>());
    let picked = population.split_off(num_to_pick);
    // println!("post {:?}", population.iter().map(|(a,b,c)| a).collect::<Vec<&f64>>());

    return population;
}

pub fn selection(
    _params: &Params,
    population: Vec<(f64, Evaluation, BCell)>,
    match_mask: &mut Vec<usize>,
) -> Vec<(f64, Evaluation, BCell)> {
    let (mut selected, drained) = elitism_selection(population, &150);
    // selected.extend(pick_n_random(drained, 70).into_iter());
    return selected;
}

pub fn elitism_selection(
    mut population: Vec<(f64, Evaluation, BCell)>,
    num: &usize,
) -> (Vec<(f64, Evaluation, BCell)>, Vec<(f64, Evaluation, BCell)>) {
    population.sort_by(|(score_a, _, _), (score_b, _, _)| score_a.partial_cmp(score_b).unwrap());

    // let mut res_cells  = population.into_iter().map(|(a, b)| b).collect::<Vec<_>>();
    let mut res_cells = population.into_iter().collect::<Vec<_>>();


    // println!("{:?}", res_cells.get(res_cells.len()-1).unwrap());
    // return res_cells.drain(..num).collect();

    let select: Vec<(f64, Evaluation, BCell)> = res_cells
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



pub fn tournament_pick(
    population: &Vec<(f64, Evaluation, BCell)>,
    num_to_pick: &usize,
    tournament_size: &usize,
) -> Vec<usize> {
    let mut rng = rand::thread_rng();

    // idx map for convenience
    let idx_list: Vec<(usize, f64)> = population.iter().enumerate().map(|(idx, (score,_,_))| (idx, score.clone())).collect();

    let mut picks: Vec<usize> = Vec::with_capacity(*num_to_pick);

    for _ in (0..*num_to_pick){
        // make pool
        let mut pool: Vec<&(usize,f64)> = (0..*tournament_size).into_iter()
            .map(|_| idx_list.get(rng.gen_range(0..population.len())).unwrap()).collect();
        pool.sort_by(|(_,score_a) ,(_,score_b)| score_a.total_cmp(score_b));

        let mut cur_fetch_idx: usize = 0;
        //todo this may crash but sort of ok if it does because of issues elsewhere
        while cur_fetch_idx < pool.len()  {
            // println!("trial {:?} pool {:?}",cur_fetch_idx, pool);
            // println!("picks {:?}", picks);
            let (cand_idx,s) = pool.get(cur_fetch_idx).unwrap();
            if picks.contains(cand_idx){
                cur_fetch_idx += 1;
            }else{
                picks.push(cand_idx.clone());
                break
            }
        }
    }



    return picks;
}

