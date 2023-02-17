
use rand::{
    distributions::{Distribution},
    Rng,
};

use crate::ais::ParamObj;
use crate::evaluation::Evaluation;
use crate::representation::BCell;

fn pick_n_random<T>(vec: Vec<T>, n: usize) -> Vec<T>{
    let mut rng = rand::thread_rng();

    let mut idx_list = Vec::new();
    while idx_list.len() < n {
        let idx = rng.gen_range(0..vec.len());
        if !idx_list.contains(&idx) {
            idx_list.push(idx.clone());
        }
    }

    idx_list.sort();

    let picks =  vec.into_iter().enumerate().filter(|(idx, _v)| idx_list.contains(idx)).map(|(_a,b)| b).collect();
    return picks
}



pub fn selection(_params: &ParamObj,population: Vec<(f64, Evaluation,BCell)>) -> Vec<(f64,Evaluation, BCell)>{
    let (mut selected, drained) = elitism_selection(
        population,
        &200
    );
    selected.extend(pick_n_random(drained, 100).into_iter());
    return selected

}

pub fn elitism_selection(
    mut population: Vec<(f64,Evaluation, BCell)>,
    num: &usize,
) -> (Vec<(f64,Evaluation, BCell)>, Vec<(f64,Evaluation, BCell)> ) {


    population.sort_by(|(score_a, _, _), (score_b, _, _)| score_a.partial_cmp(score_b).unwrap());

    // let mut res_cells  = population.into_iter().map(|(a, b)| b).collect::<Vec<_>>();
    let mut res_cells  = population.into_iter().collect::<Vec<_>>();

    // println!("{:?}", res_cells.get(res_cells.len()-1).unwrap());
    // return res_cells.drain(..num).collect();


    let select: Vec<(f64,Evaluation, BCell)> = res_cells.drain((res_cells.len()-num)..res_cells.len()).collect();
    if false {
        println!("\n selected: ");
        select.iter().for_each(|(x, _, y)| {
            println!("score {:.5}, genome dim value {:?}", x,y.dim_values.iter().map(|v| v.multiplier).collect::<Vec<_>>());
        });
        println!("\n discard: ");
        res_cells.iter().for_each(|(x, _, y)| {
            println!("score {:.5}, genome dim value {:?}", x,y.dim_values.iter().map(|v| v.multiplier).collect::<Vec<_>>());
        });
        println!();
    }
    return (select, res_cells);
}
// pub fn tournament_pick(
//     population: &Vec<(f64, BCell)>,
//     num: &usize,
//     tournament_size: &usize,
//     num_tournaments: &usize,
//     picks_per_tournament: &usize,
// ) -> Vec<BCell> {
//     let mut rng = rand::thread_rng();
//
//     let mut idx_list = Vec::new();
//     while idx_list.len() < *tournament_size {
//         let idx = rng.gen_range(0..population.len());
//         if !idx_list.contains(&idx) {
//             idx_list.push(idx.clone());
//         }
//     }
//
//     let mut tournament_pool: Vec<_> = idx_list
//         .iter()
//         .map(|idx| (idx,population.get(*idx).unwrap()[0]))
//         .collect();
//
//     tournament_pool.sort_by(|(score_a, _), (score_b, _)| score_a.partial_cmp(score_b).unwrap());
//     let b_cells  = tournament_pool.iter().map(|(a, b)| b).collect::<Vec<_>>();
//
//     let mut ret = Vec::new();
//     for n in 0..*num {
//         let a = b_cells.get(n).unwrap()
//         ret.push(**a)
//     }
//
//     return ret;
// }
//
