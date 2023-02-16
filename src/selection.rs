
use rand::{
    distributions::{Distribution},
    Rng,
};
use rand::distributions::{WeightedError, WeightedIndex};
use crate::ais::ParamObj;
use crate::representation::BCell;

pub fn selection(params: &ParamObj,population: Vec<(f64,BCell)>) -> Vec<(f64, BCell)>{
    return elitism_selection(
        population,
        &50
    );

}

pub fn elitism_selection(
    mut population: Vec<(f64, BCell)>,
    num: &usize,
) -> Vec<(f64, BCell)> {


    population.sort_by(|(score_a, _), (score_b, _)| score_a.partial_cmp(score_b).unwrap());

    // let mut res_cells  = population.into_iter().map(|(a, b)| b).collect::<Vec<_>>();
    let mut res_cells  = population.into_iter().collect::<Vec<_>>();

    // println!("{:?}", res_cells.get(res_cells.len()-1).unwrap());
    // return res_cells.drain(..num).collect();



    return res_cells.drain((res_cells.len()-num)..res_cells.len()).collect();
}
// pub fn tournament_pick<'a>(
//     population: Vec<(f64, BCell)>,
//     num: &usize,
//     tournament_size: &usize,
//     pick_with_replacement: &bool,
// ) -> Vec<BCell> {
//     let mut rng = rand::thread_rng();
//
//     let mut idx_list = Vec::new();
//     while idx_list.len() < *tournament_size {
//         let idx = rng.gen_range(0..population.len());
//         if *pick_with_replacement || !idx_list.contains(&idx) {
//             idx_list.push(idx.clone());
//         }
//     }
//
//     let mut tournament_pool: Vec<_> = idx_list
//         .iter()
//         .map(|idx| population.get(*idx).unwrap())
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
