use rand::distributions::{WeightedError, WeightedIndex};
use rand::seq::SliceRandom;
use rand::{distributions::Distribution, Rng};
use std::cmp::max;
use crate::ais::ParamObj;
use crate::representation::{BCell, DimValueType};



pub fn mutate(params: &ParamObj,score: f64, b_cell: BCell) -> BCell{
    let mut rng = rand::thread_rng();

    let mutated = match rng.gen_range(0..2){
        0 => mutate_offset(b_cell, score),
        1 => mutate_multiplier(b_cell, score),
        2 => mutate_value_type(b_cell),
        _ => {
            panic!("invalid mut")
        }
    };
    return mutated
}
pub fn get_rand_range(max: usize) -> (usize, usize) {
    let mut rng = rand::thread_rng();
    let point_1: usize = rng.gen_range(0..max);
    let point_2: usize = rng.gen_range(0..max);

    return if point_1 > point_2 {
        (point_2, point_1)
    } else {
        (point_1, point_2)
    };
}

fn flip_coin() -> i32 {
    if rand::random::<bool>(){
        return -1;
    } else {
        return 1;
    }
}

const MULTIPLIER_MUTATION_DELTA: f64 = 0.2;
pub fn mutate_multiplier(mut genome: BCell, score: f64) -> BCell {
    let mut rng = rand::thread_rng();

    let candidates_dims: Vec<usize> = genome
        .dim_values
        .iter()
        .enumerate()
        .filter(|(n, x)| !(x.value_type == DimValueType::Disabled))
        .map(|(n, x)| n)
        .collect();

    if candidates_dims.len() == 0{
        println!("no dims to mutate multiplier");
        return genome
    }

    let dim_to_mutate = rng.gen_range(0..candidates_dims.len());

    let change_dim = genome.dim_values.get_mut(dim_to_mutate).unwrap();

    let multi = flip_coin() as f64 * MULTIPLIER_MUTATION_DELTA;
    change_dim.multiplier +=  (multi* (1.0-score));

    return genome
}

const OFFSET_MUTATION_DELTA: f64 = 0.2;
pub fn mutate_offset(mut genome: BCell, score: f64) -> BCell {
    let mut rng = rand::thread_rng();

    let candidates_dims: Vec<usize> = genome
        .dim_values
        .iter()
        .enumerate()
        .filter(|(n, x)| x.value_type == DimValueType::Circle)
        .map(|(n, x)| n)
        .collect();

    if candidates_dims.len() == 0{
        println!("no dims to mutate shift");
        return genome
    }

    let dim_to_mutate = rng.gen_range(0..candidates_dims.len());

    let change_dim = genome.dim_values.get_mut(dim_to_mutate).unwrap();

    let multi = flip_coin() as f64 * OFFSET_MUTATION_DELTA;
    change_dim.offset +=  (multi*(1.0-score));

    return genome
}

pub fn mutate_value_type(mut genome: BCell) -> BCell {
    let mut rng = rand::thread_rng();

    let dim_to_mutate = rng.gen_range(0..genome.dim_values.len());

    let change_dim = genome.dim_values.get_mut(dim_to_mutate).unwrap();


    let dim_type = match rng.gen_range(0..=2) {
            0 => DimValueType::Disabled,
            1 => DimValueType::Open,
            _ => DimValueType::Circle,
        };
    change_dim.value_type =  dim_type;

    return genome
}
