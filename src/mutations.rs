use rand::distributions::{WeightedError, WeightedIndex};
use rand::seq::SliceRandom;
use rand::{distributions::Distribution, Rng};
use std::cmp::max;
use crate::ais::ParamObj;
use crate::representation::{BCell, DimValueType};


pub fn mutate(params: &ParamObj,score: f64, b_cell: BCell) -> BCell{
    let mut rng = rand::thread_rng();

    let mutated = match rng.gen_range(0..1){
        0 => mutate_offset(params,b_cell, score ),
        1 => mutate_multiplier(params,b_cell, score),
        // 2 => mutate_value_type(b_cell, score),
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

pub fn mutate_multiplier(params: &ParamObj, mut genome: BCell, score: f64 ) -> BCell {

    //println!("genome dim value {:?}", genome.dim_values.iter().map(|v|v.multiplier).collect::<Vec<_>>());
    let mut rng = rand::thread_rng();

    let candidates_dims: Vec<usize> = genome
        .dim_values
        .iter()
        .enumerate()
        .filter(|(n, x)| !(x.value_type == DimValueType::Disabled))
        .map(|(n, x)| n)
        .collect();

    if candidates_dims.len() == 0{
        // println!("no dims to mutate multiplier");
        return genome
    }

    let dim_to_mutate = rng.gen_range(0..candidates_dims.len());

    let change_dim = genome.dim_values.get_mut(dim_to_mutate).unwrap();


    // this shifts the score value from [0, 1] to [0.1, 1]
    let score_reduced_max = ((score*0.9) + 0.1) * params.offset_max_delta;
    let mut multi = score_reduced_max; //rng.gen_range(0.0..score_reduced_max);
    if rng.gen_range(0.0..1.0) > params.offset_flip_prob{
        multi *= -1.0;
    }

    if rng.gen_bool(0.5){
        multi += 1.0;

    }

    change_dim.multiplier *=  multi;

    // println!("genome dim value {:?}", genome.dim_values);
    // println!();
    //println!("genome dim value {:?}", genome.dim_values.iter().map(|v|v.multiplier).collect::<Vec<_>>());
    //println!();
    return genome
}

pub fn mutate_offset(params: &ParamObj, mut genome: BCell, score: f64) -> BCell {
    let mut rng = rand::thread_rng();

    let candidates_dims: Vec<usize> = genome
        .dim_values
        .iter()
        .enumerate()
        .filter(|(n, x)| x.value_type == DimValueType::Circle)
        .map(|(n, x)| n)
        .collect();

    if candidates_dims.len() == 0{
        // println!("no dims to mutate shift");
        return genome
    }

    let dim_to_mutate = rng.gen_range(0..candidates_dims.len());

    let change_dim = genome.dim_values.get_mut(dim_to_mutate).unwrap();

    // this shifts the score value from [0, 1] to [0.1, 1]
    let score_reduced_max = ((score*0.9) + 0.1) * params.multiplier_max_delta;
    let mut multi = rng.gen_range(0.0..score_reduced_max);
    if rng.gen_range(0.0..1.0) > params.multiplier_flip_prob{
        multi *= -1.0;
    }

    change_dim.offset +=  multi;

    return genome
}

pub fn mutate_value_type(params: ParamObj, mut genome: BCell, score: f64) -> BCell {
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
