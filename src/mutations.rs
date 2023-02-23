use rand::{distributions::Distribution, Rng};
use crate::ais::Params;

use crate::representation::{BCell, DimValueType};

pub fn mutate(params: &Params, score: f64, b_cell: BCell) -> BCell {
    let mut rng = rand::thread_rng();

    let mutated = match rng.gen_range(0..3) {
        0 => mutate_offset(params, b_cell ),
        1 => mutate_multiplier(params, b_cell),
        2 => mutate_value_type(params, b_cell),
        _ => {
            panic!("invalid mut")
        }
    };
    return mutated;
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
    if rand::random::<bool>() {
        return -1;
    } else {
        return 1;
    }
}

pub fn mutate_multiplier(params: &Params, mut genome: BCell) -> BCell {
    //println!("genome dim value {:?}", genome.dim_values.iter().map(|v|v.multiplier).collect::<Vec<_>>());
    let mut rng = rand::thread_rng();

    let candidates_dims: Vec<usize> = genome
        .dim_values
        .iter()
        .enumerate()
        .filter(|(_n, x)| !(x.value_type == DimValueType::Disabled))
        .map(|(n, _x)| n)
        .collect();

    if candidates_dims.len() == 0 {
        // println!("no dims to mutate multiplier");
        return genome;
    }

    let dim_to_mutate = rng.gen_range(0..candidates_dims.len());

    let change_dim = genome.dim_values.get_mut(dim_to_mutate).unwrap();

    let  multi = rng.gen_range(params.multiplier_mutation_multiplier_range.clone());
    change_dim.multiplier *= multi;
    // println!("genome dim value {:?}", genome.dim_values);
    // println!();
    //println!("genome dim value {:?}", genome.dim_values.iter().map(|v|v.multiplier).collect::<Vec<_>>());
    //println!();
    return genome;
}

pub fn mutate_offset(params: &Params, mut genome: BCell) -> BCell {
    let mut rng = rand::thread_rng();

    let candidates_dims: Vec<usize> = genome
        .dim_values
        .iter()
        .enumerate()
        .filter(|(_n, x)| x.value_type == DimValueType::Circle)
        .map(|(n, _x)| n)
        .collect();

    if candidates_dims.len() == 0 {
        // println!("no dims to mutate shift");
        return genome;
    }

    let dim_to_mutate = rng.gen_range(0..candidates_dims.len());

    let change_dim = genome.dim_values.get_mut(dim_to_mutate).unwrap();

    // this shifts the score value from [0, 1] to [0.1, 1]
    // let score_reduced_max = ((score * 0.9) + 0.1) * params.multiplier_max_delta;
    // let mut multi = rng.gen_range(0.0..score_reduced_max);
    let mut multi = rng.gen_range(params.offset_mutation_multiplier_range.clone());


    change_dim.offset *= multi;

    return genome;
}

pub fn mutate_value_type(params: &Params, mut genome: BCell) -> BCell {
    let mut rng = rand::thread_rng();

    let dim_to_mutate = rng.gen_range(0..genome.dim_values.len());

    let change_dim = genome.dim_values.get_mut(dim_to_mutate).unwrap();

    let dim_type =  params.value_type_valid_mutations
                .get(rng.gen_range(0..params.value_type_valid_mutations.len()))
                .unwrap()
                .clone();


    change_dim.value_type = dim_type;

    return genome;
}
