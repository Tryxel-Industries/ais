use rand::{distributions::Distribution, Rng};
use crate::ais::{MutationType, Params};

use crate::representation::{BCell, DimValueType};

pub fn mutate(params: &Params, score: f64, b_cell: BCell) -> BCell {
    let mut rng = rand::thread_rng();
    
    let mutated = match params.roll_mutation_type() {
        MutationType::Offset => mutate_offset(&params, b_cell),
        MutationType::Multiplier => mutate_multiplier(&params, b_cell),
        MutationType::ValueType => mutate_value_type(&params, b_cell),
        MutationType::Radius => mutate_radius(&params, b_cell),
        MutationType::Label => mutate_label(&params, b_cell)
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


pub fn mutate_multiplier(params: &Params, mut genome: BCell) -> BCell {
    let mut rng = rand::thread_rng();

    let candidates_dims: Vec<usize> = genome
        .dim_values
        .iter()
        .enumerate()
        .filter(|(_n, x)| !(x.value_type == DimValueType::Disabled))
        .map(|(n, _x)| n)
        .collect();

    if candidates_dims.len() == 0 {
        return genome;
    }

    let dim_to_mutate = rng.gen_range(0..candidates_dims.len());
    let change_dim = genome.dim_values.get_mut(dim_to_mutate).unwrap();

    let  multi = rng.gen_range(params.multiplier_mutation_multiplier_range.clone());
    change_dim.multiplier *= multi;
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
        return genome;
    }

    let dim_to_mutate = rng.gen_range(0..candidates_dims.len());
    let change_dim = genome.dim_values.get_mut(dim_to_mutate).unwrap();

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

pub fn mutate_label(params: &Params, mut genome: BCell) -> BCell {
    let mut rng = rand::thread_rng();

    let dim_type =  params.label_valid_mutations
        .get(rng.gen_range(0..params.label_valid_mutations.len()))
        .unwrap()
        .clone();


    genome.class_label = dim_type;

    return genome;
}

pub fn mutate_radius(params: &Params, mut genome: BCell) -> BCell {
    let mut rng = rand::thread_rng();
    

    let mut multi = rng.gen_range(params.radius_mutation_multiplier_range.clone());

    genome.radius_constant *= multi;

    return genome;
}
