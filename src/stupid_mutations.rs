use crate::evaluation::{evaluate_antibody, MatchCounter};
use rand::prelude::SliceRandom;
use rand::{distributions::Distribution, Rng};
use rayon::prelude::*;

use crate::params::{MutationType, Params};
use crate::representation::antibody::{Antibody, DimValueType, LocalSearchBorder};
use crate::representation::antigen::AntiGen;
use crate::representation::evaluated_antibody::EvaluatedAntibody;
use crate::scoring::score_antibody;

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

//TODO: fix this
pub fn mutate_orientation(params: &Params, mut genome: Antibody) -> Antibody {
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

    genome
}


pub fn mutate_clone_transform(
    eval_ab: &mut EvaluatedAntibody,
    ab_score: f64,
    params: &Params,
    fitness_scaler: f64,
    antigens: &Vec<AntiGen>,
    match_counter: &MatchCounter,
    n_clones: usize,
) -> f64{
    let mut best_op: Option<MutationOp> = None;
    let mut best_score = ab_score;

    for n  in (0..n_clones) {
        let mut_op = get_mut_op(params, fitness_scaler, &eval_ab.antibody, antigens);
        eval_ab.transform(antigens, &mut_op, false);
        eval_ab.evaluation.update_eval();
        let new_score = score_antibody(&eval_ab, params, match_counter);

        eval_ab.transform(antigens, &mut_op, true);
        if new_score > best_score{
            best_score = new_score.clone();
            best_op = Some(mut_op);
        }
    }

    if let Some(winner_op) = best_op{
        eval_ab.transform(antigens, &winner_op, false);
        eval_ab.evaluation.update_eval();
    } else {
        eval_ab.evaluation.update_eval();
    }

    return best_score;
}

pub fn get_mut_op(
    params: &Params,
    fitness_scaler: f64,
    antibody: &Antibody,
    antigens: &Vec<AntiGen>,
) -> MutationOp {
    let _rng = rand::thread_rng();

    let chosen_mutation = params.roll_mutation_type();

    // let mut_count = antibody
    //     .mutation_counter
    //     .get(&chosen_mutation)
    //     .unwrap_or(&0);

    // antibody
    //     .mutation_counter
    //     .insert(chosen_mutation.clone(), mut_count + 1);

    let mut mutation_op = match chosen_mutation {
        MutationType::Offset => mutate_offset_op(&params, antibody, fitness_scaler),
        MutationType::Multiplier => mutate_multi_op(&params, antibody, fitness_scaler),
        MutationType::MultiplierLocalSearch => {
            mutate_multiplier_local_search_op(&params, antibody, antigens, None)
        }
        MutationType::ValueType => mutate_value_type_op(&params, antibody, antigens, fitness_scaler),
        // MutationType::Radius => mutate_radius(&params, antibody, fitness_scaler),
        // MutationType::Label => mutate_label(&params, antibody),
        _ =>  panic!("alalal"),
    };

    if let Some(mut_op) = mutation_op{

        return mut_op;
    }else {
        return MutationOp{
        transformation_fn: mutate_offset,
        mut_type: MutationType::Offset,
        dim: 0,
        delta: MutationDelta::Value(0.0),
    };
    }

}

pub enum MutationDelta{
    Value(f64),
    DimAndOffset(f64,f64),
    // from dim type, to dim type, local s multi op, flip multi
    DimType((DimValueType, DimValueType, Option<Box<MutationOp>>, bool))
}

pub struct MutationOp{
    pub transformation_fn: fn(&mut Antibody, &MutationDelta, &usize, bool),
    pub mut_type: MutationType,
    pub dim: usize,
    pub delta: MutationDelta,
}

impl MutationOp {
    pub fn transform(&self, antibody: &mut Antibody){
        (self.transformation_fn)(antibody, &self.delta, &self.dim, false)

    }

    pub fn inverse_transform(&self, antibody: &mut Antibody){
        (self.transformation_fn)(antibody, &self.delta, &self.dim, true)

    }

}


//
//      OFFSETT
//

pub fn mutate_offset(mut genome: &mut Antibody, mut_delta: &MutationDelta, dim: &usize, inverse: bool) {
    if let MutationDelta::Value(delta) = mut_delta{
        let change_dim = genome.dim_values.get_mut(*dim).unwrap();

        let applied_delta = if inverse{
            delta *-1.0
        }else {
            *delta
        };
        change_dim.offset += applied_delta;
    }else {
        panic!("invalid mutation delta type")
    }
}

pub fn mutate_offset_op(params: &Params, genome: &Antibody, fitness_scaler: f64) -> Option<MutationOp> {
    let mut rng = rand::thread_rng();
    let candidates_dims: Vec<usize> = genome
        .dim_values
        .iter()
        .enumerate()
        .filter(|(_n, x)| x.value_type == DimValueType::Circle)
        .map(|(n, _x)| n)
        .collect();

    if candidates_dims.len() == 0 {
        return None;
    }

    let dim_to_mutate = candidates_dims.choose(&mut rng).unwrap().to_owned();
    let change_dim = genome.dim_values.get(dim_to_mutate).unwrap();

    let val_delta = rng.gen_range(params.offset_mutation_multiplier_range.clone());
    let scaled_delta = val_delta * fitness_scaler;

    return Some(MutationOp{
        transformation_fn: mutate_offset,
        mut_type: MutationType::Offset,
        dim: dim_to_mutate,
        delta: MutationDelta::Value(scaled_delta),
    });

}


//
//      multi
//

pub fn mutate_multi(genome: &mut Antibody, mut_delta: &MutationDelta, dim: &usize, inverse: bool) {
    if let MutationDelta::Value(delta) = mut_delta{
        let change_dim = genome.dim_values.get_mut(*dim).unwrap();

        let applied_delta = if inverse{
            delta *-1.0
        }else {
            *delta
        };
        change_dim.multiplier += applied_delta;
    }else {
        panic!("invalid mutation delta type")
    }
}

pub fn mutate_multi_op(params: &Params, genome: &Antibody, fitness_scaler: f64) -> Option<MutationOp> {
      let mut rng = rand::thread_rng();

    let candidates_dims: Vec<usize> = genome
        .dim_values
        .iter()
        .enumerate()
        .filter(|(_n, x)| !(x.value_type == DimValueType::Disabled))
        .map(|(n, _x)| n)
        .collect();

    if candidates_dims.len() == 0 {
        return None;
    }


    let dim_to_mutate = candidates_dims.choose(&mut rng).unwrap().to_owned();

    let change_dim = genome.dim_values.get(dim_to_mutate).unwrap();

    let val_delta = rng.gen_range(params.multiplier_mutation_multiplier_range.clone());
    let scaled_delta = val_delta * fitness_scaler;

    return Some(MutationOp{
        transformation_fn: mutate_multi,
        mut_type: MutationType::Multiplier,
        dim: dim_to_mutate,
        delta: MutationDelta::Value(scaled_delta),
    });

}

fn find_optimal_open_dim_multi(mut values: Vec<LocalSearchBorder>) -> (f64, f64) {
    values.sort_by(|a, b| {
        a.get_value()
            .abs()
            .partial_cmp(&b.get_value().abs())
            .unwrap()
    });

    let initial_border = if values.get(0).unwrap().get_value() > &0.0 {
        LocalSearchBorder::LeavesAt(0.0001)
    } else {
        LocalSearchBorder::LeavesAt(-0.0001)
    };
    let initial_matches = values
        .iter()
        .map(|v| v.is_same_type(&initial_border))
        .count();
    let mut best_count = initial_matches;
    let mut best_count_at = &initial_border;

    // println!("\n{:?}", values);
    // println!("inital match c {:?}", best_count);
    // println!("inital val     {:?}", best_count_at);
    // println!("find optimal spacer \n\n");

    let mut current_count = initial_matches;
    for border in &values {
        match border {
            LocalSearchBorder::EntersAt(v) => {
                current_count += 1;
            }
            LocalSearchBorder::LeavesAt(v) => {
                current_count -= 1;
            }
        }

        // println!("chk    {:?}", border);
        if current_count < best_count {
            best_count = current_count.clone();
            best_count_at = border;
            // println!("new best count     {:?}", best_count);
            // println!("new best count val {:?}", best_count_at);
        }
    }

    let multi = best_count_at.get_value().clone();
    let offset_dela = if multi > 0.0 { -0.001 } else { 0.001 };
    // let offset_dela = match best_count_at {
    //     LocalSearchBorder::EntersAt(_) => 0.001,
    //     LocalSearchBorder::LeavesAt(_) => -0.001,
    // };

    return (multi, offset_dela);
}

pub fn mutate_multi_w_offset(genome: &mut Antibody, mut_delta: &MutationDelta, dim: &usize, inverse: bool) {
    if let MutationDelta::DimAndOffset(delta, offset) = mut_delta{
        let change_dim = genome.dim_values.get_mut(*dim).unwrap();

        let (applied_delta, applied_offset) = if inverse{
            (delta *-1.0, offset*-1.0)
        }else {
            (*delta, *offset)
        };
        change_dim.offset += applied_offset;
        change_dim.multiplier += applied_delta;
    }else {
        panic!("invalid mutation delta type")
    }
}


pub fn mutate_multiplier_local_search_op(
    params: &Params,
    genome: &Antibody,
    antigens: &Vec<AntiGen>,
    dim_to_search: Option<usize>,
) -> Option<MutationOp> {
    let mut rng = rand::thread_rng();

    let candidates_dims: Vec<usize> = genome
        .dim_values
        .iter()
        .enumerate()
        .filter(|(_n, x)| x.value_type != DimValueType::Disabled)
        // .inspect(|(a,b)| println!("{:?} - {:?}", a,b.value_type))
        .map(|(n, _x)| n)
        .collect();

    if candidates_dims.len() == 0 {
        return None;
    }

    let dim_to_mutate =
        dim_to_search.unwrap_or_else(|| candidates_dims.choose(&mut rng).unwrap().to_owned());

    let dim_type = genome
        .dim_values
        .get(dim_to_mutate)
        .unwrap()
        .value_type
        .clone();

    // get all the ag's with another label and find the radius needed to reach them
    let mapped_ags: Vec<_> = antigens
        .par_iter()
        // .filter(|ag| evaluation.wrongly_matched.binary_search(&ag.id).is_ok())
        .filter(|ag| ag.class_label != genome.class_label)
        .filter_map(|ag| genome.solve_multi_for_dim_with_antigen(dim_to_mutate, ag))
        .filter(|bdr| bdr.get_value().is_finite())
        .collect();

    if mapped_ags.len() == 0 {
        // no possible matches for the genome to reach with this dim
        return None;
    }

    let (new_multi, new_offsett) = if dim_type == DimValueType::Circle {
        // the bigger the cirle multi the smaller the cirle, so we want to find the biggest value that matches an other class ag
        // and then reduce the size a bit more to not register that ag
        let multi = mapped_ags
            .iter()
            .map(|v| match v {
                LocalSearchBorder::EntersAt(v) => return v,
                LocalSearchBorder::LeavesAt(v) => {
                    panic!("shold not happen, yes i know this is bad, but its efficien yay")
                }
            })
            .filter(|v| v.is_finite())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .and_then(|v| Option::from(v * 1.001))
            .unwrap();
        (multi, 0.0)
    } else if dim_type == DimValueType::Open {
        // the smaller the open multi the smaller the aria covered, so we want to find the smallest value that matches an other class ag
        // and then reduce the size a bit more to not register that ag,
        // because of funkyness with the open intervals we can not do this with tweaking the multiplier so we shift the offsett a bit instead
        find_optimal_open_dim_multi(mapped_ags)
    } else {
        panic!("error in local s mutate")
    };

    let change_dim = genome.dim_values.get(dim_to_mutate).unwrap();

    let offset = if dim_type == DimValueType::Open {
        new_offsett
    }else {
        0.0
    };


    let scaled_delta =  change_dim.multiplier - new_multi;

    return Some(MutationOp{
        transformation_fn: mutate_multi_w_offset,
        mut_type: MutationType::MultiplierLocalSearch,
        dim: dim_to_mutate,
        delta: MutationDelta::DimAndOffset(scaled_delta, offset),
    });
}

//
//      value type
//



pub fn mutate_value_type(genome: &mut Antibody, mut_delta: &MutationDelta, dim: &usize, inverse: bool) {
    if let MutationDelta::DimType((from_type, to_type, ajust_op_option, invert_multi)) = mut_delta{


        if let Some(ajust_op) = ajust_op_option{
            if inverse{
                ajust_op.inverse_transform(genome)
            }else {
                ajust_op.transform(genome)
            };
        };

        let change_dim = genome.dim_values.get_mut(*dim).unwrap();

        if *invert_multi{
            change_dim.multiplier *= -1.0;
        }

        if inverse{
            change_dim.value_type = from_type.clone();
        }else {
            change_dim.value_type = to_type.clone();
        }
    }else {
        panic!("invalid mutation delta type")
    }
}

pub fn mutate_value_type_op(params: &Params, genome: &Antibody, antigens: &Vec<AntiGen>,fitness_scaler: f64) -> Option<MutationOp> {
     let mut rng = rand::thread_rng();

    let dim_to_mutate = rng.gen_range(0..genome.dim_values.len());

    let change_dim = genome.dim_values.get(dim_to_mutate).unwrap();

    let dim_type = params
        .value_type_valid_mutations
        .get(rng.gen_range(0..params.value_type_valid_mutations.len()))
        .unwrap()
        .clone();

    let mut inverse_multi = false;
    let mut local_s_op = None;


    if (change_dim.value_type == DimValueType::Open) & (dim_type != DimValueType::Open) {
        // from open to something else
        if change_dim.multiplier < 0.0 {
            inverse_multi = true;
        }
        // genome.radius_constant =  genome.radius_constant.sqrt();
    } else if (change_dim.value_type != DimValueType::Open) & (dim_type == DimValueType::Open) {
        // from something to open
        if rng.gen::<bool>() {
            inverse_multi = true;
        }
        // genome.radius_constant =  genome.radius_constant.powi(2);
    }


    if params.mutation_value_type_local_search_dim && dim_type != DimValueType::Disabled {
        let search_op = mutate_multiplier_local_search_op(params,genome,antigens, Some(dim_to_mutate));
        local_s_op = search_op.map(|op| Box::new(op));
    }

    return Some(MutationOp{
        transformation_fn: mutate_value_type,
        mut_type: MutationType::ValueType,
        dim: dim_to_mutate,
        delta: MutationDelta::DimType((change_dim.value_type, dim_type, local_s_op, inverse_multi)),
    });

}

pub fn mutate_radius(params: &Params, mut genome: Antibody, fitness_scaler: f64) -> Antibody {
    let mut rng = rand::thread_rng();

    let val_delta = rng.gen_range(params.radius_mutation_multiplier_range.clone());
    let scaled_delta = val_delta * fitness_scaler;
    genome.radius_constant += scaled_delta;

    return genome;
}

pub fn mutate_label(params: &Params, mut genome: Antibody) -> Antibody {
    let mut rng = rand::thread_rng();

    let dim_type = params
        .label_valid_mutations
        .get(rng.gen_range(0..params.label_valid_mutations.len()))
        .unwrap()
        .clone();

    genome.class_label = dim_type;

    return genome;
}

