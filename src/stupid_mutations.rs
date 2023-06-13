use crate::evaluation::{evaluate_antibody, MatchCounter};
use rand::prelude::SliceRandom;
use rand::{distributions::Distribution, Rng, thread_rng};
use rayon::prelude::*;

use crate::params::{MutationType, Params};
use crate::representation::antibody::{Antibody, DimValueType, LocalSearchBorder, LocalSearchBorderType};
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
    let (parent_score,_) = score_antibody(&eval_ab, params, match_counter);;
    let mut best_score = parent_score;

    // let pre = eval_ab.clone();
    //
    // println!("init score: {:}", best_score);
    let parent_cor_matches = eval_ab.evaluation.matched_ids.len();
    let parent_errors_matches = eval_ab.evaluation.wrongly_matched.len();

    // println!("\n\n#####  mut ab:");

    // println!("init struct {:?}", eval_ab.antibody.dim_values);

    for n  in 0..n_clones {
        // println!("mut round:");

        let mut_op = get_mut_op(params, fitness_scaler, &eval_ab, antigens);
        let old_multi = eval_ab.antibody.dim_values.get(mut_op.dim).unwrap().multiplier;

        eval_ab.transform(antigens, &mut_op, false);
        eval_ab.update_eval();
        let (new_score, _) = score_antibody(&eval_ab, params, match_counter);

        let child_cor_matches = eval_ab.evaluation.matched_ids.len();
        let child_errors_matches = eval_ab.evaluation.wrongly_matched.len();
        let new_multi = eval_ab.antibody.dim_values.get(mut_op.dim).unwrap().multiplier;



        eval_ab.transform(antigens, &mut_op, true);

        eval_ab.update_eval();
        let p2 = score_antibody(&eval_ab, params, match_counter).0;
        // if 0.2<(parent_score - p2).abs(){
        //     println!("op type {:?}", mut_op.mut_type);
        //     println!("pre: {:?} post:{:?}", parent_score, p2);
        //
        //     // println!("pre: \n{:?}\nPost:\n{:?}", pre, eval_ab);
        //     println!("pre: \n{:?}\nPost:\n{:?}", pre.antibody, eval_ab.antibody);
        //
        //     let samy_multi = eval_ab.antibody.dim_values.get(mut_op.dim).unwrap().multiplier;
        //     println!("multi pre: {:?} post:{:?}", old_multi, samy_multi);
        //     println!();
        //     println!("parent multi {:>4.4?}, child multi {:>4.4?}, child score: {:>4.4?}, parent score: {:>4.4?}", old_multi, new_multi, new_score, parent_score);
        //     println!("parent matches {:>3?}/{:>3?}, child matches {:>3?}/{:>3?}", parent_cor_matches, parent_errors_matches, child_cor_matches, child_errors_matches);
        //     println!("ab struct for dim {:?} is:\n{:?}", mut_op.dim,eval_ab.antibody.dim_values.get(mut_op.dim).unwrap());
        //
        //
        //     panic!("error")
        // }
        //

        if new_score > best_score{
            best_score = new_score.clone();

            // println!("\nnew best\nlabel {:?}, num cor {:?}  num err {:?} score {:?}\n", eval_ab.antibody.class_label, eval_ab.evaluation.matched_ids.len(), eval_ab.evaluation.wrongly_matched.len(), best_score);

            best_op = Some(mut_op);
        }
    }

    if let Some(winner_op) = best_op{
        eval_ab.transform(antigens, &winner_op, false);
        eval_ab.update_eval();
        eval_ab.antibody.clone_count += 1;
        if let Some(count) = eval_ab.antibody.mutation_counter.get_mut(&winner_op.mut_type){
            *count += 1;
        }else {
            eval_ab.antibody.mutation_counter.insert(winner_op.mut_type, 1);
        }
    } else {

        eval_ab.update_eval();
    }


    // println!("best struct {:?}", eval_ab.antibody.dim_values);

    return best_score;
}

pub fn get_mut_op(
    params: &Params,
    fitness_scaler: f64,
    eval_ab: &EvaluatedAntibody,
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
        MutationType::Offset => mutate_offset_op(&params, &eval_ab.antibody, fitness_scaler),
        MutationType::Multiplier => mutate_multi_op(&params, &eval_ab.antibody, fitness_scaler),
        MutationType::MultiplierLocalSearch => {
            mutate_multiplier_local_search_op(&params, &eval_ab, antigens, None)
        }
        MutationType::ValueType => mutate_value_type_op(&params, &eval_ab, antigens, fitness_scaler),
        MutationType::Radius => mutate_radius_op(&params, &eval_ab.antibody, fitness_scaler),
        MutationType::Label => mutate_label_op(&params, &eval_ab.antibody, fitness_scaler),
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
    DimType((DimValueType, DimValueType, Option<Box<MutationOp>>, bool)),
    Label(usize, usize),
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

fn cor_err_relat(corr: f64, err: f64) -> f64{
    let corr = (corr as f64 + 2.0).ln() * 1.5;
    let err = (err as f64 + 2.0).ln();
    return ((corr)/ ((corr + err).max(1.0))); //* thread_rng().gen_range(0.8..1.2)
}

fn find_optimal_open_dim_multi(mut values: Vec<LocalSearchBorder>, init_multi: &f64, open_dim: bool) -> (f64, f64) {

    // println!("#############  DIM MULTI ###################");
    // println!("lsb vec size {:?}", values.len());


    values.sort_by(|a, b| {
        a.multiplier
            .partial_cmp(&b.multiplier)
            .unwrap()
    });

    let mut neg = Vec::new();
    let mut pos = Vec::new();
    let mut end_idx = values.len()-1;

    for (idx, lbs) in values.iter().enumerate(){
        if lbs.multiplier < *init_multi{
            neg.push(lbs);
        }else {
            end_idx = idx;
            break
        }
    }

    if end_idx == 0{
        pos.extend(values.iter())
    } else if end_idx < values.len() -1 {
        pos.extend(values.iter().skip(end_idx))
    }


    neg.reverse();

    let initial_border = LocalSearchBorder{
        border_type: LocalSearchBorderType::LeavesAt,
        multiplier: *init_multi,
        same_label: false,
        boost_value: 1.0,
    };

    let initial_matches_stream = values
        .iter()
        .filter(|v| v.border_type == LocalSearchBorderType::LeavesAt);

    let initial_matches_all = initial_matches_stream.clone()
        .map(|lsb| lsb.boost_value).sum::<f64>();

    let initial_matches_cor = initial_matches_stream.filter_map(|lsb| if lsb.same_label{Some(lsb.boost_value)}else { None }).sum::<f64>();
    let initial_matches_err = initial_matches_all - initial_matches_cor;

    let mut best_score = cor_err_relat(initial_matches_cor, initial_matches_err);
    let mut cor_at_best_score = initial_matches_cor;
    let mut best_count_at = &initial_border;

    let mut current_count_corr = initial_matches_cor;
    let mut current_count_err = initial_matches_err;

    // println!("\n{:?}", values);
    // println!("\npos{:?}", pos);
    // println!("\nneg{:?}", neg);
    // println!("inital pos       {:?}", current_count_corr);
    // println!("inital neg       {:?}", current_count_err);
    // println!("inital count all {:?}", initial_matches_all);
    // println!();
    // println!("inital match c {:?}", best_score);
    // println!("inital val     {:?}", best_count_at);
    // println!("find optimal spacer \n\n");


    for lsb in neg{
        match lsb.border_type {
            LocalSearchBorderType::EntersAt => {
               if lsb.same_label{
                   current_count_corr += lsb.boost_value;
               }else {
                   current_count_err += lsb.boost_value;
               }
            }
            LocalSearchBorderType::LeavesAt => {
                if lsb.same_label{
                    current_count_corr -= lsb.boost_value;
                }else {
                    current_count_err -= lsb.boost_value;
                }
            }
        }


        let score = cor_err_relat(current_count_corr, current_count_err) ;
        // println!("score {:>3.2?} count {:>5?}/{:<5?}, chk    {:?}", score,current_count_corr, current_count_err, lsb);
        if score > best_score {
            best_score = score;
            best_count_at = lsb;
            // println!("new best count     {:>3?}/{:<3?}, ", current_count_corr, current_count_err);
            // println!("new best count val {:?}", best_count_at);
        }
    }


    let mut current_count_corr = initial_matches_cor;
    let mut current_count_err = initial_matches_err;
    for lsb in pos {
        match lsb.border_type {
            LocalSearchBorderType::EntersAt => {
                if lsb.same_label{
                    current_count_corr += lsb.boost_value;
                }else {
                    current_count_err += lsb.boost_value;
                }
            }
            LocalSearchBorderType::LeavesAt => {
                if lsb.same_label{
                    current_count_corr -= lsb.boost_value;
                }else {
                    current_count_err -= lsb.boost_value;
                }
            }
        }



        // println!("count {:>5?}, chk    {:?}",current_count, lsb);
        let score = cor_err_relat(current_count_corr, current_count_err);
        if score > best_score {
            best_score = score;
            best_count_at = lsb;

            // println!("new best count     {:?}", best_count);
            // println!("new best count val {:?}", best_count_at);
        }
    }


    /*for border in &values {
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
*/
    let multi_raw = best_count_at.multiplier.clone();
    let (multi, offset_dela) = if best_count_at.multiplier == *init_multi{
        (1.0, 0.0)
    } else{
        if open_dim{
            if multi_raw > 0.0 { (multi_raw,-0.001) } else { (multi_raw,0.001) }
        }else {
            (multi_raw*0.999, 0.0)
        }
    };

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
    genome: &EvaluatedAntibody,
    antigens: &Vec<AntiGen>,
    dim_to_search: Option<usize>,
) -> Option<MutationOp> {
    let mut rng = rand::thread_rng();

    let candidates_dims: Vec<usize> = genome.antibody
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

    let dim_type = genome.antibody
        .dim_values
        .get(dim_to_mutate)
        .unwrap()
        .value_type
        .clone();

    // get all the ag's with another label and find the radius needed to reach them
    let mapped_ags: Vec<_> = antigens
        .par_iter()
        // .filter(|ag| evaluation.wrongly_matched.binary_search(&ag.id).is_ok())
        // .filter(|ag| ag.class_label != genome.antibody.class_label)
        .filter_map(|ag| genome.get_antigen_local_search_border(dim_to_mutate, ag))
        .filter(|bdr| bdr.multiplier.is_finite())
        .filter(|bdr| bdr.multiplier.abs() < 1000.0)
        .collect();

    // println!("invalid count {:?}", mapped_ags.iter().filter(|bdr| !bdr.get_value().is_finite()).count());
    // println!("tot  count    {:?}", mapped_ags.len());

    if mapped_ags.len() == 0 {
        // no possible matches for the genome to reach with this dim
        return None;
    }



    let multi = genome.antibody
        .dim_values
        .get(dim_to_mutate)
        .unwrap()
        .multiplier;

    let (new_multi, new_offsett) = if dim_type == DimValueType::Circle {
        // the bigger the cirle multi the smaller the cirle, so we want to find the biggest value that matches an other class ag
        // and then reduce the size a bit more to not register that ag
  /*      let multi_optn = mapped_ags
            .iter()
            // .filter(|lsb| !*lsb.is_corr())
            // .map(|v| match v {
            //     LocalSearchBorder::EntersAt(v, corr) => return v,
            //     LocalSearchBorder::LeavesAt(v, corr) => {
            //         panic!("shold not happen, yes i know this is bad, but its efficien yay")
            //     }
            // })
            .map(|v| match v {
                LocalSearchBorder::EntersAt(v, corr) => LocalSearchBorder::EntersAt(v.abs(), *corr),
                LocalSearchBorder::LeavesAt(v, corr) => LocalSearchBorder::LeavesAt(v.abs(), *corr)
            })
            // .inspect(|lsb| println!("lsb: {:?}", lsb))
            .filter(|v| v.is_finite())
            // this is max because the biggest value in the wrong labels equates to the smallest valid radius
            .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
            .and_then(|v| Option::from(v * 1.001));*/


        let mut lsb_list: Vec<_> = mapped_ags;
        lsb_list.iter_mut().for_each(|lsb| lsb.multiplier = lsb.multiplier.abs());

        // let lsb_list: Vec<_> = mapped_ags
        //     .iter()
        //     .map(|lsb| lsb.multiplier = lsb.multiplier.abs())
        //     .map(|v| match v {
        //         LocalSearchBorder::EntersAt(v, corr) => LocalSearchBorder::EntersAt(v.abs(), *corr),
        //         LocalSearchBorder::LeavesAt(v, corr) => LocalSearchBorder::LeavesAt(v.abs(), *corr)
        //     }).collect();

        find_optimal_open_dim_multi(lsb_list, &multi, false)
        // if let Some(multi) = multi_optn{
        //     (multi, 0.0)
        // }else {
        //     return None;
        // }

        // println!("final multi: {:?}", multi);
    } else if dim_type == DimValueType::Open {
        // the smaller the open multi the smaller the aria covered, so we want to find the smallest value that matches an other class ag
        // and then reduce the size a bit more to not register that ag,
        // because of funkyness with the open intervals we can not do this with tweaking the multiplier so we shift the offsett a bit instead

        find_optimal_open_dim_multi(mapped_ags, &multi, true)
    } else {
        panic!("error in local s mutate")
    };

    let change_dim = genome.antibody.dim_values.get(dim_to_mutate).unwrap();

    let offset = if dim_type == DimValueType::Open {
        new_offsett
    }else {
        0.0
    };


    let scaled_delta = new_multi- change_dim.multiplier;

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

        let change_dim = genome.dim_values.get_mut(*dim).unwrap();

        if inverse{

            change_dim.value_type = from_type.clone();
            if let Some(ajust_op) = ajust_op_option{
                ajust_op.inverse_transform(genome)
            }

        }else {
            change_dim.value_type = to_type.clone();
            if let Some(ajust_op) = ajust_op_option{
                ajust_op.transform(genome)
            }
        }

 /*       let change_dim = genome.dim_values.get_mut(*dim).unwrap();
        if *invert_multi{
            change_dim.multiplier *= -1.0;
        }*/


    }else {
        panic!("invalid mutation delta type")
    }
}

pub fn mutate_value_type_op(params: &Params, genome: &EvaluatedAntibody, antigens: &Vec<AntiGen>,fitness_scaler: f64) -> Option<MutationOp> {
     let mut rng = rand::thread_rng();

    let dim_to_mutate = rng.gen_range(0..genome.antibody.dim_values.len());

    let change_dim = genome.antibody.dim_values.get(dim_to_mutate).unwrap();

    let dim_type = params
        .value_type_valid_mutations
        .get(rng.gen_range(0..params.value_type_valid_mutations.len()))
        .unwrap()
        .clone();

    let mut inverse_multi = false;
    let mut local_s_op = None;


    // if (change_dim.value_type == DimValueType::Open) & (dim_type != DimValueType::Open) {
    //     // from open to something else
    //     if change_dim.multiplier < 0.0 {
    //         inverse_multi = true;
    //     }
    //     // genome.radius_constant =  genome.radius_constant.sqrt();
    // } else if (change_dim.value_type != DimValueType::Open) & (dim_type == DimValueType::Open) {
    //     // from something to open
    //     if rng.gen::<bool>() {
    //         inverse_multi = true;
    //     }
    //     // genome.radius_constant =  genome.radius_constant.powi(2);
    // }
    //

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

//
//      Radius
//

pub fn mutate_radius(mut genome: &mut Antibody, mut_delta: &MutationDelta, dim: &usize, inverse: bool) {
    if let MutationDelta::Value(delta) = mut_delta{
        let applied_delta = if inverse{
            delta *-1.0
        }else {
            *delta
        };
        genome.radius_constant += applied_delta;
    }else {
        panic!("invalid mutation delta type")
    }
}

pub fn mutate_radius_op(params: &Params, genome: &Antibody, fitness_scaler: f64) -> Option<MutationOp> {
    let mut rng = rand::thread_rng();

    let val_delta = rng.gen_range(params.radius_mutation_multiplier_range.clone());
    let scaled_delta = val_delta * fitness_scaler;

    return Some(MutationOp{
        transformation_fn: mutate_radius,
        mut_type: MutationType::Radius,
        dim: 0,
        delta: MutationDelta::Value(scaled_delta),
    });

}






pub fn mutate_label(mut genome: &mut Antibody, mut_delta: &MutationDelta, dim: &usize, inverse: bool) {

    if let MutationDelta::Label(old,new) = mut_delta{
        let applied_delta = if inverse{
            genome.class_label = old.clone();
        }else {
            genome.class_label = new.clone();
        };
    }else {
        panic!("invalid mutation delta type")
    }
}

pub fn mutate_label_op(params: &Params, genome: &Antibody, fitness_scaler: f64) -> Option<MutationOp> {

    let old_label = genome.class_label;

    let mut rng = rand::thread_rng();

    let new_label = params
        .label_valid_mutations
        .get(rng.gen_range(0..params.label_valid_mutations.len()))
        .unwrap()
        .clone();

    return Some(MutationOp{
        transformation_fn: mutate_label,
        mut_type: MutationType::Label,
        dim: 0,
        delta: MutationDelta::Label(old_label,new_label),
    });

}



