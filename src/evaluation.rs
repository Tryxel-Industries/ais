use rand::distributions::Uniform;
use crate::bucket_empire::{BucketEmpireOfficialRangeNotationSystemClasses, BucketKing};
use crate::representation::{AntiGen, BCell, DimValueType};
use rayon::prelude::*;
use crate::ais::Params;

#[derive(Clone, Debug)]
pub struct Evaluation {
    pub matched_ids: Vec<usize>,
    pub wrongly_matched: Vec<usize>,
}

pub fn evaluate_b_cell(
    bk: &BucketKing<AntiGen>,
    _params: &Params,
    antigens: &Vec<AntiGen>,
    b_cell: &BCell,
) -> Evaluation {
    // todo: flip offset mabye
    let dim_radus = b_cell
        .dim_values
        .iter()
        .map(|dv| {
            return match dv.value_type {
                DimValueType::Disabled => BucketEmpireOfficialRangeNotationSystemClasses::Open,
                DimValueType::Open => {
                    // return  BucketEmpireOfficialRangeNotationSystemClasses::Open;
                    let value = (b_cell.radius_constant / dv.multiplier) - dv.offset;
                    if dv.multiplier < 0.0 {
                        return BucketEmpireOfficialRangeNotationSystemClasses::UpperBound(value);
                    } else {
                        return BucketEmpireOfficialRangeNotationSystemClasses::LowerBound(value);
                    }
                }
                DimValueType::Circle => {
                    let value = (b_cell.radius_constant.sqrt() / dv.multiplier);
                    return BucketEmpireOfficialRangeNotationSystemClasses::Symmetric((dv.offset- value, dv.offset+value));
                }
            };
        })
        .collect();

    let cell_values = b_cell
        .dim_values
        .iter()
        .map(|cell| cell.offset)
        .collect::<Vec<f64>>();
    let mut idx_list = bk
        .get_potential_matches_indexes_with_raw_values(&cell_values, &dim_radus)
        .unwrap();
    idx_list.sort();

    let registered_antigens = antigens
        .iter()
        // .filter(|ag| idx_list.binary_search(&ag.id).is_ok())
        .filter(|ag| b_cell.test_antigen(ag))
        .collect::<Vec<_>>();

    let mut corr_matched = Vec::with_capacity(registered_antigens.len());
    let mut wrong_matched = Vec::with_capacity(registered_antigens.len());

    for registered_antigen in registered_antigens {
        if registered_antigen.class_label == b_cell.class_label {
            corr_matched.push(registered_antigen.id)
        } else {
            wrong_matched.push(registered_antigen.id)
        }
    }
    // let with_same_label = registered_antigens.iter().filter(|ag|ag.class_label==b_cell.class_label ).collect::<Vec<_>>();
    // let num_wrong = registered_antigens.iter().filter(|ag|ag.class_label!=b_cell.class_label ).collect::<Vec<_>>();
    // let num_wrong = registered_antigens.len()-with_same_label.len();

    // let score = (with_same_label.len()as f64) - ((num_wrong as f64/2.0));

    // println!("matched {:?}\n", corr_matched);
    let ret_evaluation = Evaluation {
        matched_ids: corr_matched,
        wrongly_matched: wrong_matched,
    };
    // println!("num reg {:?} same label {:?} other label {:?}",antigens.len(), with_same_label.len(), num_wrong);
    return ret_evaluation;
}
pub fn merge_evaluation_matches(evaluations: Vec<&Evaluation>, mask: Vec<usize>) -> Vec<usize> {

    let merged =
        evaluations
            .iter()
            .filter(|e| e.wrongly_matched.len() == 0)
            .map(|e| &e.matched_ids)
            .fold(mask, |mut acc, b| {
                b.iter().for_each(|v| {
                    if let Some(elem) = acc.get_mut(*v) {
                        *elem += 1;
                    }
                });
                return acc;
            });
    return merged;
}

pub fn gen_merge_mask(scored_population: &Vec<(Evaluation, BCell)>) -> Vec<usize>{
    let evaluations = scored_population
        .iter()
        .map(|(b, _)| b)
        .collect::<Vec<&Evaluation>>();

    let max_id = evaluations
        .iter()
        .map(|e| e.matched_ids.iter().max().unwrap_or(&0))
        .max()
        .unwrap()
        + 1;
    let mask = vec![0usize; max_id];

    return merge_evaluation_matches(evaluations, mask);
}


pub fn expand_merge_mask(scored_population: &Vec<(Evaluation, BCell)>, mask: Vec<usize>) -> Vec<usize>{
    let evaluations = scored_population
        .iter()
        .map(|(b, _)| b)
        .collect::<Vec<&Evaluation>>();
    return merge_evaluation_matches(evaluations, mask);
}
pub fn score_b_cells(scored_population: Vec<(Evaluation, BCell)>, merged_mask: &Vec<usize>) -> Vec<(f64, Evaluation, BCell)>{

    // println!("{:?}", merged_mask);
    // println!("len {:?}", merged_mask.len());
    // println!("sum {:?}", merged_mask.iter().sum::<usize>());
    // println!("match -s: ");
    let scored = scored_population
        // .into_iter()
        .into_par_iter() // TODO: set paralell
        .map(|(eval, cell)| {
            let mut bonus_sum: f64 = 0.0;
            let mut base_sum: f64 = 0.0;

            let mut n_shared = 0;
            let mut roll_shared = 0;

            for mid in &eval.matched_ids {
                let sharers = merged_mask.get(*mid).unwrap_or(&0);
                if *sharers > 1 {
                    n_shared += 1 as usize;
                    roll_shared += sharers;
                    // matched_sum += (0.5+(1.0 / ((*sharers as f64) / 2.0))).min(1.0)
                    // matched_sum += (1.0 / ((*sharers as f64) / 2.0)).max(0.5)

                    let delta =  (1.0/ *sharers as f64).min(0.0);
                    // let delta =  (1.0 / ((*sharers as f64) / 2.0)).max(0.5);


                    bonus_sum += delta;
                } else {
                    bonus_sum += 5.0;
                }
                base_sum += 1.0;

            }
            let n_wrong = eval.wrongly_matched.len() as f64;
            let n_right = eval.matched_ids.len() as f64;

            let mut purity = n_right / n_wrong;
            let mut accuracy = n_right / (n_wrong + n_right);
            let mut crowdedness = 1.0 / (roll_shared as f64 / n_shared as f64);

            if !purity.is_finite() {
                purity = 0.0;
            }
            if !accuracy.is_finite() {
                accuracy = 0.0;
            }
            if !crowdedness.is_finite() {
                crowdedness = 0.0;
            }

            // let score = crowdedness+purity+accuracy ;
            // let score = ((base_sum - n_wrong)+(bonus_sum*0.5))*accuracy ;
            let score = ((bonus_sum) / (n_wrong + 1.0));
            // let score = (matched_sum - n_wrong).max(0.0) ;

            // println!("\n score {:?}", score);
            return (score, eval, cell);
        })
        .collect();

    return scored;
}
