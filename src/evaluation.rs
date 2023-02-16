use crate::ais::ParamObj;
use crate::representation::{AntiGen, BCell};

pub fn evaluate_b_cell(params: &ParamObj,antigens: &Vec<AntiGen>, b_cell: &BCell) -> f64{
    let registered_antigens = antigens.iter().filter(|ag| b_cell.test_antigen(ag)).collect::<Vec<_>>();

    let with_same_label = registered_antigens.iter().filter(|ag|ag.class_label==b_cell.class_label ).collect::<Vec<_>>();
    let num_wrong = antigens.len()-with_same_label.len();
    let score = with_same_label.len()as f64/(num_wrong as f64+1.0).powi(2);


    // println!("num reg {:?} same label {:?} other label {:?}",antigens.len(), with_same_label.len(), num_wrong);
    return score
}