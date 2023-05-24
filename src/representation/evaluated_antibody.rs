use crate::params::{MutationType, Params};
use std::collections::HashMap;
use std::ops::Range;
use nalgebra::{DVector, DMatrix};
use strum_macros::Display;
use rayon::prelude::*;

use crate::representation::antigen::AntiGen;

use strum_macros::EnumString;
use crate::evaluation::Evaluation;
use crate::params::MutationType::ValueType;
use crate::representation::antibody::{Antibody, DimValueType};
use crate::stupid_mutations::MutationOp;


#[derive(Clone, Debug)]
pub struct EvaluatedAntibody{
    pub evaluation: Evaluation,
    pub antibody: Antibody
}


impl EvaluatedAntibody {

    pub fn is_registered(&self, antigen: &AntiGen) -> bool {
        if let Some((afinVal,label)) = self.evaluation.affinity_ag_map.get(&antigen.id) {
            return if *afinVal <= 0.0 {
                true
            } else {
                false
            }
        }else {
            panic!("non evaluated ag tested")
        }
    }

    pub fn transform(&mut self,antigens: &Vec<AntiGen>, mut_op: &MutationOp, inverse: bool){
        let b_dim = self.antibody.dim_values.get(mut_op.dim).unwrap().clone();

        let mut cur_dim_vals = antigens.iter().map(|ag|{
            let antigen_dim_val = ag.values.get(mut_op.dim).unwrap();
            let current = match b_dim.value_type {
                DimValueType::Disabled => 0.0,
                DimValueType::Open => b_dim.multiplier * (antigen_dim_val - b_dim.offset),
                DimValueType::Circle => {
                    (b_dim.multiplier * (antigen_dim_val - b_dim.offset)).powi(2)
                }
            };
            return current.clone().to_owned();
        });

        if inverse{
            mut_op.inverse_transform(&mut self.antibody);
        }else {
            mut_op.transform(&mut self.antibody);
        }

        let b_dim = self.antibody.dim_values.get(mut_op.dim).unwrap();

        for ( ag, old_dim_afin) in antigens.iter().zip(cur_dim_vals){
            let antigen_dim_val = ag.values.get(mut_op.dim).unwrap();
            let (current_afin, current_label) = self.evaluation.affinity_ag_map.get(&ag.id).unwrap();

             let new_val = match b_dim.value_type {
                DimValueType::Disabled => 0.0,
                DimValueType::Open => b_dim.multiplier * (antigen_dim_val - b_dim.offset),
                DimValueType::Circle => {
                    (b_dim.multiplier * (antigen_dim_val - b_dim.offset)).powi(2)
                }
            };

            let delta = new_val - old_dim_afin;
            let new_afin = current_afin + delta;
            self.evaluation.affinity_ag_map.insert(ag.id,(new_afin, *current_label));
        }
    }

}
