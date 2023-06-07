use crate::params::{MutationType, Params};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Range;
use itertools::any;
use strum_macros::Display;

use crate::representation::antigen::AntiGen;

use crate::evaluation::Evaluation;
use crate::params::MutationType::ValueType;
use crate::representation::antibody::{Antibody, DimValueType, LocalSearchBorder, LocalSearchBorderType};
use crate::stupid_mutations::{MutationDelta, MutationOp};
use strum_macros::EnumString;

#[derive(Clone, Debug)]
pub struct EvaluatedAntibody {
    pub evaluation: Evaluation,
    pub antibody: Antibody,
}

impl EvaluatedAntibody {
    pub fn update_eval(&mut self){
        self.evaluation.update_eval(self.antibody.class_label);
    }

    pub fn get_antigen_local_search_border(
        &self,
        check_dim: usize,
        antigen: &AntiGen,
    ) -> Option<LocalSearchBorder> {
        let (antigen_affinity, label) = self.evaluation.affinity_ag_map.get(&antigen.id).unwrap();

        let ab_dim = self.antibody.dim_values.get(check_dim).unwrap();
        let antigen_dim_val = antigen.values.get(check_dim).unwrap();

        let dim_value = match ab_dim.value_type {
            DimValueType::Disabled => 0.0,
            DimValueType::Open => ab_dim.multiplier * (antigen_dim_val - ab_dim.offset),
            DimValueType::Circle => (ab_dim.multiplier * (antigen_dim_val - ab_dim.offset)).powi(2),
        };

        let afin_wo_dim = antigen_affinity - dim_value;

        // todo: check sign
        let dim_border_multi = match ab_dim.value_type {
            DimValueType::Disabled => 0.0,
            DimValueType::Open => -afin_wo_dim / (antigen_dim_val - ab_dim.offset),
            DimValueType::Circle => {
                ((-afin_wo_dim) / (antigen_dim_val - ab_dim.offset).powi(2)).sqrt()
            }
        };

        let lsb = match ab_dim.value_type {
            DimValueType::Disabled => None,
            DimValueType::Open => {
                /*        let cd_base = (ag_check_dim_val - cd_offset);
                let affinity_with_zero_multi = roll_sum - self.radius_constant;

                // solve for check dim multi
                let res_match_multi = (self.radius_constant - roll_sum) / cd_base;

                let res_is_pos = res_match_multi > 0.0;
                let cur_is_pos = cd_multiplier > 0.0;
                */
                return if false {
                    // res_is_pos != cur_is_pos {
                    // if the solved value is on the other side of 0 return as unsolvable
                    None
                } else {
                    let lsb_type = if *antigen_affinity <= 0.0 {
                        // By testing what ag's the system matches when the multi is set to 0 we
                        // can quicly figure out if the treshold will leave or enter the ag

                        LocalSearchBorderType::LeavesAt
                    } else {
                        LocalSearchBorderType::EntersAt
                    };

                    Some(LocalSearchBorder{
                        border_type: lsb_type,
                        multiplier: dim_border_multi,
                        same_label: self.antibody.class_label==antigen.class_label,
                        boost_value: self.antibody.boosting_model_alpha,
                    })

                };
            }
            DimValueType::Circle => {
                return if !(dim_border_multi.is_finite()) {
                    // | (dim_border_multi <= 0.0) {
                    // if the rest sub radius is negative it is not solvable
                    None
                } else {
                    let lsb_type = if *antigen_affinity <= 0.0 {
                        LocalSearchBorderType::LeavesAt
                    } else {
                        LocalSearchBorderType::EntersAt
                    };

                    Some(LocalSearchBorder{
                        border_type: lsb_type,
                        multiplier: dim_border_multi,
                        same_label: self.antibody.class_label==antigen.class_label,
                        boost_value: self.antibody.boosting_model_alpha,
                    })
                };
            }
        };
        return lsb;
    }

    pub fn is_registered(&self, antigen: &AntiGen) -> bool {
        if let Some((afin_val, label)) = self.evaluation.affinity_ag_map.get(&antigen.id) {
            return if *afin_val <= 0.0 { true } else { false };
        } else {
            panic!("non evaluated ag tested")
        }
    }

    pub fn transform(&mut self, antigens: &Vec<AntiGen>, mut_op: &MutationOp, inverse: bool) {
        match mut_op.mut_type {
            MutationType::Radius => {
                let delta = if let MutationDelta::Value(delta) = mut_op.delta {
                    delta
                } else {
                    panic!("error")
                };

                if inverse {
                    mut_op.inverse_transform(&mut self.antibody);

                    for (afin, label) in self.evaluation.affinity_ag_map.values_mut() {
                        *afin += delta
                    }
                } else {
                    mut_op.transform(&mut self.antibody);
                    for (afin, label) in self.evaluation.affinity_ag_map.values_mut() {
                        *afin -= delta
                    }
                }
            }
            MutationType::Label => {
                if inverse {
                    mut_op.inverse_transform(&mut self.antibody);
                } else {
                    mut_op.transform(&mut self.antibody);
                }
            }
            _ => {
                let b_dim = self.antibody.dim_values.get(mut_op.dim).unwrap().clone();

                let mut cur_dim_vals = antigens.iter().map(|ag| {
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

                if inverse {
                    mut_op.inverse_transform(&mut self.antibody);
                } else {
                    mut_op.transform(&mut self.antibody);
                }

                let b_dim = self.antibody.dim_values.get(mut_op.dim).unwrap();

                for (ag, old_dim_afin) in antigens.iter().zip(cur_dim_vals) {
                    let antigen_dim_val = ag.values.get(mut_op.dim).unwrap();
                    let (current_afin, current_label) =
                        self.evaluation.affinity_ag_map.get(&ag.id).unwrap();

                    let new_val = match b_dim.value_type {
                        DimValueType::Disabled => 0.0,
                        DimValueType::Open => b_dim.multiplier * (antigen_dim_val - b_dim.offset),
                        DimValueType::Circle => {
                            (b_dim.multiplier * (antigen_dim_val - b_dim.offset)).powi(2)
                        }
                    };

                    let delta = new_val - old_dim_afin;
                    let new_afin = current_afin + delta;
                    self.evaluation
                        .affinity_ag_map
                        .insert(ag.id, (new_afin, *current_label));
                }
            }
        }
    }
}
