use crate::params::MutationType;
use std::collections::HashMap;
use std::ops::Range;
use nalgebra::{DVector, DMatrix};
use strum_macros::Display;

use serde::Serialize;
use crate::representation::antigen::AntiGen;

use strum_macros::EnumString;
use crate::params::MutationType::ValueType;

#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash, Serialize, Display, EnumString)]
pub enum DimValueType {
    Disabled,
    Open,
    Circle,
}

#[derive(Clone, Debug)]
pub enum InitType{
    Random,
    Antibody,
    NA,

}
#[derive(Clone, Debug)]
pub struct AntibodyDim {
    // the multiplier for the dim value
    pub multiplier: f64,
    // the shift used for the dim if the value type is a circle
    pub offset: f64,
    // the exponent
    pub value_type: DimValueType,
}


#[derive(Clone, Debug)]
pub struct Antibody {
    pub dim_values: Vec<AntibodyDim>,
    pub radius_constant: f64,
    pub class_label: usize,
    pub boosting_model_alpha: f64,
    pub final_train_label_membership: Option<(f64,f64)>,
    pub final_train_label_affinity: Option<(f64,f64)>,
    pub init_type: InitType,
    //todo: remove when running hyper optimized
    pub mutation_counter: HashMap<MutationType, usize>,
    pub clone_count: usize,
}


#[derive(Debug, Eq, PartialEq)]
pub enum LocalSearchBorderType {
    EntersAt,
    LeavesAt,
}

#[derive(Debug)]
pub struct LocalSearchBorder {
    pub border_type: LocalSearchBorderType,
    pub multiplier: f64,
    pub same_label: bool,
    pub boost_value: f64
}

impl LocalSearchBorder {
}


impl Antibody {


    //
    // initializers
    //

    ///
    /// used for local searching
    ///
    pub fn solve_multi_for_dim_with_antigen(
        &self,
        check_dim: usize,
        antigen: &AntiGen,
    ) -> Option<LocalSearchBorder> {
        let mut roll_sum: f64 = 0.0;

        for i in 0..antigen.values.len() {
            if i == check_dim {
                continue;
            }
            let b_dim = self.dim_values.get(i).unwrap();
            let antigen_dim_val = antigen.values.get(i).unwrap();
            roll_sum += match b_dim.value_type {
                DimValueType::Disabled => 0.0,
                DimValueType::Open => b_dim.multiplier * (antigen_dim_val - b_dim.offset),
                DimValueType::Circle => {
                    (b_dim.multiplier * (antigen_dim_val - b_dim.offset)).powi(2)
                }
            };
        }

        let ag_check_dim_val = antigen.values.get(check_dim).unwrap();
        let ab_check_dim = self.dim_values.get(check_dim).unwrap();
        let cd_value_type = ab_check_dim.value_type;
        let cd_multiplier = ab_check_dim.multiplier;
        let cd_offset = ab_check_dim.offset;

        // if all parts - radius is < 0 there is a match
        let multi = match cd_value_type {
            DimValueType::Disabled => None,
            DimValueType::Open => {
                let cd_base = (ag_check_dim_val - cd_offset);
                let affinity_with_zero_multi = roll_sum - self.radius_constant;

                // solve for check dim multi
                let res_match_multi = (self.radius_constant - roll_sum) / cd_base;

                let res_is_pos = res_match_multi > 0.0;
                let cur_is_pos = cd_multiplier > 0.0;

                return if res_is_pos != cur_is_pos {
                    // if the solved value is on the other side of 0 return as unsolvable
                    None
                } else {
                    let lsb_type = if affinity_with_zero_multi <= 0.0 {
                        // By testing what ag's the system matches when the multi is set to 0 we
                        // can quicly figure out if the treshold will leave or enter the ag

                        LocalSearchBorderType::LeavesAt
                    } else {
                        LocalSearchBorderType::EntersAt
                    };

                    Some(LocalSearchBorder{
                        border_type: lsb_type,
                        multiplier: res_match_multi,
                        same_label: self.class_label==antigen.class_label,
                        boost_value: self.boosting_model_alpha,
                    })

                }
            }
            DimValueType::Circle => {
                let cd_base = (ag_check_dim_val - cd_offset).powi(2);

                // solve for check dim multi
                let rest_sub_radius_sum = roll_sum - self.radius_constant;

                return if rest_sub_radius_sum < 0.0 {
                    // if the rest sub radius is negative it is not solvable
                    None
                } else {
                    let res_match_multi = (rest_sub_radius_sum / cd_base).sqrt();
                    // Some(LocalSearchBorder::EntersAt(res_match_multi, self.class_label == antigen.class_label))

                     let lsb_type = if rest_sub_radius_sum <= 0.0 {
                        LocalSearchBorderType::LeavesAt
                    } else {
                        LocalSearchBorderType::EntersAt
                    };

                    Some(LocalSearchBorder{
                        border_type: lsb_type,
                        multiplier: res_match_multi,
                        same_label: self.class_label==antigen.class_label,
                        boost_value: self.boosting_model_alpha,
                    })
                }


            }
        };

        return multi;
    }

    pub fn get_affinity_dist(&self, antigen: &AntiGen) -> f64 {
        let mut roll_sum: f64 = 0.0;
        for i in 0..antigen.values.len() {
            let b_dim = self.dim_values.get(i).unwrap();
            let antigen_dim_val = antigen.values.get(i).unwrap();
            roll_sum += match b_dim.value_type {
                DimValueType::Disabled => 0.0,
                DimValueType::Open => b_dim.multiplier * (antigen_dim_val - b_dim.offset),
                DimValueType::Circle => {
                    (b_dim.multiplier * (antigen_dim_val - b_dim.offset)).powi(2)
                }
            };
        }

        return roll_sum;

    }
    pub fn test_antigen(&self, antigen: &AntiGen) -> bool {

        if self.dim_values.iter().all(|ab| ab.value_type==DimValueType::Disabled) {
            // all dims are disabled
            return false;
        }

        let affinity_dist = self.get_affinity_dist(antigen);


        let v = affinity_dist - self.radius_constant;
        // println!("roll_s {:?}, radius: {:?}", roll_sum, self.radius_constant);
        return if v <= 0.0 {
            true
        } else {
            false
        }
    }

    pub fn test_antigen_and_get_dist(&self, antigen: &AntiGen) -> (bool, f64) {
        if self.dim_values.iter().all(|ab| ab.value_type == DimValueType::Disabled) {
            // all dims are disabled
            return (false, 10000.0);
        }

        let affinity_dist = self.get_affinity_dist(antigen);


        let v = affinity_dist - self.radius_constant;
        // println!("roll_s {:?}, radius: {:?}", roll_sum, self.radius_constant);
        return if v <= 0.0 {
            (true, v)
        } else {
            (false, v)
        }
    }
}
