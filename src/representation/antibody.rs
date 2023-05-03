use crate::params::MutationType;
use std::collections::HashMap;
use arrayfire::{af_print, Array, Dim4, tile};
use nalgebra::{DVector, DMatrix};
use strum_macros::Display;

use crate::representation::antigen::AntiGen;

use strum_macros::EnumString;

#[derive(Clone, Copy, PartialEq, Debug, Display, EnumString)]
pub enum DimValueType {
    Disabled,
    Open,
    Circle,
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
    //todo: remove when running hyper optimized
    pub mutation_counter: HashMap<MutationType, usize>,
    pub clone_count: usize,
}

#[derive(Debug)]
pub enum LocalSearchBorder {
    EntersAt(f64),
    LeavesAt(f64),
}

impl LocalSearchBorder {
    pub fn get_value(&self) -> &f64 {
        match self {
            LocalSearchBorder::EntersAt(v) => v,
            LocalSearchBorder::LeavesAt(v) => v,
        }
    }

    pub fn is_same_type(&self, other: &LocalSearchBorder) -> bool {
        return match self {
            LocalSearchBorder::EntersAt(_) => match other {
                LocalSearchBorder::EntersAt(_) => true,
                LocalSearchBorder::LeavesAt(_) => false,
            },
            LocalSearchBorder::LeavesAt(_) => match other {
                LocalSearchBorder::EntersAt(_) => false,
                LocalSearchBorder::LeavesAt(_) => true,
            },
        };
    }
}

pub struct AntibodyVectorValues{
    pub offset: Array<f64>,
    pub multiplier: Array<f64>,
    pub exponent: Array<u8>,
}


impl Antibody {

    pub fn get_as_vector_values(&self) -> AntibodyVectorValues {


      /*  self.dim_values.iter().enumerate().for_each(|(n,v)|{
           println!("n: {:<5} offset: {:<5} value t: {:<5} multi: {:<5} ", n , v.offset, v.value_type, v.multiplier)
        });*/
        let num_dims = self.dim_values.len() as u64;
        let offsets: Vec<_> = self.dim_values.iter().map(|x| x.offset).collect();
        let multipliers: Vec<_> = self.dim_values.iter().map(|x| x.multiplier).collect();
        let exponents: Vec<_> = self.dim_values.iter().map(|x| {
            match x.value_type {
                DimValueType::Disabled => {0}
                DimValueType::Open => {1}
                DimValueType::Circle => {2}
            }
        }).collect();


        let offset_array = Array::<f64>::new(offsets.as_slice(),Dim4::new(&[1, num_dims ,1, 1]));
        let multiplier_array = Array::<f64>::new(multipliers.as_slice(),Dim4::new(&[1, num_dims ,1, 1]));
        let exponent_array = Array::<u8>::new(exponents.as_slice(),Dim4::new(&[1, num_dims ,1, 1]));


        return AntibodyVectorValues{
            offset: offset_array,
            multiplier: multiplier_array,
            exponent: exponent_array,
        }
    }



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

                if res_is_pos != cur_is_pos {
                    // if the solved value is on the other side of 0 return as unsolvable
                    return None;
                } else {
                    if affinity_with_zero_multi <= 0.0 {
                        // By testing what ag's the system matches when the multi is set to 0 we
                        // can quicly figure out if the treshold will leave or enter the ag
                        return Some(LocalSearchBorder::LeavesAt(res_match_multi));
                    } else {
                        return Some(LocalSearchBorder::EntersAt(res_match_multi));
                    }
                }
            }
            DimValueType::Circle => {
                let cd_base = (ag_check_dim_val - cd_offset).powi(2);

                // solve for check dim multi
                let rest_sub_radius_sum = self.radius_constant - roll_sum;

                if rest_sub_radius_sum < 0.0 {
                    // if the rest sub radius is negative it is not solvable
                    return None;
                } else {
                    let res_match_multi = (rest_sub_radius_sum / cd_base).sqrt();
                    return Some(LocalSearchBorder::EntersAt(res_match_multi));
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
        let affinity_dist = self.get_affinity_dist(antigen);

        if affinity_dist == 0.0 {
            // all dims are disabled
            return false;
        }

        let v = affinity_dist - self.radius_constant;
        // println!("roll_s {:?}, radius: {:?}", roll_sum, self.radius_constant);
        if v <= 0.0 {
            return true;
        } else {
            return false;
        }
    }
}
