use strum_macros::Display;

use crate::representation::antigen::AntiGen;

#[derive(Clone, Copy, PartialEq, Debug, Display)]
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
}

impl Antibody {
    //
    // initializers
    //

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

