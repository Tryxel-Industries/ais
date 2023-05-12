use std::collections::HashMap;
use std::ops::RangeInclusive;

use nalgebra::{DMatrix, DVector};
use rand::distributions::{Distribution, Uniform};
use rand::{Rng, random};

use rayon::prelude::*;
use crate::params::Params;

use crate::representation::antibody::{Antibody, AntibodyDim, DimValueType};
use crate::representation::antigen::AntiGen;

pub struct AntibodyFactory {
    params: Params,
    n_dims: usize,

    antibody_multiplier_ranges: Vec<Uniform<f64>>,
    antibody_radius_range: Uniform<f64>,
    //antibody_allowed_value_types: Vec<DimValueType>,

    rng_multiplier_ranges: Vec<Uniform<f64>>,
    rng_offset_ranges: Vec<Uniform<f64>>,
    rng_radius_range: Uniform<f64>,
   // rng_allowed_value_types: Vec<DimValueType>,

    class_labels: Vec<usize>,
}

impl AntibodyFactory {
    pub fn new(
        params: &Params,
        n_dims: usize,
        class_labels: Vec<usize>,
    ) -> Self {


        let range_to_uniform = |range: RangeInclusive<f64>| {
            return vec![range; n_dims]
                .iter()
                .map(|x| Uniform::new_inclusive(x.start(), x.end()))
                .collect::<Vec<Uniform<f64>>>();
        };

        return Self {
            params: params.clone(),
            n_dims,
            antibody_multiplier_ranges: range_to_uniform(params.antibody_ag_init_multiplier_range.clone()),
            antibody_radius_range: Uniform::new_inclusive(
                params.antibody_ag_init_range_range.clone().start(),
                params.antibody_ag_init_range_range.clone().end(),
            ),

            rng_multiplier_ranges: range_to_uniform( params.antibody_rand_init_multiplier_range.clone()),
            rng_offset_ranges: range_to_uniform(params.antibody_rand_init_offset_range.clone()),
            rng_radius_range: Uniform::new_inclusive(
                params.antibody_rand_init_range_range.clone().start(),
                params.antibody_rand_init_range_range.clone().end(),
            ),

            class_labels,
        };
    }

    fn gen_random_genome(&self, label: Option<usize>) -> Antibody {

        let mut dim_multipliers: Vec<AntibodyDim> = Vec::with_capacity(self.n_dims);

        let dim_multipliers: Vec<AntibodyDim> = (0..self.n_dims).par_bridge().map(|i| {
            let mut rng = rand::thread_rng();
            let offset = self.rng_offset_ranges.get(i).unwrap().sample(&mut rng);
            let mut multiplier = self.rng_multiplier_ranges.get(i).unwrap().sample(&mut rng);

            let value_type = self.params.roll_dim_type_rand_ab();

            if value_type == DimValueType::Open {
                if rng.gen::<bool>() {
                    multiplier *= -1.0;
                }
            }

            return AntibodyDim {
                multiplier,
                offset,
                value_type,
            }
        }).collect();

        // for i in 0..self.n_dims {
        //     let offset = self.rng_offset_ranges.get(i).unwrap().sample(&mut rng);
        //     let mut multiplier = self.rng_multiplier_ranges.get(i).unwrap().sample(&mut rng);
        //
        //     let value_type = self.params.roll_dim_type_rand_ab();
        //
        //     if value_type == DimValueType::Open {
        //         num_open += 1;
        //         if rng.gen::<bool>() {
        //             multiplier *= -1.0;
        //         }
        //     }
        //
        //     dim_multipliers.push(AntibodyDim {
        //         multiplier,
        //         offset,
        //         value_type,
        //     })
        // }
        //

        let mut rng = rand::thread_rng();
        let mut radius_constant = self.rng_radius_range.sample(&mut rng);

        /*     for _ in 0..num_open {
                    radius_constant = radius_constant.sqrt();
                }
        */
        let class_label = if let Some(lbl) = label {
            lbl
        } else {
            self.class_labels
                .get(rng.gen_range(0..self.class_labels.len()))
                .unwrap()
                .clone()
        };

        return Antibody {
            dim_values: dim_multipliers,
            radius_constant,
            class_label,
            mutation_counter: HashMap::new(),
            clone_count: 0,
            final_train_label_membership: None,
            boosting_model_alpha: 1.0,
            final_train_label_affinity: None,
        };
    }
    pub fn generate_random_genome_with_label(&self, label: usize) -> Antibody {
        return self.gen_random_genome(Some(label));
    }
    pub fn generate_random_genome(&self) -> Antibody {
        return self.gen_random_genome(None);
    }

    pub fn generate_from_antigen(&self, antigen: &AntiGen) -> Antibody {

        let mut dim_multipliers: Vec<AntibodyDim> = Vec::with_capacity(self.n_dims);


        let dim_multipliers: Vec<AntibodyDim> = (0..self.n_dims).par_bridge().map(|i| {
            let mut rng = rand::thread_rng();

            let offset = antigen.values.get(i).unwrap().clone();
            let mut multiplier = self
                .antibody_multiplier_ranges
                .get(i)
                .unwrap()
                .sample(&mut rng);
            let value_type = self.params.roll_dim_type_from_ag_ab();


            if value_type == DimValueType::Open {
                if rng.gen::<bool>() {
                    multiplier *= -1.0;
                }
            }

            return AntibodyDim {
                multiplier,
                offset,
                value_type,
            }
        }).collect();

        let mut rng = rand::thread_rng();
        // for i in 0..self.n_dims {
        //     let offset = antigen.values.get(i).unwrap().clone();
        //     let mut multiplier = self
        //         .antibody_multiplier_ranges
        //         .get(i)
        //         .unwrap()
        //         .sample(&mut rng);
        //     let value_type = self.params.roll_dim_type_from_ag_ab();
        //
        //
        //     if value_type == DimValueType::Open {
        //         num_open += 1;
        //         if rng.gen::<bool>() {
        //             multiplier *= -1.0;
        //         }
        //     }
        //
        //
        //
        //
        //     lengths[i] = multiplier;
        //     offsets[i] = offset;
        //
        //
        //     dim_multipliers.push(AntibodyDim {
        //         multiplier,
        //         offset,
        //         value_type,
        //     })
        // }

        let mut radius_constant = self.antibody_radius_range.sample(&mut rng);
        let class_label = antigen.class_label;

        // for _ in 0..num_open {
        //     radius_constant = radius_constant.sqrt();
        // }
        return Antibody {
            dim_values: dim_multipliers,
            radius_constant,
            class_label,
            mutation_counter: HashMap::new(),
            clone_count: 0,
            final_train_label_membership: None,
            boosting_model_alpha: 1.0,
            final_train_label_affinity: None,
        };
    }
}
