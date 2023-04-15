use std::collections::HashMap;
use std::ops::RangeInclusive;

use rand::distributions::{Distribution, Uniform};
use rand::Rng;

use crate::representation::antibody::{Antibody, AntibodyDim, DimValueType};
use crate::representation::antigen::AntiGen;

pub struct AntibodyFactory {
    n_dims: usize,

    antibody_multiplier_ranges: Vec<Uniform<f64>>,
    antibody_radius_range: Uniform<f64>,
    antibody_allowed_value_types: Vec<DimValueType>,

    rng_multiplier_ranges: Vec<Uniform<f64>>,
    rng_offset_ranges: Vec<Uniform<f64>>,
    rng_radius_range: Uniform<f64>,
    rng_allowed_value_types: Vec<DimValueType>,

    class_labels: Vec<usize>,
}

impl AntibodyFactory {
    pub fn new(
        n_dims: usize,

        antibody_multiplier_ranges: RangeInclusive<f64>,
        antibody_radius_range: RangeInclusive<f64>,
        antibody_allowed_value_types: Vec<DimValueType>,

        rng_multiplier_ranges: RangeInclusive<f64>,
        rng_offset_ranges: RangeInclusive<f64>,
        rng_radius_range: RangeInclusive<f64>,
        rng_allowed_value_types: Vec<DimValueType>,

        class_labels: Vec<usize>,
    ) -> Self {
        let range_to_uniform = |range: RangeInclusive<f64>| {
            return vec![range; n_dims]
                .iter()
                .map(|x| Uniform::new_inclusive(x.start(), x.end()))
                .collect::<Vec<Uniform<f64>>>();
        };

        return Self {
            n_dims,
            antibody_multiplier_ranges: range_to_uniform(antibody_multiplier_ranges),
            antibody_radius_range: Uniform::new_inclusive(
                antibody_radius_range.start(),
                antibody_radius_range.end(),
            ),
            antibody_allowed_value_types,

            rng_multiplier_ranges: range_to_uniform(rng_multiplier_ranges),
            rng_offset_ranges: range_to_uniform(rng_offset_ranges),
            rng_radius_range: Uniform::new_inclusive(
                rng_radius_range.start(),
                rng_radius_range.end(),
            ),
            rng_allowed_value_types,

            class_labels,
        };
    }

    fn gen_random_genome(&self, label: Option<usize>) -> Antibody {
        let mut rng = rand::thread_rng();

        let mut dim_multipliers: Vec<AntibodyDim> = Vec::with_capacity(self.n_dims);

        let mut num_open = 0;
        for i in 0..self.n_dims {
            let offset = self.rng_offset_ranges.get(i).unwrap().sample(&mut rng);
            let mut multiplier = self.rng_multiplier_ranges.get(i).unwrap().sample(&mut rng);

            let value_type = self
                .rng_allowed_value_types
                .get(rng.gen_range(0..self.rng_allowed_value_types.len()))
                .unwrap()
                .clone();

            if value_type == DimValueType::Open {
                num_open += 1;
                if rng.gen::<bool>() {
                    multiplier *= -1.0;
                }
            }

            dim_multipliers.push(AntibodyDim {
                multiplier,
                offset,
                value_type,
            })
        }

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
        };
    }
    pub fn generate_random_genome_with_label(&self, label: usize) -> Antibody {
        return self.gen_random_genome(Some(label));
    }
    pub fn generate_random_genome(&self) -> Antibody {
        return self.gen_random_genome(None);
    }

    pub fn generate_from_antigen(&self, antigen: &AntiGen) -> Antibody {
        let mut rng = rand::thread_rng();

        let mut dim_multipliers: Vec<AntibodyDim> = Vec::with_capacity(self.n_dims);
        let mut num_open = 0;
        for i in 0..self.n_dims {
            let offset = antigen.values.get(i).unwrap().clone();
            let mut multiplier = self
                .antibody_multiplier_ranges
                .get(i)
                .unwrap()
                .sample(&mut rng);
            let value_type = self
                .antibody_allowed_value_types
                .get(rng.gen_range(0..self.antibody_allowed_value_types.len()))
                .unwrap()
                .clone();

            if value_type == DimValueType::Open {
                num_open += 1;
                if rng.gen::<bool>() {
                    multiplier *= -1.0;
                }
            }

            dim_multipliers.push(AntibodyDim {
                multiplier,
                offset,
                value_type,
            })
        }

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
        };
    }
}
