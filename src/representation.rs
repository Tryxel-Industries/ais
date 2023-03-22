use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::ops::RangeInclusive;

#[derive(Clone, PartialEq, Debug)]
pub enum DimValueType {
    Disabled,
    Open,
    Circle,
}

pub struct BCellFactory {
    n_dims: usize,

    b_cell_multiplier_ranges: Vec<Uniform<f64>>,
    b_cell_radius_range: Uniform<f64>,
    b_cell_allowed_value_types: Vec<DimValueType>,

    rng_multiplier_ranges: Vec<Uniform<f64>>,
    rng_offset_ranges: Vec<Uniform<f64>>,
    rng_radius_range: Uniform<f64>,
    rng_allowed_value_types: Vec<DimValueType>,

    class_labels: Vec<usize>,
}

impl BCellFactory {
    pub fn new(
        n_dims: usize,

        b_cell_multiplier_ranges: RangeInclusive<f64>,
        b_cell_radius_range: RangeInclusive<f64>,
        b_cell_allowed_value_types: Vec<DimValueType>,

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
            b_cell_multiplier_ranges: range_to_uniform(b_cell_multiplier_ranges),
            b_cell_radius_range: Uniform::new_inclusive(
                b_cell_radius_range.start(),
                b_cell_radius_range.end(),
            ),
            b_cell_allowed_value_types,

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

    fn gen_random_genome(&self, label: Option<usize>) -> BCell {
        let mut rng = rand::thread_rng();

        let mut dim_multipliers: Vec<BCellDim> = Vec::with_capacity(self.n_dims);

        let mut num_open = 0;
        for i in 0..self.n_dims {
            let offset = self.rng_offset_ranges.get(i).unwrap().sample(&mut rng) * -1.0;
            let multiplier = self.rng_multiplier_ranges.get(i).unwrap().sample(&mut rng);

            let value_type = self
                .rng_allowed_value_types
                .get(rng.gen_range(0..self.rng_allowed_value_types.len()))
                .unwrap()
                .clone();

            if value_type == DimValueType::Open {
                num_open += 1
            }

            dim_multipliers.push(BCellDim {
                multiplier,
                offset,
                value_type,
            })
        }

        let mut radius_constant = self.rng_radius_range.sample(&mut rng);

        for _ in 0..num_open {
            radius_constant = radius_constant.sqrt();
        }

        let class_label = if let Some(lbl) = label {
            lbl
        } else {
            self.class_labels
                .get(rng.gen_range(0..self.class_labels.len()))
                .unwrap()
                .clone()
        };

        return BCell {
            dim_values: dim_multipliers,
            radius_constant,
            class_label,
        };
    }
    pub fn generate_random_genome_with_label(&self, label: usize) -> BCell {
        return self.gen_random_genome(Some(label));
    }
    pub fn generate_random_genome(&self) -> BCell {
        return self.gen_random_genome(None);
    }

    pub fn generate_from_antigen(&self, antigen: &AntiGen) -> BCell {
        let mut rng = rand::thread_rng();

        let mut dim_multipliers: Vec<BCellDim> = Vec::with_capacity(self.n_dims);
        let mut num_open = 0;
        for i in 0..self.n_dims {
            let offset = antigen.values.get(i).unwrap().clone() * -1.0;
            let multiplier = self
                .b_cell_multiplier_ranges
                .get(i)
                .unwrap()
                .sample(&mut rng);
            let value_type = self
                .b_cell_allowed_value_types
                .get(rng.gen_range(0..self.b_cell_allowed_value_types.len()))
                .unwrap()
                .clone();

            if value_type == DimValueType::Open{
                num_open += 1
            }

            dim_multipliers.push(BCellDim {
                multiplier,
                offset,
                value_type,
            })
        }

        let mut  radius_constant = self.b_cell_radius_range.sample(&mut rng);
        let class_label = antigen.class_label;

        for _ in 0..num_open {
            radius_constant = radius_constant.sqrt();
        }
        return BCell {
            dim_values: dim_multipliers,
            radius_constant,
            class_label,
        };
    }
}

#[derive(Clone, Debug)]
pub struct BCellDim {
    // the multiplier for the dim value
    pub multiplier: f64,
    // the shift used for the dim if the value type is a circle
    pub offset: f64,
    // the exponent
    pub value_type: DimValueType,
}

#[derive(Clone, Debug)]
pub struct BCell {
    pub dim_values: Vec<BCellDim>,
    pub radius_constant: f64,
    pub class_label: usize,
}

impl BCell {
    //
    // initializers
    //

    pub fn test_antigen(&self, antigen: &AntiGen) -> bool {
        let mut roll_sum: f64 = 0.0;
        for i in 0..antigen.values.len() {
            let b_dim = self.dim_values.get(i).unwrap();
            let antigen_dim_val = antigen.values.get(i).unwrap();
            roll_sum += match b_dim.value_type {
                DimValueType::Disabled => 0.0,
                DimValueType::Open => b_dim.multiplier * (antigen_dim_val + b_dim.offset),
                DimValueType::Circle => {
                    (b_dim.multiplier * (antigen_dim_val + b_dim.offset)).powi(2)
                }
            };
        }

        // println!("roll_s {:?}, radius: {:?}", roll_sum, self.radius_constant);
        if roll_sum <= self.radius_constant {
            return true;
        } else {
            return false;
        }
    }
}

#[derive(Clone)]
pub struct AntiGen {
    pub id: usize,
    pub class_label: usize,
    pub values: Vec<f64>,
}

impl AntiGen {
    pub fn new(id: usize, class_label: usize, values: Vec<f64>) -> Self {
        Self {
            id,
            class_label,
            values,
        }
    }
}
