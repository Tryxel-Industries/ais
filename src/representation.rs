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
    // the multiplier for the dim value
    dim_multiplier_ranges: Vec<Uniform<f64>>,
    // the shift used for the dim if the value type is a circle
    dim_offset_ranges: Vec<Uniform<f64>>,

    radius_range: Uniform<f64>,

    allowed_value_types: Vec<DimValueType>,

    use_multiplier: bool,
    use_rand_radius: bool,

    class_labels: Vec<usize>,
}

impl BCellFactory {
    pub fn new(
        n_dims: usize,
        dim_multiplier_ranges: Vec<RangeInclusive<f64>>,
        dim_offset_ranges: Vec<RangeInclusive<f64>>,
        radius_range: RangeInclusive<f64>,
        allowed_value_types: Vec<DimValueType>,
        use_multiplier: bool,
        use_rand_radius: bool,
        class_labels: Vec<usize>,
    ) -> Self {
        let dim_multiplier_ranges_mapped: Vec<Uniform<f64>> = dim_multiplier_ranges
            .iter()
            .map(|x| Uniform::new_inclusive(x.start(), x.end()))
            .collect();
        let dim_offset_ranges_mapped: Vec<Uniform<f64>> = dim_offset_ranges
            .iter()
            .map(|x| Uniform::new_inclusive(x.start(), x.end()))
            .collect();
        return Self {
            n_dims,
            dim_multiplier_ranges: dim_multiplier_ranges_mapped,
            dim_offset_ranges: dim_offset_ranges_mapped,
            radius_range: Uniform::new_inclusive(radius_range.start(), radius_range.end()),
            allowed_value_types,
            use_multiplier,
            use_rand_radius,
            class_labels,
        };
    }

    pub fn generate_random_genome(&self) -> BCell {
        let mut rng = rand::thread_rng();

        let mut dim_multipliers: Vec<BCellDim> = Vec::with_capacity(self.n_dims);
        for i in 0..self.n_dims {
            let offset = self.dim_multiplier_ranges.get(i).unwrap().sample(&mut rng) * -1.0;

            let multiplier = if self.use_multiplier {
                self.dim_multiplier_ranges.get(i).unwrap().sample(&mut rng)
            } else {
                1.0
            };

            let value_type = self
                .allowed_value_types
                .get(rng.gen_range(0..self.allowed_value_types.len()))
                .unwrap()
                .clone();
            dim_multipliers.push(BCellDim {
                multiplier,
                offset,
                value_type,
            })
        }

        let radius_constant = self.radius_range.sample(&mut rng);
        let class_label = self
            .class_labels
            .get(rng.gen_range(0..self.allowed_value_types.len()))
            .unwrap()
            .clone();

        return BCell {
            dim_values: dim_multipliers,
            radius_constant,
            class_label,
        };
    }

    pub fn generate_from_antigen(&self, antigen: &AntiGen) -> BCell {
        let mut rng = rand::thread_rng();

        let mut dim_multipliers: Vec<BCellDim> = Vec::with_capacity(self.n_dims);
        for i in 0..self.n_dims {
            let offset = antigen.values.get(i).unwrap().clone() * -1.0;

            let multiplier = if self.use_multiplier {
                self.dim_multiplier_ranges.get(i).unwrap().sample(&mut rng)
            } else {
                1.0
            };

            let value_type = self
                .allowed_value_types
                .get(rng.gen_range(0..self.allowed_value_types.len()))
                .unwrap()
                .clone();
            dim_multipliers.push(BCellDim {
                multiplier,
                offset,
                value_type,
            })
        }

        let radius_constant = if self.use_rand_radius {
            self.radius_range.sample(&mut rng)
        } else {
            1.0
        };
        let class_label = antigen.class_label;

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
                DimValueType::Open => b_dim.multiplier * antigen_dim_val,
                DimValueType::Circle => {
                    ((b_dim.multiplier * antigen_dim_val) + b_dim.offset).powi(2)
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
