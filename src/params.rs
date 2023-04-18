use std::ops::RangeInclusive;

use rand::prelude::SliceRandom;

use crate::representation::antibody::DimValueType;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum MutationType {
    Offset,
    Multiplier,
    MultiplierLocalSearch,
    ValueType,
    Radius,
    Label,
}

pub struct Params {
    // -- train params -- //
    pub antigen_pop_fraction: f64,
    pub leak_fraction: f64,
    pub leak_rand_prob: f64,
    pub generations: usize,

    pub mutation_offset_weight: usize,
    pub mutation_multiplier_weight: usize,
    pub mutation_multiplier_local_search_weight: usize,
    pub mutation_radius_weight: usize,
    pub mutation_value_type_weight: usize,
    pub mutation_label_weight: usize,

    pub mutation_value_type_local_search_dim: bool,

    pub offset_mutation_multiplier_range: RangeInclusive<f64>,
    pub multiplier_mutation_multiplier_range: RangeInclusive<f64>,
    pub radius_mutation_multiplier_range: RangeInclusive<f64>,
    pub value_type_valid_mutations: Vec<DimValueType>,
    pub label_valid_mutations: Vec<usize>,

    // selection
    pub max_replacment_frac: f64,
    pub tournament_size: usize,

    pub n_parents_mutations: usize,

    pub antibody_init_expand_radius: bool,

    // -- B-cell from antigen initialization -- //
    pub antibody_ag_init_multiplier_range: RangeInclusive<f64>,
    pub antibody_ag_init_value_types: Vec<DimValueType>,
    pub antibody_ag_init_range_range: RangeInclusive<f64>,

    // -- B-cell from random initialization -- //
    pub antibody_rand_init_offset_range: RangeInclusive<f64>,
    pub antibody_rand_init_multiplier_range: RangeInclusive<f64>,
    pub antibody_rand_init_value_types: Vec<DimValueType>,
    pub antibody_rand_init_range_range: RangeInclusive<f64>,
}

impl Params {
    pub fn roll_mutation_type(&self) -> MutationType {
        let weighted = vec![
            (MutationType::Offset, self.mutation_offset_weight),
            (MutationType::Multiplier, self.mutation_multiplier_weight),
            (
                MutationType::MultiplierLocalSearch,
                self.mutation_multiplier_local_search_weight,
            ),
            (MutationType::ValueType, self.mutation_value_type_weight),
            (MutationType::Radius, self.mutation_radius_weight),
            (MutationType::Label, self.mutation_label_weight),
        ];

        let mut rng = rand::thread_rng();
        return weighted
            .choose_weighted(&mut rng, |v| v.1)
            .unwrap()
            .0
            .clone();
    }
}

pub struct VerbosityParams {
    pub show_initial_pop_info: bool,
    pub iter_info_interval: Option<usize>,
    pub full_pop_acc_interval: Option<usize>,
    pub show_class_info: bool,
    pub make_plots: bool,

    pub display_final_ab_info: bool,
    pub display_detailed_error_info: bool,
    pub display_final_acc_info: bool,
}

impl VerbosityParams {
    pub fn n_fold_defaults() -> VerbosityParams {
        return VerbosityParams {
            show_initial_pop_info: false,
            iter_info_interval: None,
            full_pop_acc_interval: None,
            show_class_info: false,
            make_plots: false,
            display_final_ab_info: false,
            display_detailed_error_info: false,
            display_final_acc_info: false,
        };
    }
}
