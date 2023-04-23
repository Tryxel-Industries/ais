use std::{env, slice};
use std::ops::{Range, RangeInclusive};

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
    OffsetVector,
    LengthMatrix,
    OrientationMatrix,
}

pub enum ReplaceFractionType{
    Linear(Range<f64>),
    MaxRepFrac(f64)
}

pub struct Params {
    // -- train params -- //
    pub boost: usize,
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

    // reduction
    pub membership_required: f64,

    // selection
    pub replace_frac_type: ReplaceFractionType,
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
            (MutationType::OrientationMatrix, self.mutation_label_weight),
            (MutationType::LengthMatrix, self.mutation_label_weight),
            (MutationType::OffsetVector, self.mutation_label_weight),
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


fn width_to_range(width: f64) -> RangeInclusive<f64>{
    return (1.0-width)..=(1.0+width)
}

fn param_string_to_bool(param_string: String) -> bool{
    match param_string.as_str() {
        "True"| "t"| "true" => true,
        "False"| "f"| "false" => false,

        _ => {
            panic!("error")
        }
    }
}

fn filter_category_list<T: Clone + std::cmp::PartialEq>(use_param_option: Option<bool>, value_type: T, value_list: &mut Vec<T>){
    if let Some(use_param) = use_param_option {
        let has_param = value_list.contains(&value_type);
        if use_param{
            if !has_param {
                value_list.push(value_type.clone());
            }
        }else {
            if has_param {
                let content: Vec<_> = value_list.iter().filter(|x| **x != value_type).map(|x| x.clone()).collect();
                value_list.clear();
                value_list.extend(content);
            }
        }
    }
}

pub fn modify_config_by_args(params: &mut Params) {
    let args: Vec<String> = env::args().collect();

    let mut will_use_open = None;
    let mut will_use_disabled = None;


    for arg in args {
        if arg.starts_with("--") {
            let (key, value) = arg.strip_prefix("--").unwrap().split_once("=").unwrap();
            match key {
                "tournament_size" => params.tournament_size = value.parse().unwrap(),
                "leak_fraction" => params.leak_fraction = value.parse().unwrap(),
                "antigen_pop_fraction" => params.antigen_pop_fraction = value.parse().unwrap(),
                "max_replacment_frac" => params.replace_frac_type = ReplaceFractionType::MaxRepFrac(value.parse().unwrap()),

                "mutation_value_type_local_search_dim" => params.mutation_value_type_local_search_dim = param_string_to_bool(value.parse().unwrap()),
                "antibody_init_expand_radius" => params.antibody_init_expand_radius = param_string_to_bool(value.parse().unwrap()),


                "mutation_offset_weight" => params.mutation_offset_weight = value.parse().unwrap(),
                "mutation_multiplier_weight" => params.mutation_multiplier_weight = value.parse().unwrap(),
                "mutation_multiplier_local_search_weight" => params.mutation_multiplier_local_search_weight = value.parse().unwrap(),
                "mutation_radius_weight" => params.mutation_radius_weight = value.parse().unwrap(),
                "mutation_value_type_weight" => params.mutation_value_type_weight = value.parse().unwrap(),

                "use_open_dims" => will_use_open = Some(param_string_to_bool(value.parse().unwrap())),
                "use_disabled_dims" => will_use_disabled = Some(param_string_to_bool(value.parse().unwrap())),

                "offset_mutation_multiplier_range" => params.offset_mutation_multiplier_range = width_to_range(value.parse().unwrap()),
                "multiplier_mutation_multiplier_range" => params.multiplier_mutation_multiplier_range = width_to_range(value.parse().unwrap()),
                "radius_mutation_multiplier_range" => params.radius_mutation_multiplier_range = width_to_range(value.parse().unwrap()),


                _ => panic!("invalid config arg"),
            }
        }
    }
    for v in vec![&mut params.value_type_valid_mutations, &mut params.antibody_ag_init_value_types, &mut params.antibody_rand_init_value_types]{
        filter_category_list::<DimValueType>(will_use_open, DimValueType::Open, v);
        filter_category_list::<DimValueType>(will_use_disabled, DimValueType::Disabled, v);
    }



}
