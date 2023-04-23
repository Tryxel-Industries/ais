use crate::evaluation::Evaluation;
use crate::representation::{antibody::Antibody, antibody_factory::AntibodyFactory, antigen::AntiGen};
use crate::ais::{ArtificialImmuneSystem};
use crate::dataset_readers::{read_iris};
use crate::mutations::{mutate_length_matrix, mutate_offset_vector, get_rand_range, mutate_orientation_matrix};
use super::*;


#[test]
fn test_mutation_offset() {
    let (params, verbosity_params) = get_params();
    let abf = gen_dummy_ab_factory(3);
    let mut dummy_ab = abf.generate_random_genome();
    let pre_genome = dummy_ab.clone();
    let fitness_scaler = 0.5;
    mutate_offset_vector(&params, &mut dummy_ab, fitness_scaler);
    let m = pre_genome.offset
    .iter()
    .zip(dummy_ab.offset.iter())
    .filter(|a| a.0 != a.1).collect::<Vec<(&f64, &f64)>>();
    assert!(m.len() == 1);
}

#[test]
fn test_mutation_length() {
    let (params, verbosity_params) = get_params();
    let abf = gen_dummy_ab_factory(6);
    let mut dummy_ab = abf.generate_random_genome();
    let pre_genome = dummy_ab.clone();
    let fitness_scaler = 0.5;
    mutate_length_matrix(&params, &mut dummy_ab, fitness_scaler);
    let y = &pre_genome.length_matrix - &dummy_ab.length_matrix;
    let m: Vec<_> = y
    .column_iter()
    // .inspect(|f| println!("{}", &f))
    .filter(|a| a.sum() != 0.0).collect();
    assert!(m.len() == 1);
}

#[test]
fn test_mutation_orientation() {
    let (params, verbosity_params) = get_params();
    let abf = gen_dummy_ab_factory(768);
    let mut dummy_ab = abf.generate_random_genome();
    let pre_genome = dummy_ab.clone();
    let fitness_scaler = 0.5;
    mutate_orientation_matrix(&params, &mut dummy_ab);
    let y = &pre_genome.orientation_matrix - &dummy_ab.orientation_matrix;
    let m: Vec<_> = y
    .column_iter()
    // .inspect(|f| println!("{}", &f))
    .filter(|a| a.sum() != 0.0).collect();
    assert!(m.len() == 2);
}


fn get_params() -> (Params, VerbosityParams) {
    let mut params = Params {
        // -- train params -- //
        boost: 0,
        antigen_pop_fraction: 1.0,
        generations: 500,

        mutation_offset_weight: 5,
        mutation_multiplier_weight: 5,
        mutation_multiplier_local_search_weight: 3,
        mutation_radius_weight: 5,
        mutation_value_type_weight: 3,

        mutation_label_weight: 0,

        mutation_value_type_local_search_dim: true,

        // offset_mutation_multiplier_range: 0.8..=1.2,
        // multiplier_mutation_multiplier_range: 0.8..=1.2,
        // radius_mutation_multiplier_range: 0.8..=1.2,
        offset_mutation_multiplier_range: 0.5..=1.5,
        multiplier_mutation_multiplier_range: 0.5..=1.5,
        radius_mutation_multiplier_range: 0.5..=1.5,
        // value_type_valid_mutations: vec![DimValueType::Disabled,DimValueType::Circle],
        value_type_valid_mutations: vec![
            DimValueType::Circle,
            DimValueType::Disabled,
            DimValueType::Open,
        ],
        // value_type_valid_mutations: vec![DimValueType::Circle],
        label_valid_mutations: vec![1, 2],

        //selection
        membership_required: 0.0,
        leak_fraction: 0.5,
        leak_rand_prob: 0.5,
        replace_frac_type: ReplaceFractionType::MaxRepFrac(0.6),
        tournament_size: 1,
        n_parents_mutations: 40,

        antibody_init_expand_radius: true,

        // -- B-cell from antigen initialization -- //
        antibody_ag_init_multiplier_range: 0.8..=1.2,
        // antibody_ag_init_value_types: vec![DimValueType::Circle],
        // antibody_ag_init_value_types: vec![DimValueType::Disabled ,DimValueType::Circle],
        antibody_ag_init_value_types: vec![
            DimValueType::Circle,
            DimValueType::Disabled,
            DimValueType::Open,
        ],
        antibody_ag_init_range_range: 0.1..=0.4,

        // -- B-cell from random initialization -- //
        antibody_rand_init_offset_range: 0.0..=1.0,
        antibody_rand_init_multiplier_range: 0.8..=1.2,
        // antibody_rand_init_value_types: vec![DimValueType::Circle, DimValueType::Disabled],
        antibody_rand_init_value_types: vec![
            DimValueType::Circle,
            DimValueType::Disabled,
            // DimValueType::Open,
        ],
        antibody_rand_init_range_range: 0.1..=0.4,
    };

    let frac_verbosity_params = VerbosityParams {
        show_initial_pop_info: false,
        iter_info_interval: Some(1),
        full_pop_acc_interval: None,
        // full_pop_acc_interval: None,
        show_class_info: false,
        make_plots: true,
        display_final_ab_info: true,
        display_detailed_error_info: true,
        display_final_acc_info: true,
    };
    (params, frac_verbosity_params)
}

fn gen_dummy_ab_factory(dims: usize) -> AntibodyFactory {
    let (params, verbosity_params) = get_params();
    let t = AntibodyFactory::new(
        dims,
        params.antibody_ag_init_multiplier_range.clone(),
        params.antibody_ag_init_range_range.clone(),
        params.antibody_ag_init_value_types.clone(),
        params.antibody_rand_init_multiplier_range.clone(),
        params.antibody_rand_init_offset_range.clone(),
        params.antibody_rand_init_range_range.clone(),
        params.antibody_rand_init_value_types.clone(),
        vec![1, 2],
    );
    t
}
fn init_pop_for_testing(n_dims: usize) -> (Vec<f64>, Vec<f64>, Vec<(f64, Evaluation, Antibody)>) {

    let (params, verbosity_params) = get_params();

    let mut ais = ArtificialImmuneSystem::new();
    let train = get_antigens();
    
    ais.train(&train, &params, &verbosity_params)
}
fn get_antigens() -> Vec<AntiGen> {
    read_iris()
}