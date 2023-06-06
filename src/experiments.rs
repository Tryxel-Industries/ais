extern crate core;

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::exit;
use std::time::Instant;

use plotters::prelude::*;
use rand::prelude::SliceRandom;
use rand::Rng;
use statrs::statistics::Statistics;

use crate::ais::{ ArtificialImmuneSystem};
use crate::bucket_empire::BucketKing;

use crate::datasets::{Datasets, get_dataset, get_dataset_optimal_params};
use crate::display::{eval_display, show_ab_dim_multipliers, show_ab_dim_offsets, show_ab_dim_value_types};
use crate::evaluation::MatchCounter;
use crate::experiment_logger::{ExperimentLogger, ExperimentProperty, LoggedValue};
use crate::experiment_logger::ExperimentProperty::{BoostAccuracyTest, FoldAccuracy};
use crate::params::{modify_config_by_args, Params, PopSizeType, ReplaceFractionType, VerbosityParams};
use crate::plotting::plot_hist;
use crate::prediction::EvaluationMethod;
use crate::representation::antibody::{Antibody, DimValueType};
use crate::representation::antigen::{AntiGen, AntiGenSplitShell};
use crate::representation::news_article_mapper::NewsArticleAntigenTranslator;
use crate::result_export::{dump_to_csv, read_ab_csv};
use crate::scoring::score_antibodies;
use crate::util::{split_train_test, split_train_test_n_fold};
use crate::ais_n_fold_test;
use crate::ais_frac_test;

fn ex_1_wine_crowding() {
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();


    let dataset_used = Datasets::Wine;
    // embedding params
    let use_num_to_fetch = Some(500);
    let max_sentences_per_article = Some(20);
    let use_whitening = true;

    let mut logger = ExperimentLogger::new(
        dataset_used.clone(),
        vec![
            //
            ExperimentProperty::TrainAccuracy,
            ExperimentProperty::TestAccuracy,
            ExperimentProperty::AvgTrainScore,
            ExperimentProperty::PopLabelMemberships,
            ExperimentProperty::PopDimTypeMemberships,

            ExperimentProperty::ScoreComponents,

            //
            // ExperimentProperty::Runtime,
            //
            // ExperimentProperty::BoostAccuracy,
            // ExperimentProperty::BoostAccuracyTest

            ExperimentProperty::FoldAccuracy,
        ],
        1
    );


    let mut translator = NewsArticleAntigenTranslator::new();
    let mut antigens =  get_dataset(dataset_used.clone(), use_num_to_fetch, max_sentences_per_article,&mut translator, use_whitening);

    println!("Dataset used {:?}", dataset_used.to_string());

    // let mut rng = rand::thread_rng();
    // antigens.shuffle(&mut rng);


    let class_labels = antigens
        .iter()
        .map(|x| x.class_label)
        .collect::<HashSet<_>>();

    let log_file_name = format!("./logg_dat/ex_1_logg_dat_{:?}.json", dataset_used);

    let mut params =  if false{
        get_dataset_optimal_params(dataset_used, class_labels)
    } else {
        Params {
            eval_method: EvaluationMethod::Fraction,
            boost: 0,
            // -- train params -- //
            antigen_pop_size: PopSizeType::Fraction(0.5),
            // antigen_pop_size: PopSizeType::BoostingFixed(400),
            generations: 300,

            mutation_offset_weight: 1,
            mutation_multiplier_weight: 1,
            mutation_multiplier_local_search_weight: 1,
            mutation_radius_weight: 1,
            mutation_value_type_weight: 1,

            mutation_label_weight: 0,

            mutation_value_type_local_search_dim: true,

            ratio_lock: true,
            crowding: true,



            // -- reduction -- //
            membership_required: 0.0,

            offset_mutation_multiplier_range: -0.5..=0.5,
            multiplier_mutation_multiplier_range: -0.5..=0.5,
            radius_mutation_multiplier_range: -0.5..=0.5,

            value_type_valid_mutations: vec![
                DimValueType::Circle,
                DimValueType::Disabled,
                DimValueType::Open,
            ],

            label_valid_mutations: class_labels.clone().into_iter().collect::<Vec<usize>>(),

            correctness_weight: 0.9,
            coverage_weight: 1.5,
            uniqueness_weight: -0.1,
            good_afin_weight: 0.0,
            bad_afin_weight: 2.0,

            //selection
            leak_fraction: 0.0,
            leak_rand_prob: 0.5,
            // replace_frac_type: ReplaceFractionType::Linear(0.5..0.01),
            replace_frac_type: ReplaceFractionType::Linear(0.8..0.3),
            // replace_frac_type: ReplaceFractionType::Linear(0.6..0.5),
            // replace_frac_type: ReplaceFractionType::MaxRepFrac(0.8),
            tournament_size: 1,
            n_parents_mutations: 40,

            antibody_init_expand_radius: false,

            // -- B-cell from antigen initialization -- //
            antibody_ag_init_multiplier_range: 0.8..=1.2,
            antibody_ag_init_value_types: vec![
                (DimValueType::Circle, 1),
                (DimValueType::Disabled, 1),
                (DimValueType::Open, 1),
            ],
            antibody_ag_init_range_range: 0.1..=0.4,

            // -- B-cell from random initialization -- //
            antibody_rand_init_offset_range: 0.0..=1.0,
            antibody_rand_init_multiplier_range: 0.8..=1.2,
            antibody_rand_init_value_types: vec![
                (DimValueType::Circle, 1),
                (DimValueType::Disabled, 1),
                (DimValueType::Open, 1),
            ],
            antibody_rand_init_range_range: 0.1..=0.4,
        }
    };
    logger.log_params(&params);


    let frac_verbosity_params = VerbosityParams {
        show_initial_pop_info: true,
        // iter_info_interval: None,
        // full_pop_acc_interval: None,
        iter_info_interval: Some(50),
        full_pop_acc_interval: Some(50),
        show_class_info: false,
        make_plots: true,
        display_final_ab_info: true,
        display_detailed_error_info: true,
        display_final_acc_info: true,
        print_boost_info: true,
    };
    modify_config_by_args(&mut params);

    if true{
        ais_frac_test(params, antigens, &frac_verbosity_params, 0.1, translator, &mut logger);
        // ais_n_fold_test(params, antigens, &VerbosityParams::n_fold_defaults(), 5, translator,&mut logger);
    }else {
        for n in 0..10{
            ais_n_fold_test(params.clone(), antigens.clone(), &VerbosityParams::n_fold_defaults(), 10, translator.clone() ,&mut logger);
        }
        logger.log_multi_run_acc()
    }

    logger.dump_to_json_file(log_file_name);
}

fn ex_1_diabetes_crowding() {
    //NB!: Remember to switch scoring to f1
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();


    let dataset_used = Datasets::Sonar;
    // embedding params
    let use_num_to_fetch = Some(500);
    let max_sentences_per_article = Some(20);
    let use_whitening = true;

    let mut logger = ExperimentLogger::new(
        dataset_used.clone(),
        vec![
            //
            ExperimentProperty::TrainAccuracy,
            ExperimentProperty::TestAccuracy,
            ExperimentProperty::AvgTrainScore,
            ExperimentProperty::PopLabelMemberships,
            ExperimentProperty::PopDimTypeMemberships,

            ExperimentProperty::ScoreComponents,

            //
            ExperimentProperty::Runtime,
            //
            // ExperimentProperty::BoostAccuracy,
            // ExperimentProperty::BoostAccuracyTest

            ExperimentProperty::FoldAccuracy,
        ],
        1
    );


    let mut translator = NewsArticleAntigenTranslator::new();
    let mut antigens =  get_dataset(dataset_used.clone(), use_num_to_fetch, max_sentences_per_article,&mut translator, use_whitening);

    println!("Dataset used {:?}", dataset_used.to_string());

    // let mut rng = rand::thread_rng();
    // antigens.shuffle(&mut rng);


    let class_labels = antigens
        .iter()
        .map(|x| x.class_label)
        .collect::<HashSet<_>>();

    let log_file_name = format!("./logg_dat/logg_dat_{:?}.json", dataset_used);

    let mut params =  if false{
        get_dataset_optimal_params(dataset_used, class_labels)
    } else {
        Params {
            eval_method: EvaluationMethod::Fraction,
            boost: 0,
            // -- train params -- //
            // antigen_pop_size: PopSizeType::Fraction(0.25),
            antigen_pop_size: PopSizeType::BoostingFixed(400),
            generations: 200,

            mutation_offset_weight: 1,
            mutation_multiplier_weight: 1,
            mutation_multiplier_local_search_weight: 1,
            mutation_radius_weight: 1,
            mutation_value_type_weight: 1,

            mutation_label_weight: 0,

            mutation_value_type_local_search_dim: true,

            ratio_lock: true,
            crowding: true,


            // -- reduction -- //
            membership_required: 0.0,

            offset_mutation_multiplier_range: -0.5..=0.5,
            multiplier_mutation_multiplier_range: -0.5..=0.5,
            radius_mutation_multiplier_range: -0.5..=0.5,

            value_type_valid_mutations: vec![
                DimValueType::Circle,
                DimValueType::Disabled,
                DimValueType::Open,
            ],

            label_valid_mutations: class_labels.clone().into_iter().collect::<Vec<usize>>(),

            correctness_weight: 0.9,
            coverage_weight: 1.5,
            uniqueness_weight: 0.0,
            good_afin_weight: 0.0,
            bad_afin_weight: 1.6,

            //selection
            leak_fraction: 0.0,
            leak_rand_prob: 0.5,
            // replace_frac_type: ReplaceFractionType::Linear(0.5..0.01),
            replace_frac_type: ReplaceFractionType::Linear(0.8..0.3),
            // replace_frac_type: ReplaceFractionType::Linear(0.6..0.5),
            // replace_frac_type: ReplaceFractionType::MaxRepFrac(0.8),
            tournament_size: 1,
            n_parents_mutations: 10,

            antibody_init_expand_radius: false,

            // -- B-cell from antigen initialization -- //
            antibody_ag_init_multiplier_range: 0.8..=1.2,
            antibody_ag_init_value_types: vec![
                (DimValueType::Circle, 1),
                (DimValueType::Disabled, 1),
                (DimValueType::Open, 1),
            ],
            antibody_ag_init_range_range: 0.1..=0.4,

            // -- B-cell from random initialization -- //
            antibody_rand_init_offset_range: 0.0..=1.0,
            antibody_rand_init_multiplier_range: 0.8..=1.2,
            antibody_rand_init_value_types: vec![
                (DimValueType::Circle, 1),
                (DimValueType::Disabled, 1),
                (DimValueType::Open, 1),
            ],
            antibody_rand_init_range_range: 0.1..=0.4,
        }
    };
    logger.log_params(&params);


    let frac_verbosity_params = VerbosityParams {
        show_initial_pop_info: true,
        // iter_info_interval: None,
        // full_pop_acc_interval: None,
        iter_info_interval: Some(50),
        full_pop_acc_interval: Some(50),
        show_class_info: false,
        make_plots: true,
        display_final_ab_info: true,
        display_detailed_error_info: true,
        display_final_acc_info: true,
        print_boost_info: true,
    };
    modify_config_by_args(&mut params);

    if true{
        ais_frac_test(params, antigens, &frac_verbosity_params, 0.1, translator, &mut logger);
        // ais_n_fold_test(params, antigens, &VerbosityParams::n_fold_defaults(), 5, translator,&mut logger);
    }else {
        for n in 0..10{
            ais_n_fold_test(params.clone(), antigens.clone(), &VerbosityParams::n_fold_defaults(), 10, translator.clone() ,&mut logger);
        }
        logger.log_multi_run_acc()
    }

    logger.dump_to_json_file(log_file_name);
}

fn ex_1_sonar_crowding() {
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();


    let dataset_used = Datasets::Sonar;
    // embedding params
    let use_num_to_fetch = Some(500);
    let max_sentences_per_article = Some(20);
    let use_whitening = true;

    let mut logger = ExperimentLogger::new(
        dataset_used.clone(),
        vec![
            //
            ExperimentProperty::TrainAccuracy,
            ExperimentProperty::TestAccuracy,
            ExperimentProperty::AvgTrainScore,
            ExperimentProperty::PopLabelMemberships,
            ExperimentProperty::PopDimTypeMemberships,

            ExperimentProperty::ScoreComponents,

            //
            ExperimentProperty::Runtime,
            //
            // ExperimentProperty::BoostAccuracy,
            // ExperimentProperty::BoostAccuracyTest

            ExperimentProperty::FoldAccuracy,
        ],
        1
    );


    let mut translator = NewsArticleAntigenTranslator::new();
    let mut antigens =  get_dataset(dataset_used.clone(), use_num_to_fetch, max_sentences_per_article,&mut translator, use_whitening);

    println!("Dataset used {:?}", dataset_used.to_string());

    let mut rng = rand::thread_rng();
    antigens.shuffle(&mut rng);


    let class_labels = antigens
        .iter()
        .map(|x| x.class_label)
        .collect::<HashSet<_>>();

    let log_file_name = format!("./logg_dat/logg_dat_{:?}.json", dataset_used);

    let mut params =  if false{
        get_dataset_optimal_params(dataset_used, class_labels)
    } else {
        Params {
            eval_method: EvaluationMethod::Fraction,
            boost: 0,
            // -- train params -- //
            antigen_pop_size: PopSizeType::Fraction(0.5),
            // antigen_pop_size: PopSizeType::BoostingFixed(400),
            generations: 300,

            mutation_offset_weight: 1,
            mutation_multiplier_weight: 1,
            mutation_multiplier_local_search_weight: 1,
            mutation_radius_weight: 1,
            mutation_value_type_weight: 1,

            mutation_label_weight: 0,

            mutation_value_type_local_search_dim: true,

            ratio_lock: true,
            crowding: true,


            // -- reduction -- //
            membership_required: 0.0,

            offset_mutation_multiplier_range: -0.5..=0.5,
            multiplier_mutation_multiplier_range: -0.5..=0.5,
            radius_mutation_multiplier_range: -0.5..=0.5,

            value_type_valid_mutations: vec![
                DimValueType::Circle,
                DimValueType::Disabled,
                DimValueType::Open,
            ],

            label_valid_mutations: class_labels.clone().into_iter().collect::<Vec<usize>>(),

            correctness_weight: 0.9,
            coverage_weight: 1.5,
            uniqueness_weight: -0.1,
            good_afin_weight: 0.0,
            bad_afin_weight: 1.8,

            //selection
            leak_fraction: 0.0,
            leak_rand_prob: 0.5,
            // replace_frac_type: ReplaceFractionType::Linear(0.5..0.01),
            replace_frac_type: ReplaceFractionType::Linear(0.8..0.3),
            // replace_frac_type: ReplaceFractionType::Linear(0.6..0.5),
            // replace_frac_type: ReplaceFractionType::MaxRepFrac(0.8),
            tournament_size: 1,
            n_parents_mutations: 10,

            antibody_init_expand_radius: false,

            // -- B-cell from antigen initialization -- //
            antibody_ag_init_multiplier_range: 0.8..=1.2,
            antibody_ag_init_value_types: vec![
                (DimValueType::Circle, 1),
                (DimValueType::Disabled, 1),
                (DimValueType::Open, 1),
            ],
            antibody_ag_init_range_range: 0.1..=0.4,

            // -- B-cell from random initialization -- //
            antibody_rand_init_offset_range: 0.0..=1.0,
            antibody_rand_init_multiplier_range: 0.8..=1.2,
            antibody_rand_init_value_types: vec![
                (DimValueType::Circle, 1),
                (DimValueType::Disabled, 1),
                (DimValueType::Open, 1),
            ],
            antibody_rand_init_range_range: 0.1..=0.4,
        }
    };
    logger.log_params(&params);


    let frac_verbosity_params = VerbosityParams {
        show_initial_pop_info: true,
        // iter_info_interval: None,
        // full_pop_acc_interval: None,
        iter_info_interval: Some(50),
        full_pop_acc_interval: Some(50),
        show_class_info: false,
        make_plots: true,
        display_final_ab_info: true,
        display_detailed_error_info: true,
        display_final_acc_info: true,
        print_boost_info: true,
    };
    modify_config_by_args(&mut params);

    if true{
        // ais_frac_test(params, antigens, &frac_verbosity_params, 0.1, translator, &mut logger);
        ais_n_fold_test(params, antigens, &VerbosityParams::n_fold_defaults(), 10, translator,&mut logger);
    }else {
        for n in 0..10{
            ais_n_fold_test(params.clone(), antigens.clone(), &VerbosityParams::n_fold_defaults(), 10, translator.clone() ,&mut logger);
        }
        logger.log_multi_run_acc()
    }

    logger.dump_to_json_file(log_file_name);
}

fn ex_1_ionosphere_crowding() {
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();


    let dataset_used = Datasets::Ionosphere;
    // embedding params
    let use_num_to_fetch = Some(500);
    let max_sentences_per_article = Some(20);
    let use_whitening = true;

    let mut logger = ExperimentLogger::new(
        dataset_used.clone(),
        vec![
            //
            ExperimentProperty::TrainAccuracy,
            ExperimentProperty::TestAccuracy,
            ExperimentProperty::AvgTrainScore,
            ExperimentProperty::PopLabelMemberships,
            ExperimentProperty::PopDimTypeMemberships,

            ExperimentProperty::ScoreComponents,

            //
            ExperimentProperty::Runtime,
            //
            // ExperimentProperty::BoostAccuracy,
            // ExperimentProperty::BoostAccuracyTest

            ExperimentProperty::FoldAccuracy,
        ],
        1
    );


    let mut translator = NewsArticleAntigenTranslator::new();
    let mut antigens =  get_dataset(dataset_used.clone(), use_num_to_fetch, max_sentences_per_article,&mut translator, use_whitening);

    println!("Dataset used {:?}", dataset_used.to_string());

    let mut rng = rand::thread_rng();
    antigens.shuffle(&mut rng);


    let class_labels = antigens
        .iter()
        .map(|x| x.class_label)
        .collect::<HashSet<_>>();

    let log_file_name = format!("./logg_dat/logg_dat_{:?}.json", dataset_used);

    let mut params =  if false{
        get_dataset_optimal_params(dataset_used, class_labels)
    } else {
        Params {
            eval_method: EvaluationMethod::Fraction,
            boost: 0,
            // -- train params -- //
            antigen_pop_size: PopSizeType::Fraction(0.5),
            // antigen_pop_size: PopSizeType::BoostingFixed(400),
            generations: 300,

            mutation_offset_weight: 1,
            mutation_multiplier_weight: 1,
            mutation_multiplier_local_search_weight: 1,
            mutation_radius_weight: 1,
            mutation_value_type_weight: 1,

            mutation_label_weight: 0,

            mutation_value_type_local_search_dim: true,

            ratio_lock: true,
            crowding: true,


            // -- reduction -- //
            membership_required: 0.0,

            offset_mutation_multiplier_range: -0.5..=0.5,
            multiplier_mutation_multiplier_range: -0.5..=0.5,
            radius_mutation_multiplier_range: -0.5..=0.5,

            value_type_valid_mutations: vec![
                DimValueType::Circle,
                DimValueType::Disabled,
                DimValueType::Open,
            ],

            label_valid_mutations: class_labels.clone().into_iter().collect::<Vec<usize>>(),

            correctness_weight: 0.9,
            coverage_weight: 1.5,
            uniqueness_weight: -0.1,
            good_afin_weight: 0.0,
            bad_afin_weight: 2.0,

            //selection
            leak_fraction: 0.0,
            leak_rand_prob: 0.5,
            // replace_frac_type: ReplaceFractionType::Linear(0.5..0.01),
            replace_frac_type: ReplaceFractionType::Linear(0.8..0.3),
            // replace_frac_type: ReplaceFractionType::Linear(0.6..0.5),
            // replace_frac_type: ReplaceFractionType::MaxRepFrac(0.8),
            tournament_size: 1,
            n_parents_mutations: 10,

            antibody_init_expand_radius: false,

            // -- B-cell from antigen initialization -- //
            antibody_ag_init_multiplier_range: 0.8..=1.2,
            antibody_ag_init_value_types: vec![
                (DimValueType::Circle, 1),
                (DimValueType::Disabled, 1),
                (DimValueType::Open, 1),
            ],
            antibody_ag_init_range_range: 0.1..=0.4,

            // -- B-cell from random initialization -- //
            antibody_rand_init_offset_range: 0.0..=1.0,
            antibody_rand_init_multiplier_range: 0.8..=1.2,
            antibody_rand_init_value_types: vec![
                (DimValueType::Circle, 1),
                (DimValueType::Disabled, 1),
                (DimValueType::Open, 1),
            ],
            antibody_rand_init_range_range: 0.1..=0.4,
        }
    };
    logger.log_params(&params);


    let frac_verbosity_params = VerbosityParams {
        show_initial_pop_info: true,
        // iter_info_interval: None,
        // full_pop_acc_interval: None,
        iter_info_interval: Some(50),
        full_pop_acc_interval: Some(50),
        show_class_info: false,
        make_plots: true,
        display_final_ab_info: true,
        display_detailed_error_info: true,
        display_final_acc_info: true,
        print_boost_info: true,
    };
    modify_config_by_args(&mut params);

    if true{
        // ais_frac_test(params, antigens, &frac_verbosity_params, 0.1, translator, &mut logger);
        ais_n_fold_test(params, antigens, &VerbosityParams::n_fold_defaults(), 10, translator,&mut logger);
    }else {
        for n in 0..10{
            ais_n_fold_test(params.clone(), antigens.clone(), &VerbosityParams::n_fold_defaults(), 10, translator.clone() ,&mut logger);
        }
        logger.log_multi_run_acc()
    }

    logger.dump_to_json_file(log_file_name);
}

fn ex_1_iris_crowding() {
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();


    let dataset_used = Datasets::Iris;
    // embedding params
    let use_num_to_fetch = Some(500);
    let max_sentences_per_article = Some(20);
    let use_whitening = true;

    let mut logger = ExperimentLogger::new(
        dataset_used.clone(),
        vec![
            //
            ExperimentProperty::TrainAccuracy,
            ExperimentProperty::TestAccuracy,
            ExperimentProperty::AvgTrainScore,
            ExperimentProperty::PopLabelMemberships,
            ExperimentProperty::PopDimTypeMemberships,

            ExperimentProperty::ScoreComponents,

            //
            ExperimentProperty::Runtime,
            //
            // ExperimentProperty::BoostAccuracy,
            // ExperimentProperty::BoostAccuracyTest

            ExperimentProperty::FoldAccuracy,
        ],
        1
    );


    let mut translator = NewsArticleAntigenTranslator::new();
    let mut antigens =  get_dataset(dataset_used.clone(), use_num_to_fetch, max_sentences_per_article,&mut translator, use_whitening);

    println!("Dataset used {:?}", dataset_used.to_string());

    let mut rng = rand::thread_rng();
    antigens.shuffle(&mut rng);


    let class_labels = antigens
        .iter()
        .map(|x| x.class_label)
        .collect::<HashSet<_>>();

    let log_file_name = format!("./logg_dat/logg_dat_{:?}.json", dataset_used);

    let mut params =  if false{
        get_dataset_optimal_params(dataset_used, class_labels)
    } else {
        Params {
            eval_method: EvaluationMethod::Fraction,
            boost: 0,
            // -- train params -- //
            antigen_pop_size: PopSizeType::Fraction(0.5),
            // antigen_pop_size: PopSizeType::BoostingFixed(400),
            generations: 300,

            mutation_offset_weight: 1,
            mutation_multiplier_weight: 1,
            mutation_multiplier_local_search_weight: 1,
            mutation_radius_weight: 1,
            mutation_value_type_weight: 1,

            mutation_label_weight: 0,

            mutation_value_type_local_search_dim: true,

            ratio_lock: true,
            crowding: false,


            // -- reduction -- //
            membership_required: 0.0,

            offset_mutation_multiplier_range: -0.5..=0.5,
            multiplier_mutation_multiplier_range: -0.5..=0.5,
            radius_mutation_multiplier_range: -0.5..=0.5,

            value_type_valid_mutations: vec![
                DimValueType::Circle,
                DimValueType::Disabled,
                DimValueType::Open,
            ],

            label_valid_mutations: class_labels.clone().into_iter().collect::<Vec<usize>>(),

            correctness_weight: 0.9,
            coverage_weight: 1.5,
            uniqueness_weight: -0.1,
            good_afin_weight: 0.0,
            bad_afin_weight: 1.8,

            //selection
            leak_fraction: 0.0,
            leak_rand_prob: 0.5,
            // replace_frac_type: ReplaceFractionType::Linear(0.5..0.01),
            replace_frac_type: ReplaceFractionType::Linear(0.8..0.3),
            // replace_frac_type: ReplaceFractionType::Linear(0.6..0.5),
            // replace_frac_type: ReplaceFractionType::MaxRepFrac(0.8),
            tournament_size: 1,
            n_parents_mutations: 10,

            antibody_init_expand_radius: false,

            // -- B-cell from antigen initialization -- //
            antibody_ag_init_multiplier_range: 0.8..=1.2,
            antibody_ag_init_value_types: vec![
                (DimValueType::Circle, 1),
                (DimValueType::Disabled, 1),
                (DimValueType::Open, 1),
            ],
            antibody_ag_init_range_range: 0.1..=0.4,

            // -- B-cell from random initialization -- //
            antibody_rand_init_offset_range: 0.0..=1.0,
            antibody_rand_init_multiplier_range: 0.8..=1.2,
            antibody_rand_init_value_types: vec![
                (DimValueType::Circle, 1),
                (DimValueType::Disabled, 1),
                (DimValueType::Open, 1),
            ],
            antibody_rand_init_range_range: 0.1..=0.4,
        }
    };
    logger.log_params(&params);


    let frac_verbosity_params = VerbosityParams {
        show_initial_pop_info: true,
        // iter_info_interval: None,
        // full_pop_acc_interval: None,
        iter_info_interval: Some(50),
        full_pop_acc_interval: Some(50),
        show_class_info: false,
        make_plots: true,
        display_final_ab_info: true,
        display_detailed_error_info: true,
        display_final_acc_info: true,
        print_boost_info: true,
    };
    modify_config_by_args(&mut params);

    if true{
        // ais_frac_test(params, antigens, &frac_verbosity_params, 0.1, translator, &mut logger);
        ais_n_fold_test(params, antigens, &VerbosityParams::n_fold_defaults(), 10, translator,&mut logger);
    }else {
        for n in 0..10{
            ais_n_fold_test(params.clone(), antigens.clone(), &VerbosityParams::n_fold_defaults(), 10, translator.clone() ,&mut logger);
        }
        logger.log_multi_run_acc()
    }

    logger.dump_to_json_file(log_file_name);
}
