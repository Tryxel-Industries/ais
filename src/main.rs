#![feature(fn_traits)]
#![feature(get_many_mut)]
#![feature(exclusive_range_pattern)]
#![allow(unused)]

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
use crate::experiment_logger::{ExperimentLogger, ExperimentProperty};
use crate::experiment_logger::ExperimentProperty::BoostAccuracyTest;
use crate::mutations::mutate;
use crate::params::{modify_config_by_args, Params, PopSizeType, ReplaceFractionType, VerbosityParams};
use crate::plotting::plot_hist;
use crate::prediction::EvaluationMethod;
use crate::representation::antibody::{Antibody, DimValueType};
use crate::representation::antigen::{AntiGen, AntiGenSplitShell};
use crate::representation::news_article_mapper::NewsArticleAntigenTranslator;
use crate::result_export::{dump_to_csv, read_ab_csv};
use crate::scoring::score_antibodies;
use crate::util::{split_train_test, split_train_test_n_fold};

mod ais;
mod bucket_empire;
mod evaluation;
pub mod mutations;
mod params;
mod plotting;
pub mod representation;
mod result_export;
mod scoring;
mod selection;
mod testing;
mod util;
mod datasets;
mod experiment_logger;
mod prediction;
mod display;
mod stupid_mutations;

pub mod entities {
    include!(concat!(env!("OUT_DIR"), "/protobuf.entities.rs"));
}

fn ais_n_fold_test(
    params: Params,
    mut shelled_antigens: Vec<AntiGenSplitShell>,
    verbosity_params: &VerbosityParams,
    n_folds: usize,
    translator: NewsArticleAntigenTranslator,
    logger: &mut ExperimentLogger,
) {
    // println!("antigens values    {:?}", antigens.iter().map(|v| &v.values).collect::<Vec<_>>());

    let folds = split_train_test_n_fold(&shelled_antigens, n_folds);
    let antigens = shelled_antigens.into_iter().flat_map(|ag| ag.upack()).collect();

    let mut fold_score_estimator = vec![1f64; folds.len()];


    let mut train_acc_vals = Vec::new();
    let mut test_acc_vals = Vec::new();
    for (n, (train, test)) in folds.iter().enumerate() {
        let (train_acc, test_acc) = ais_test(&antigens, train, test, verbosity_params, &params, &translator, logger);


        let estm = fold_score_estimator.get_mut(n).unwrap();
        std::mem::replace(estm, test_acc);
        train_acc_vals.push(train_acc);
        test_acc_vals.push(test_acc);
        println!(
            "on fold {:<2?} the test accuracy was: {:<5.4?} and the train accuracy was: {:.4}. Current best score estimate {:.4?}",
            n, test_acc, train_acc, fold_score_estimator.clone().mean()
        );
    }

    let train_mean: f64 = train_acc_vals.iter().mean();
    let train_std: f64 = train_acc_vals.iter().std_dev();

    let test_mean: f64 = test_acc_vals.iter().mean();
    let test_std: f64 = test_acc_vals.iter().std_dev();

    println!("train_acc: {:<5.4?} std {:.4}", train_mean, train_std);
    println!("test_acc: {:<5.4?} std {:.4}", test_mean, test_std);
    // println!("train size: {:?}, test size {:?}", train.len(), test.len());
}

fn ais_frac_test(
    params: Params,
    mut shelled_antigens: Vec<AntiGenSplitShell>,
    verbosity_params: &VerbosityParams,
    test_frac: f64,
    translator: NewsArticleAntigenTranslator,
    logger: &mut ExperimentLogger,
) {
    // println!("antigens values    {:?}", antigens.iter().map(|v| &v.values).collect::<Vec<_>>());

    // let mut rng = rand::thread_rng();
    // antigens.shuffle(&mut rng);

    // println!("antigens values    {:?}", antigens.iter().map(|v| &v.values).collect::<Vec<_>>());


    let (train_slice, test) = split_train_test(&shelled_antigens, test_frac);
    let antigens = shelled_antigens.into_iter().flat_map(|ag| ag.upack()).collect();

    let (train_acc, test_acc) = ais_test(&antigens, &train_slice, &test, verbosity_params, &params, &translator, logger);

    println!("train_acc: {:?} test_acc: {:?} ", train_acc, test_acc)
}

fn ais_test(
    antigens: &Vec<AntiGen>,
    train_slice: &Vec<AntiGen>,
    test: &Vec<AntiGen>,
    verbosity_params: &VerbosityParams,
    params: &Params,
    translator: &NewsArticleAntigenTranslator,
    logger: &mut ExperimentLogger,
) -> (f64, f64) {
    let class_labels = antigens
        .iter()
        .map(|x| x.class_label)
        .collect::<HashSet<_>>();

    if verbosity_params.show_initial_pop_info {
        println!(
            "train size: {:?} test size: {:?}",
            train_slice.len(),
            test.len()
        );
    }

    let start = Instant::now();
    let train = train_slice.clone().to_vec();

    let mut ais = ArtificialImmuneSystem::new();


    let (train_acc_hist, train_score_hist, init_scored_pop) =if params.boost > 0{
        ais.train_immunobosting(&train, &params, verbosity_params, params.boost, &test, translator, logger)
    }else {
        ais.train(&train, &params, verbosity_params, logger)
    };



    if verbosity_params.make_plots {
        plot_hist(train_acc_hist, "acuracy");
        plot_hist(train_score_hist, "score");
    }
    let duration = start.elapsed();


    let train_acc = eval_display(&train, &ais , &translator, "TRAIN".to_string(), true, Some(&params.eval_method));
    let test_acc = eval_display(&test, &ais, &translator, "TEST".to_string(), true, Some(&params.eval_method));
    // eval_display(&train, &ais, &params, &translator, "Train".to_string())


    dump_to_csv(antigens, &ais.antibodies);


    return (train_acc, test_acc);
    // ais.pred_class(test.get(0).unwrap());
}


fn trail_run_from_ab_csv(){
    let ab_file_path = "out/antibodies.csv";


    let dataset_used = Datasets::EmbeddingKaggle;
    // embedding params
    // let use_num_to_fetch = Some(1000);
    let use_num_to_fetch = None;
    let use_whitening = true;


    let mut translator = NewsArticleAntigenTranslator::new();
    let mut shelled_antigens =  get_dataset(dataset_used, use_num_to_fetch, None,&mut translator, use_whitening);
    let mut antigens: Vec<_> = shelled_antigens.into_iter().flat_map(|sag| sag.upack()).collect();

    let antibodies = read_ab_csv(ab_file_path.parse().unwrap());

    let mut ais = ArtificialImmuneSystem::new();
    ais.antibodies = antibodies;




    let train_acc = eval_display(&antigens, &ais, &translator, "Full SET".to_string(),true, None);




}
fn trail_training() {
    rayon::ThreadPoolBuilder::new().num_threads(29).build_global().unwrap();


    let dataset_used = Datasets::PrimaDiabetes;
    // embedding params
    let use_num_to_fetch = Some(200);
    let max_sentences_per_article = Some(20);
    let use_whitening = true;

    let mut logger = ExperimentLogger::new(
        dataset_used.clone(),
        vec![
            // ExperimentProperty::BoostAccuracy,
            // ExperimentProperty::BoostAccuracyTest
        ]
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


    let mut params =  if false{
        get_dataset_optimal_params(dataset_used, class_labels)
    } else {
        Params {
            eval_method: EvaluationMethod::Fraction,
            boost: 10,
            // -- train params -- //
            // antigen_pop_size: PopSizeType::Fraction(1.0),
            antigen_pop_size: PopSizeType::BoostingFixed(50),
            generations: 1000,

            mutation_offset_weight: 1,
            mutation_multiplier_weight: 1,
            mutation_multiplier_local_search_weight: 1,
            mutation_radius_weight: 0,
            mutation_value_type_weight: 1,

            mutation_label_weight: 0,

            mutation_value_type_local_search_dim: true,

            // -- reduction -- //
            membership_required: 0.75,

            offset_mutation_multiplier_range: -0.5..=0.5,
            multiplier_mutation_multiplier_range: -0.5..=0.5,
            radius_mutation_multiplier_range: -0.5..=0.5,

            value_type_valid_mutations: vec![
                DimValueType::Circle,
                DimValueType::Disabled,
                DimValueType::Open,
            ],

            label_valid_mutations: class_labels.clone().into_iter().collect::<Vec<usize>>(),

            correctness_weight: 1.0,
            coverage_weight: 1.0,
            uniqueness_weight: 0.5,
            good_afin_weight: 0.0,
            bad_afin_weight: 1.0,

            //selection
            leak_fraction: 0.5,
            leak_rand_prob: 0.5,
            // replace_frac_type: ReplaceFractionType::Linear(0.5..0.01),
            replace_frac_type: ReplaceFractionType::Linear(0.8..0.3),
            // replace_frac_type: ReplaceFractionType::Linear(0.6..0.5),
            // replace_frac_type: ReplaceFractionType::MaxRepFrac(0.8),
            tournament_size: 1,
            n_parents_mutations: 40,

            antibody_init_expand_radius: true,

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

    let frac_verbosity_params = VerbosityParams {
        show_initial_pop_info: false,
        iter_info_interval: None,
        full_pop_acc_interval: None,
        // iter_info_interval: Some(1),
        // full_pop_acc_interval: Some(50),
        show_class_info: false,
        make_plots: false,
        display_final_ab_info: true,
        display_detailed_error_info: true,
        display_final_acc_info: true,
        print_boost_info: true,
    };
    modify_config_by_args(&mut params);

    ais_frac_test(params, antigens, &frac_verbosity_params, 0.1, translator, &mut logger);
    // ais_n_fold_test(params, antigens, &VerbosityParams::n_fold_defaults(), 10, translator,&mut logger);

    logger.dump_to_json_file("./bip_bop.json".to_string())
}

fn main(){
    // trail_run_from_ab_csv();
    trail_training();
}