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

use crate::ais::{evaluate_population, ArtificialImmuneSystem};
use crate::bucket_empire::BucketKing;
use crate::dataset_readers::{
    read_diabetes, read_glass, read_ionosphere, read_iris, read_iris_snipped, read_kaggle_semantic,
    read_pima_diabetes, read_sonar, read_spirals, read_wine,
};
use crate::evaluation::MatchCounter;
use crate::mutations::mutate;
use crate::params::{Params, VerbosityParams};
use crate::plotting::plot_hist;
use crate::representation::antibody::{Antibody, DimValueType};
use crate::representation::antigen::AntiGen;
use crate::result_export::dump_to_csv;
use crate::scoring::score_antibodies;
use crate::util::{split_train_test, split_train_test_n_fold};

mod ais;
mod bucket_empire;
mod dataset_readers;
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

fn ais_n_fold_test(
    params: Params,
    mut antigens: Vec<AntiGen>,
    verbosity_params: &VerbosityParams,
    n_folds: usize,
) {
    // println!("antigens values    {:?}", antigens.iter().map(|v| &v.values).collect::<Vec<_>>());

    let folds = split_train_test_n_fold(&antigens, n_folds);

    let mut train_acc_vals = Vec::new();
    let mut test_acc_vals = Vec::new();
    for (n, (train, test)) in folds.iter().enumerate() {
        let (train_acc, test_acc) = ais_test(&antigens, train, test, verbosity_params, &params);
        train_acc_vals.push(train_acc);
        test_acc_vals.push(test_acc);
        println!(
            "on fold {:<2?} the test accuracy was: {:<5.4?} and the train accuracy was: {:.4}",
            n, test_acc, train_acc
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
    mut antigens: Vec<AntiGen>,
    verbosity_params: &VerbosityParams,
    test_frac: f64,
) {
    // println!("antigens values    {:?}", antigens.iter().map(|v| &v.values).collect::<Vec<_>>());

    // let mut rng = rand::thread_rng();
    // antigens.shuffle(&mut rng);

    // println!("antigens values    {:?}", antigens.iter().map(|v| &v.values).collect::<Vec<_>>());

    let mut rng = rand::thread_rng();
    antigens.shuffle(&mut rng);

    let (train_slice, test) = split_train_test(&antigens, test_frac);

    let (train_acc, test_acc) = ais_test(&antigens, &train_slice, &test, verbosity_params, &params);

    println!("train_acc: {:?} test_acc: {:?} ", train_acc, test_acc)
}

fn ais_test(
    antigens: &Vec<AntiGen>,
    train_slice: &Vec<AntiGen>,
    test: &Vec<AntiGen>,
    verbosity_params: &VerbosityParams,
    params: &Params,
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
    let (train_acc_hist, train_score_hist, init_scored_pop) =
        ais.train(&train, &params, verbosity_params);

    if verbosity_params.make_plots {
        plot_hist(train_acc_hist, "acuracy");
        plot_hist(train_score_hist, "score");
    }
    let duration = start.elapsed();

    let mut zero_reg_cells = 0;
    // display final

    let n_dims = antigens.get(0).unwrap().values.len();
    let mut bk: BucketKing<AntiGen> =
        BucketKing::new(n_dims, (0.0, 1.0), 10, |ag| ag.id, |ag| &ag.values);
    bk.add_values_to_index(&antigens);

    let pop = init_scored_pop
        .clone()
        .into_iter()
        .map(|(a, b, c)| c)
        .collect();
    let evaluated_pop = evaluate_population(&bk, &params, pop, &antigens);

    let max_ag_id = antigens.iter().max_by_key(|ag| ag.id).unwrap().id;
    let mut match_counter = MatchCounter::new(max_ag_id);

    match_counter.add_evaluations(
        evaluated_pop
            .iter()
            .map(|(evaluation, _)| evaluation)
            .collect::<Vec<_>>(),
    );

    let count_map: HashMap<usize, usize> = class_labels
        .clone()
        .iter()
        .map(|x| {
            (
                x.clone(),
                antigens
                    .iter()
                    .filter(|ag| ag.class_label == *x)
                    .collect::<Vec<&AntiGen>>()
                    .len(),
            )
        })
        .collect();
    let scored_pop = score_antibodies(evaluated_pop, &count_map, &match_counter);

    scored_pop.iter().for_each(|(disc_score, eval, antibody)| {
        let registered_antigens = test
            .iter()
            .filter(|ag| antibody.test_antigen(ag))
            .collect::<Vec<_>>();
        let with_same_label = registered_antigens
            .iter()
            .filter(|ag| ag.class_label == antibody.class_label)
            .collect::<Vec<_>>();
        let num_wrong = registered_antigens
            .iter()
            .filter(|ag| ag.class_label != antibody.class_label)
            // .inspect(|ag| println!("er ag id {:?}", ag.id))
            .collect::<Vec<_>>();

        let score = with_same_label.len() as f64 / (num_wrong.len() as f64 + 1.0);

        if verbosity_params.display_final_ab_info {
            //registered_antigens.len() > 0 {
            println!(
                "genome dim values    {:?}",
                antibody
                    .dim_values
                    .iter()
                    .map(|v| v.multiplier)
                    .collect::<Vec<_>>()
            );
            println!(
                "genome offset values {:?}",
                antibody
                    .dim_values
                    .iter()
                    .map(|v| v.offset)
                    .collect::<Vec<_>>()
            );
            println!(
                "genome value type    {:?}",
                antibody
                    .dim_values
                    .iter()
                    .map(|v| &v.value_type)
                    .collect::<Vec<_>>()
            );
            println!("genome matches    {:?}", eval.matched_ids);
            println!("genome errors     {:?}", eval.wrongly_matched);

            println!("genome value radius    {:?}", antibody.radius_constant);
            println!("genome mutation map    {:?}", antibody.mutation_counter);
            println!("genome clone count     {:?}", antibody.clone_count);

            println!(
                "num reg {:?} same label {:?} other label {:?}, score {:?}, discounted score {:?}",
                registered_antigens.len(),
                with_same_label.len(),
                num_wrong.len(),
                score,
                disc_score
            );
            println!()
        } else {
            zero_reg_cells += 1;
        }
    });

    if verbosity_params.display_detailed_error_info {
        println!("zero reg cells {}", zero_reg_cells);
        println!(
            "########## error mask \n{:?}",
            match_counter.incorrect_match_counter
        );
        println!(
            "########## match mask \n{:?}",
            match_counter.correct_match_counter
        );

        for n in (0..match_counter.correct_match_counter.len()) {
            let wrong = match_counter.incorrect_match_counter.get(n).unwrap();
            let right = match_counter.correct_match_counter.get(n).unwrap();

            if wrong > right {
                println!("idx: {:>4?}  cor: {:>3?} - wrong {:>3?}", n, right, wrong);
                // println!("ag dat: {:?}", antigens.iter().filter(|ag| ag.id == n).last().unwrap());
            }
        }
    }

    //train
    let mut n_corr = 0;
    let mut per_class_corr: HashMap<usize, usize> = HashMap::new();
    let mut n_wrong = 0;
    let mut n_no_detect = 0;
    for antigen in train_slice {
        let pred_class = ais.is_class_correct(&antigen);
        if let Some(v) = pred_class {
            if v {
                n_corr += 1;
                let class_count = per_class_corr.get(&antigen.class_label).unwrap_or(&0);
                per_class_corr.insert(antigen.class_label, *class_count + 1);
            } else {
                n_wrong += 1;
            }
        } else {
            n_no_detect += 1
        }
    }
    let train_acc = n_corr as f64 / (train_slice.len() as f64);

    //test

    let mut test_n_corr = 0;
    let mut test_n_wrong = 0;
    let mut test_per_class_corr = HashMap::new();
    let mut test_n_no_detect = 0;
    for antigen in test {
        let pred_class = ais.is_class_correct(&antigen);
        if let Some(v) = pred_class {
            if v {
                test_n_corr += 1;
                let class_count = test_per_class_corr.get(&antigen.class_label).unwrap_or(&0);
                test_per_class_corr.insert(antigen.class_label, *class_count + 1);
            } else {
                test_n_wrong += 1
            }
        } else {
            test_n_no_detect += 1
        }
    }

    let test_acc = test_n_corr as f64 / (test.len() as f64);

    if verbosity_params.display_final_acc_info {
        println!("=============================================================================");
        println!("      TRAIN");
        println!("=============================================================================");
        println!();
        println!("dataset size {:?}", train.len());
        println!(
            "corr {:?}, false {:?}, no_detect {:?}, frac: {:?}",
            n_corr, n_wrong, n_no_detect, train_acc,
        );
        println!("per class cor {:?}", per_class_corr);

        println!("=============================================================================");
        println!("      TEST");
        println!("=============================================================================");

        println!();
        println!("dataset size {:?}", test.len());
        println!(
            "corr {:?}, false {:?}, no_detect {:?}, frac: {:?}",
            test_n_corr, test_n_wrong, test_n_no_detect, test_acc
        );
        println!("per class cor {:?}", test_per_class_corr);

        println!(
            "Total runtime: {:?}, \nPer iteration: {:?}",
            duration,
            duration.as_nanos() / params.generations as u128
        );
    }

    dump_to_csv(antigens, &ais.antibodies);

    return (train_acc, test_acc);
    // ais.pred_class(test.get(0).unwrap());
}
fn modify_config_by_args(params: &mut Params) {
    let args: Vec<String> = env::args().collect();

    for arg in args {
        if arg.starts_with("--") {
            let (key, value) = arg.strip_prefix("--").unwrap().split_once("=").unwrap();
            match key {
                "tournament_size" => params.tournament_size = value.parse().unwrap(),
                "leak_fraction" => params.leak_fraction = value.parse().unwrap(),
                "antigen_pop_fraction" => params.antigen_pop_fraction = value.parse().unwrap(),
                _ => panic!("invalid config arg"),
            }
        }
    }
}

fn main() {
    // let mut antigens = read_iris();
    // let mut antigens = read_iris_snipped();
    // let mut antigens = read_wine();
    // let mut antigens = read_diabetes();
    let mut antigens = read_spirals();

    // let mut antigens = read_pima_diabetes();
    // let mut antigens = read_sonar();
    // let mut antigens = read_glass();
    // let mut antigens = read_ionosphere();

    // let mut antigens = read_kaggle_semantic();
    // let _ = antigens.split_off(3000);

    let mut rng = rand::thread_rng();
    antigens.shuffle(&mut rng);

    let class_labels = antigens
        .iter()
        .map(|x| x.class_label)
        .collect::<HashSet<_>>();

    let mut params = Params {
        // -- train params -- //
        antigen_pop_fraction: 1.0,
        generations: 1000,

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
        label_valid_mutations: class_labels.clone().into_iter().collect::<Vec<usize>>(),

        //selection
        leak_fraction: 0.5,
        leak_rand_prob: 0.5,
        max_replacment_frac: 0.6,
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
    modify_config_by_args(&mut params);

    // ais_frac_test(params, antigens, &frac_verbosity_params, 0.2);
    ais_n_fold_test(params, antigens, &VerbosityParams::n_fold_defaults(), 5)
}
