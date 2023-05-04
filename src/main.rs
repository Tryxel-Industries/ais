#![feature(fn_traits)]
#![feature(get_many_mut)]
#![feature(exclusive_range_pattern)]
#![allow(unused)]


use std::collections::{HashMap, HashSet};
use std::{env, f64};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::exit;
use std::time::Instant;
use arrayfire::{af_print, Array, col, Dim4, eval, pow, randu, seq, tile, view};
use json::value;

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
use crate::params::{modify_config_by_args, Params, PopSizeType, ReplaceFractionType, VerbosityParams};
use crate::plotting::plot_hist;
use crate::proto_test::{read_fnn_embeddings, read_kaggle_embeddings};
use crate::representation::antibody::{Antibody, DimValueType};
use crate::representation::antigen::{AntiGen, AntiGenPop};
use crate::representation::news_article_mapper::NewsArticleAntigenTranslator;
use crate::result_export::{dump_to_csv, read_ab_csv};
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
mod proto_test;

pub mod entities {
    include!(concat!(env!("OUT_DIR"), "/protobuf.entities.rs"));
}

fn ais_n_fold_test(
    params: Params,
    mut antigens: Vec<AntiGen>,
    verbosity_params: &VerbosityParams,
    n_folds: usize,
    translator: NewsArticleAntigenTranslator
) {
    // println!("antigens values    {:?}", antigens.iter().map(|v| &v.values).collect::<Vec<_>>());

    let folds = split_train_test_n_fold(&antigens, n_folds);

    let mut train_acc_vals = Vec::new();
    let mut test_acc_vals = Vec::new();
    for (n, (train, test)) in folds.iter().enumerate() {
        let (train_acc, test_acc) = ais_test(&antigens, train, test, verbosity_params, &params, &translator);
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
    translator: NewsArticleAntigenTranslator
) {
    // println!("antigens values    {:?}", antigens.iter().map(|v| &v.values).collect::<Vec<_>>());

    // let mut rng = rand::thread_rng();
    // antigens.shuffle(&mut rng);

    // println!("antigens values    {:?}", antigens.iter().map(|v| &v.values).collect::<Vec<_>>());


    let (train_slice, test) = split_train_test(&antigens, test_frac);


    let (train_acc, test_acc) = ais_test(&antigens, &train_slice, &test, verbosity_params, &params, &translator);

    println!("train_acc: {:?} test_acc: {:?} ", train_acc, test_acc)
}

fn ais_test(
    antigens: &Vec<AntiGen>,
    train_slice: &Vec<AntiGen>,
    test: &Vec<AntiGen>,
    verbosity_params: &VerbosityParams,
    params: &Params,
    translator: &NewsArticleAntigenTranslator
) -> (f64, f64) {

    let antigen_pop = AntiGenPop::new(antigens.clone());
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
        ais.train_immunobosting(&train, &params, verbosity_params, params.boost, &test, translator)
    }else {
        ais.train(&train, &params, verbosity_params)
    };



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

    let evaluated_pop = evaluate_population( &params, pop, &antigen_pop);



    let max_ag_id = antigens.iter().max_by_key(|ag| ag.id).unwrap().id;
    let mut match_counter = MatchCounter::new(antigens);

    match_counter.add_evaluations(
        evaluated_pop
            .iter()
            .map(|(evaluation, _)| evaluation)
            .collect::<Vec<_>>(),
    );


    let scored_pop = score_antibodies(evaluated_pop, &match_counter);

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
        let pred_class = ais.is_class_correct_with_membership(&antigen);
        // let pred_class = ais.is_class_correct(&antigen);

        // if pred_class.is_some() != pred_class_m.is_some(){
        //     println!("\n\nres value Diff registered, using valilla res was {:?}, using membership {:?}", pred_class, pred_class_m);
        // }
        // if pred_class.is_some() && pred_class_m.is_some(){
        //     if pred_class_m.unwrap() != pred_class.unwrap(){
        //         println!("Diff registered, using valilla res was {:?}, using membership {:?}", pred_class.unwrap(), pred_class_m.unwrap());
        //
        //     }
        // }
        //

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

    let mut membership_test_n_corr = 0;
    let mut membership_test_n_wrong = 0;


    let mut membership_test_per_class_corr = HashMap::new();
    let mut test_n_no_detect = 0;
    for antigen in test {
        let pred_class = ais.is_class_correct_with_membership(&antigen);

        if let Some(v) = pred_class {
            if v {
                membership_test_n_corr += 1;
                let class_count = membership_test_per_class_corr.get(&antigen.class_label).unwrap_or(&0);
                membership_test_per_class_corr.insert(antigen.class_label, *class_count + 1);
            } else {
                membership_test_n_wrong += 1
            }
        }
        let pred_class = ais.is_class_correct(&antigen);
        if let Some(v) = pred_class {
            if v {
                test_n_corr += 1;
            } else {
                test_n_wrong += 1
            }
        } else {
            test_n_no_detect += 1
        }
    }

    let translator_formatted = test.iter().chain(train.iter()).map(|ag| {
        let pred_class = ais.is_class_correct_with_membership(&ag);
         if let Some(v) = pred_class {
            if v {
                return (Some(true), ag)
            } else {
                return (Some(false), ag)
            }
        }else {
             return (None, ag)
         }
    }).collect();

    translator.get_show_ag_acc(translator_formatted);

    let test_acc = test_n_corr as f64 / (test.len() as f64);
    let test_precession = test_n_corr as f64 / (test_n_wrong as f64 + test_n_corr as f64).max(1.0);
    let membership_test_acc = membership_test_n_corr as f64 / (test.len() as f64);
    let membership_test_precession = membership_test_n_corr as f64 / (membership_test_n_wrong as f64 + membership_test_n_corr as f64).max(1.0);

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
            "without membership: corr {:>2?}, false {:>3?}, no_detect {:>3?}, presission: {:>2.3?}, frac: {:2.3?}",
            test_n_corr, test_n_wrong, test_n_no_detect,test_precession, test_acc
        );
        println!(
            "with membership:    corr {:>2?}, false {:>3?}, no_detect {:>3?}, presission: {:>2.3?}, frac: {:2.3?}",
            membership_test_n_corr, membership_test_n_wrong, test.len()-(membership_test_n_corr+membership_test_n_wrong), membership_test_precession, membership_test_acc
        );
        println!("per class cor {:?}", membership_test_per_class_corr);

        println!(
            "Total runtime: {:?}, \nPer iteration: {:?}",
            duration,
            duration.as_nanos() / params.generations as u128
        );
    }

    dump_to_csv(antigens, &ais.antibodies);


    return (train_acc, membership_test_acc);
    // ais.pred_class(test.get(0).unwrap());
}


fn ag_file_test(){
    let ab_file_path = "out/antibodies.csv";

    let mut translator = NewsArticleAntigenTranslator::new();
    let mut antigens = read_kaggle_embeddings(None, &mut translator, true);

    let antibodies = read_ab_csv(ab_file_path.parse().unwrap());

    let mut ais = ArtificialImmuneSystem::new();
    ais.antibodies = antibodies;



   let translator_formatted = antigens.iter().map(|ag| {
        let pred_class = ais.is_class_correct_with_membership(&ag);
         if let Some(v) = pred_class {
            if v {
                return (Some(true), ag)
            } else {
                return (Some(false), ag)
            }
        }else {
             return (None, ag)
         }
    }).collect();

    translator.get_show_ag_acc(translator_formatted);
}
fn ais_run() {
    // let mut antigens = read_iris();
    // let mut antigens = read_iris_snipped();
    // let mut antigens = read_wine();
    // let mut antigens = read_diabetes();
    // let mut antigens = read_spirals();

    // let mut antigens = read_pima_diabetes();
    // let mut antigens = read_sonar();
    // let mut antigens = read_glass();
    // let mut antigens = read_ionosphere();

    // let mut antigens = read_kaggle_semantic();
    // let _ = antigens.split_off(3000);


    let mut translator = NewsArticleAntigenTranslator::new();
    let mut antigens = read_kaggle_embeddings(Some(500), &mut translator, true);
    // let mut antigens = read_fnn_embeddings(Some(100), &mut translator, true);

    // antigens.iter().for_each(|x1| {
    //     if x1.class_label == 1{
    //         println!("features {:?}", x1.values);
    //     }
    // });
    // let _ = antigens.split_off(1000);

    // let mut rng = rand::thread_rng();
    // antigens.shuffle(&mut rng);

    let class_labels = antigens
        .iter()
        .map(|x| x.class_label)
        .collect::<HashSet<_>>();

    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();
    let mut params = Params {
        gpu_accelerate: false,
        boost: 0,
        // -- train params -- //
        // antigen_pop_size: PopSizeType::Fraction(1.0),
        antigen_pop_size: PopSizeType::Number(2000),
        generations: 50,

        mutation_offset_weight: 5,
        mutation_multiplier_weight: 5,
        mutation_multiplier_local_search_weight: 0,
        mutation_radius_weight: 5,
        mutation_value_type_weight: 3,

        mutation_label_weight: 0,

        mutation_value_type_local_search_dim: true,

        // -- reduction -- //
        membership_required: 0.80,


        // offset_mutation_multiplier_range: 0.8..=1.2,
        // multiplier_mutation_multiplier_range: 0.8..=1.2,
        // radius_mutation_multiplier_range: 0.8..=1.2,
        offset_mutation_multiplier_range: -0.5..=0.5,
        multiplier_mutation_multiplier_range: -0.5..=0.5,
        radius_mutation_multiplier_range: -0.5..=0.5,
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
        // replace_frac_type: ReplaceFractionType::Linear(0.8..0.3),
        replace_frac_type: ReplaceFractionType::MaxRepFrac(0.6),
        tournament_size: 1,
        n_parents_mutations: 40,

        antibody_init_expand_radius: true,

        // -- B-cell from antigen initialization -- //
        antibody_ag_init_multiplier_range: 0.8..=1.2,
        antibody_ag_init_value_types: vec![
            (DimValueType::Circle,1),
            (DimValueType::Disabled,1),
            (DimValueType::Open,1),
        ],
        antibody_ag_init_range_range: 0.1..=0.4,

        // -- B-cell from random initialization -- //
        antibody_rand_init_offset_range: 0.0..=1.0,
        antibody_rand_init_multiplier_range: 0.8..=1.2,
        antibody_rand_init_value_types: vec![
            (DimValueType::Circle,1),
            (DimValueType::Disabled,1),
            (DimValueType::Open,1),
        ],
        antibody_rand_init_range_range: 0.1..=0.4,
    };

    let frac_verbosity_params = VerbosityParams {
        show_initial_pop_info: true,
        iter_info_interval: None,
        full_pop_acc_interval: None,
        // iter_info_interval: Some(1),
        // full_pop_acc_interval: Some(10),
        show_class_info: false,
        make_plots: false,
        display_final_ab_info: false,
        display_detailed_error_info: false,
        display_final_acc_info: true,
    };
    modify_config_by_args(&mut params);

    ais_frac_test(params, antigens, &frac_verbosity_params, 0.2, translator);
    // ais_n_fold_test(params, antigens, &VerbosityParams::n_fold_defaults(), 10, translator)
}

fn gpu_test(){

    let ab_file_path = "out/antibodies.csv";


    let mut translator = NewsArticleAntigenTranslator::new();
    let mut antigens = read_kaggle_embeddings(None, &mut translator, true);
    // let mut antigens = read_iris_snipped();
    // let mut antigens = read_kaggle_embeddings(None, &mut translator, true);

    let antigen_pop = AntiGenPop::new(antigens.clone());

    let antibodies = read_ab_csv(ab_file_path.parse().unwrap());

    let antibody = antibodies.get(0).unwrap();

    // antibodies.get(3).unwrap().dim_values.iter().enumerate().for_each(|(n,v)|{
    //     println!("n: {:<5} offset: {:<5} value t: {:<5} multi: {:<5} ", n , v.offset, v.value_type, v.multiplier)
    // });


    let start = Instant::now();
    let mask = antigen_pop.get_registered_antigens(&antibody, &None);
    let registered_antigens_gpu: Vec<_> = antigens.iter()
        .zip(mask.iter())
        .filter(|(a,b)| **b)
        .map(|(a,b)|a).collect();
    let duration_gpu = start.elapsed();

    let start = Instant::now();
    // let cpu_mask: Vec<_> = antigens
    //     .iter()
    //     .map(|ag| antibody.test_antigen(ag)).collect();
    //
    let registered_antigens = antigens
        .iter()
        // .filter(|ag| idx_list.binary_search(&ag.id).is_ok())
        .filter(|ag| antibody.test_antigen(ag))
        .collect::<Vec<_>>();
    let duration_cpu = start.elapsed();

    println!("cpu time {:?}, gpu time {:?}", duration_cpu, duration_gpu);




    registered_antigens.iter().zip(registered_antigens_gpu.iter()).for_each(|(cpu,gpu)|{

        println!("cpu: {:?}  gpu: {:?}", cpu.id, gpu.id);
    });

    // print!("eq: {:?}",  mask.iter().zip(cpu_mask.iter()).all(|(a,b)|a==b));



    // println!("shape {:?}",exponents.len() );
    // println!("n_dims {:?}",num_dims );
    //
    // af_print!("offset arr ", offset_array);

    // /*let mut antigens = read_iris();
    //
    // let ag_pop = AntiGenPop::new(antigens);
    // af_print!("dataset values", ag_pop.ag_array);
    // */
    // let num_rows: u64 = 5;
    // let num_cols: u64 = 3;
    // let dims = Dim4::new(&[10, 5, 1, 1]);
    // let dims_mul = Dim4::new(&[1, 5, 1, 1]);
    //
    //
    // let a = Array::<i32>::new( (0..=50).collect::<Vec<i32>>().as_slice(),dims);
    // let multi = Array::<i32>::new( vec![2;5].as_slice(),dims_mul);
    // println!("device is {:?}", a.get_backend());
    // af_print!("initial", a);
    // af_print!("initial", view!(a[1:3:1, 1:1:0]));
    // // println!("seq: {:?}", seq!(1:3:1));
    // // af_print!("mul", multi);
    // // af_print!("mul * 2 ", multi.clone() + multi.clone());
    // //
    // let tiled = tile(&multi,Dim4::new(&[10,1, 1, 1]));
    // af_print!("tiled", tiled);
    // let pow_val = pow(&a,&tiled, false);
    // af_print!("add",pow_val);
    //
    // let col = col(&pow_val, 0);
    // let mut buffer= vec!(i32::default();col.elements());
    // col.host(&mut buffer);
    // println!("col {:?}",buffer );

}

fn main(){
    // ag_file_test()
    ais_run()
    // gpu_test()
}