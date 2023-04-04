#![feature(fn_traits)]
#![feature(get_many_mut)]
#![feature(exclusive_range_pattern)]

extern crate core;

use crate::ais::{evaluate_population, ArtificialImmuneSystem, Params};
use crate::bucket_empire::BucketKing;
use crate::dataset_readers::{
    read_diabetes, read_glass, read_ionosphere, read_iris, read_iris_snipped, read_pima_diabetes,
    read_sonar, read_spirals, read_wine,
};
use crate::evaluation::{gen_error_merge_mask, gen_merge_mask, score_b_cells};
use plotters::prelude::*;
use rand::prelude::SliceRandom;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ptr::null;
use std::time::Instant;

use crate::mutations::mutate;
use crate::representation::{AntiGen, BCell, DimValueType};

use crate::selection::selection;

use crate::ais::MutationType::ValueType;
use statrs::statistics::Median;
use statrs::statistics::Statistics;

mod ais;
mod bucket_empire;
mod dataset_readers;
mod enums;
mod evaluation;
mod model;
pub mod mutations;
pub mod representation;
mod selection;
mod simple_rr_ais;
#[derive(Clone)]
struct TestStruct {
    idx: i32,
    a: Vec<f64>,
}

fn bkt_test() {
    println!("Hello, world!");
    let dims = 500;
    let mut the_king: BucketKing<TestStruct> =
        BucketKing::new(dims, (-1.0, 1.0), 4, |x| x.idx as usize, |x1| &x1.a);

    let mut test_dat = Vec::new();

    let mut rng = rand::thread_rng();
    for i in 0..50000 {
        let vals: Vec<f64> = (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect();
        test_dat.push(TestStruct { a: vals, idx: i })
    }
    let vals: Vec<f64> = (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let check_val = TestStruct {
        a: vals,
        idx: 99999,
    };

    let _chk2 = check_val.clone();
    the_king.add_values_to_index(&test_dat);

    let res = the_king.get_potential_matches_indexes(&check_val).unwrap();
    println!("{:?}", res);

    let res2 = the_king
        .get_potential_matches_indexes(&test_dat.get(5).unwrap())
        .unwrap();
    println!("sanity {:?}", res2);

    // the_king.add_values_to_index(&vec![check_val]);
    // let res = the_king.get_potential_matches_indexes(&chk2).unwrap();
    // println!("{:?}",res);
}

fn plot_hist(hist: Vec<f64>, file_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("train_graph{:?}.png", file_name);
    let root = BitMapBackend::new(&path, (3000, 1000)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_y = hist.iter().max_by(|a, b| a.total_cmp(b)).unwrap().clone() as f32;
    let min_y = hist.iter().min_by(|a, b| a.total_cmp(b)).unwrap().clone() as f32;
    // let min_y = hist.get(hist.len()/2).unwrap().clone() as f32;
    let mut chart = ChartBuilder::on(&root)
        .caption("y=x^2", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0f32..(hist.len() as f32), min_y..max_y)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            hist.iter().enumerate().map(|(y, x)| (y as f32, *x as f32)),
            &RED,
        ))?
        .label("y = x^2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    return Ok(());
}

fn split_train_test(antigens: &Vec<AntiGen>, test_frac: f64) -> (Vec<AntiGen>, Vec<AntiGen>) {
    let classes: HashSet<usize> = antigens.iter().map(|ag| ag.class_label).collect();

    let mut train: Vec<AntiGen> = Vec::new();
    let mut test: Vec<AntiGen> = Vec::new();

    for class in classes {
        let mut of_class: Vec<_> = antigens
            .iter()
            .filter(|ag| ag.class_label == class)
            .cloned()
            .collect();
        // println!("num in class {:?} is {:?}", class ,of_class.len());
        let num_test = (of_class.len() as f64 * test_frac) as usize;
        let class_train = of_class.split_off(num_test);

        train.extend(class_train);
        test.extend(of_class);

        // println!("train s {:?} test s {:?}", train.len() ,test.len());
    }

    return (train, test);
}

fn split_train_test_n_fold(
    antigens: &Vec<AntiGen>,
    n_folds: usize,
) -> Vec<(Vec<AntiGen>, Vec<AntiGen>)> {
    let classes: HashSet<usize> = antigens.iter().map(|ag| ag.class_label).collect();
    let fold_frac = 1.0 / n_folds as f64;

    let mut folds: Vec<Vec<AntiGen>> = Vec::new();
    for _ in 0..n_folds {
        folds.push(Vec::new());
    }

    for class in classes {
        let mut of_class: Vec<_> = antigens
            .iter()
            .filter(|ag| ag.class_label == class)
            .cloned()
            .collect();
        let class_fold_size = (of_class.len() as f64 * fold_frac).floor() as usize;

        // println!("class {:?} has {:?} elements per fold", class, class_fold_size);
        for n in 0..(n_folds - 1) {
            let new_vals = of_class.drain(..class_fold_size);

            let mut fold_vec = folds.get_mut(n).unwrap();
            fold_vec.extend(new_vals)
        }

        folds.get_mut(n_folds - 1).unwrap().extend(of_class);
    }

    let mut ret_folds: Vec<(Vec<AntiGen>, Vec<AntiGen>)> = Vec::new();

    for fold in 0..n_folds {
        let test_fold = folds.get(fold).unwrap().clone();
        // println!("fold {:?} has {:?} elements", fold, test_fold.len());
        let mut train_fold = Vec::new();
        for join_fold in 0..n_folds {
            if join_fold != fold {
                train_fold.extend(folds.get(join_fold).unwrap().clone());
            }
        }
        ret_folds.push((train_fold, test_fold))
    }

    // println!("folds {:?}", folds);
    return ret_folds;
}

fn ais_n_fold_test(params: Params, mut antigens: Vec<AntiGen>) {
    // println!("antigens values    {:?}", antigens.iter().map(|v| &v.values).collect::<Vec<_>>());

    let folds = split_train_test_n_fold(&antigens, 5);

    let mut train_acc_vals = Vec::new();
    let mut test_acc_vals = Vec::new();
    for (train, test) in folds.iter() {
        let (train_acc, test_acc) = ais_test(&antigens, train, test, false, &params);
        train_acc_vals.push(train_acc);
        test_acc_vals.push(test_acc);
    }

    let train_mean: f64 = train_acc_vals.iter().mean();
    let train_std: f64 = train_acc_vals.iter().std_dev();

    let test_mean: f64 = test_acc_vals.iter().mean();
    let test_std: f64 = test_acc_vals.iter().std_dev();

    println!("train_acc: {:<5.4?} std {:.4}", train_mean, train_std);
    println!("test_acc: {:<5.4?} std {:.4}", test_mean, test_std);
    // println!("train size: {:?}, test size {:?}", train.len(), test.len());
}

fn ais_frac_test(params: Params, mut antigens: Vec<AntiGen>, verbose: bool) {
    // println!("antigens values    {:?}", antigens.iter().map(|v| &v.values).collect::<Vec<_>>());

    // let mut rng = rand::thread_rng();
    // antigens.shuffle(&mut rng);

    // println!("antigens values    {:?}", antigens.iter().map(|v| &v.values).collect::<Vec<_>>());

    let mut rng = rand::thread_rng();
    antigens.shuffle(&mut rng);

    let (train_slice, test) = split_train_test(&antigens, 0.2);

    let (train_acc, test_acc) = ais_test(&antigens, &train_slice, &test, verbose, &params);

    println!("train_acc: {:?} test_acc: {:?} ", train_acc, test_acc)
}

fn _vec_of_vec_to_csv(dump: Vec<Vec<String>>, path: &str) {
    let mut ret_vec: Vec<Vec<String>> = Vec::new();

    let f = File::create(path).unwrap();
    let mut writer = BufWriter::new(f);
    let mut line = String::new();
    for dump_vec in dump {
        line = dump_vec
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
            + "\n";
        let _ = writer.write(line.as_ref());
    }
    writer.flush();
}
fn dump_to_csv(antigens: &Vec<AntiGen>, b_cells: &Vec<BCell>) {
    let csv_formatted_antigens = antigens
        .iter()
        .map(|ag| {
            let mut ret_vec = vec![ag.class_label.to_string(), ag.id.to_string()];
            ret_vec.extend(ag.values.clone().iter().map(|v| v.to_string()));
            return ret_vec;
        })
        .collect::<Vec<_>>();
    _vec_of_vec_to_csv(csv_formatted_antigens, "out/antigens.csv");

    let csv_formatted_b_cells = b_cells
        .iter()
        .map(|cell| {
            let mut ret_vec = vec![
                cell.class_label.to_string(),
                cell.radius_constant.to_string(),
            ];
            ret_vec.extend(cell.dim_values.clone().iter().flat_map(|d| {
                vec![
                    d.value_type.to_string(),
                    d.offset.to_string(),
                    d.multiplier.to_string(),
                ]
            }));
            return ret_vec;
        })
        .collect::<Vec<_>>();
    _vec_of_vec_to_csv(csv_formatted_b_cells, "out/b_cells.csv");
}
fn ais_test(
    antigens: &Vec<AntiGen>,
    train_slice: &Vec<AntiGen>,
    test: &Vec<AntiGen>,
    verbose: bool,
    params: &Params,
) -> (f64, f64) {
    let class_labels = antigens
        .iter()
        .map(|x| x.class_label)
        .collect::<HashSet<_>>();

    if verbose {
        println!(
            "train size: {:?} test size: {:?}",
            train_slice.len(),
            test.len()
        );
    }

    let start = Instant::now();
    let train = train_slice.clone().to_vec();

    let mut ais = ArtificialImmuneSystem::new();
    let (train_acc_hist, train_score_hist, init_scored_pop) = ais.train(&train, &params, verbose);

    if verbose {
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

    let error_match_mask = gen_error_merge_mask(&evaluated_pop);
    let match_mask = gen_merge_mask(&evaluated_pop);

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
    let scored_pop = score_b_cells(evaluated_pop, &match_mask, &error_match_mask, &count_map);

    scored_pop.iter().for_each(|(disc_score, eval, b_cell)| {
        let registered_antigens = test
            .iter()
            .filter(|ag| b_cell.test_antigen(ag))
            .collect::<Vec<_>>();
        let with_same_label = registered_antigens
            .iter()
            .filter(|ag| ag.class_label == b_cell.class_label)
            .collect::<Vec<_>>();
        let num_wrong = registered_antigens
            .iter()
            .filter(|ag| ag.class_label != b_cell.class_label)
            // .inspect(|ag| println!("er ag id {:?}", ag.id))
            .collect::<Vec<_>>();

        let score = with_same_label.len() as f64 / (num_wrong.len() as f64 + 1.0);

        if verbose {
            //registered_antigens.len() > 0 {
            println!(
                "genome dim values    {:?}",
                b_cell
                    .dim_values
                    .iter()
                    .map(|v| v.multiplier)
                    .collect::<Vec<_>>()
            );
            println!(
                "genome offset values {:?}",
                b_cell
                    .dim_values
                    .iter()
                    .map(|v| v.offset)
                    .collect::<Vec<_>>()
            );
            println!(
                "genome value type    {:?}",
                b_cell
                    .dim_values
                    .iter()
                    .map(|v| &v.value_type)
                    .collect::<Vec<_>>()
            );
            println!("genome matches    {:?}", eval.matched_ids);
            println!("genome errors     {:?}", eval.wrongly_matched);

            println!("genome value radius    {:?}", b_cell.radius_constant);

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

    if verbose {
        println!("zero reg cells {}", zero_reg_cells);
        println!("########## error mask \n{:?}", error_match_mask);
        println!("########## match mask \n{:?}", match_mask);

        for n in (0..match_mask.len()) {
            let wrong = error_match_mask.get(n).unwrap();
            let right = match_mask.get(n).unwrap();

            if wrong > right {
                println!("idx: {:2>?}  cor: {:2>?} - wrong {:2>?}", n, right, wrong);
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

    if verbose {
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

    dump_to_csv(antigens, &ais.b_cells);

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

    let mut rng = rand::thread_rng();
    antigens.shuffle(&mut rng);

    let class_labels = antigens
        .iter()
        .map(|x| x.class_label)
        .collect::<HashSet<_>>();

    let mut params = Params {
        // -- train params -- //
        antigen_pop_fraction: 1.0,
        generations: 2000,

        mutation_offset_weight: 3,
        mutation_multiplier_weight: 3,
        mutation_radius_weight: 2,
        mutation_value_type_weight: 0,

        mutation_label_weight: 0,

        // offset_mutation_multiplier_range: 0.8..=1.2,
        // multiplier_mutation_multiplier_range: 0.8..=1.2,
        // radius_mutation_multiplier_range: 0.8..=1.2,
        offset_mutation_multiplier_range: 0.5..=1.5,
        multiplier_mutation_multiplier_range: 0.5..=1.5,
        radius_mutation_multiplier_range: 0.5..=1.5,
        // value_type_valid_mutations: vec![DimValueType::Disabled,DimValueType::Circle],
        value_type_valid_mutations: vec![
            DimValueType::Circle,
            DimValueType::Open,
            DimValueType::Disabled,
        ],
        // value_type_valid_mutations: vec![DimValueType::Circle],
        label_valid_mutations: class_labels.clone().into_iter().collect::<Vec<usize>>(),

        //selection
        leak_fraction: 0.5,
        leak_rand_prob: 0.5,
        max_replacment_frac: 0.7,
        tournament_size: 10,
        n_parents_mutations: 40,

        b_cell_init_expand_radius: true,

        // -- B-cell from antigen initialization -- //
        b_cell_ag_init_multiplier_range: 0.8..=1.2,
        // b_cell_ag_init_value_types: vec![DimValueType::Circle],
        // b_cell_ag_init_value_types: vec![DimValueType::Disabled ,DimValueType::Circle],
        b_cell_ag_init_value_types: vec![
            DimValueType::Circle,
            DimValueType::Disabled,
            DimValueType::Open,
        ],
        b_cell_ag_init_range_range: 0.1..=0.4,

        // -- B-cell from random initialization -- //
        b_cell_rand_init_offset_range: 0.0..=1.0,
        b_cell_rand_init_multiplier_range: 0.8..=1.2,
        // b_cell_rand_init_value_types: vec![DimValueType::Circle, DimValueType::Disabled],
        b_cell_rand_init_value_types: vec![
            DimValueType::Circle,
            DimValueType::Disabled,
            DimValueType::Open,
        ],
        b_cell_rand_init_range_range: 0.1..=0.4,
    };
    modify_config_by_args(&mut params);

    ais_frac_test(params, antigens, true)
    // ais_n_fold_test(params, antigens)
}
