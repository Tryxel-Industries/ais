#![feature(fn_traits)]
#![feature(get_many_mut)]
#![feature(exclusive_range_pattern)]

extern crate core;

use std::collections::{HashMap, HashSet};
use std::ptr::null;
use crate::ais::{ArtificialImmuneSystem, evaluate_population, Params};
use crate::bucket_empire::BucketKing;
use crate::dataset_readers::{read_diabetes, read_iris, read_wine};
use plotters::prelude::*;
use rand::prelude::SliceRandom;
use rand::Rng;
use std::time::Instant;
use crate::evaluation::{gen_error_merge_mask, gen_merge_mask, score_b_cells};

use crate::mutations::mutate;
use crate::representation::{AntiGen, DimValueType};

use crate::selection::selection;

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

fn split_train_test(antigens: &Vec<AntiGen>, test_frac: f64) -> (Vec<AntiGen>, Vec<AntiGen>){
    let classes: HashSet<usize> = antigens.iter().map(|ag| ag.class_label).collect();

    let mut train: Vec<AntiGen> = Vec::new();
    let mut test: Vec<AntiGen> = Vec::new();

    for class in classes{
        let mut of_class: Vec<_> = antigens.iter().filter(|ag|  ag.class_label == class).cloned().collect();
        println!("num in class {:?} is {:?}", class ,of_class.len());
        let num_test = (of_class.len() as f64 * test_frac) as usize;
        let class_train = of_class.split_off(num_test);

        train.extend(class_train);
        test.extend(of_class);

        println!("train s {:?} test s {:?}", train.len() ,test.len());
    }

    return (train,test)
}
fn ais_test() {
    // let mut antigens = read_iris();
    // let mut antigens = read_wine();
    let mut antigens = read_diabetes();

    let class_labels = antigens
            .iter()
            .map(|x| x.class_label)
            .collect::<HashSet<_>>();

    // println!("antigens values    {:?}", antigens.iter().map(|v| &v.values).collect::<Vec<_>>());

    let mut rng = rand::thread_rng();
    antigens.shuffle(&mut rng);

    let (train_slice, test) = split_train_test(&antigens, 0.3);


    let params = Params {
        // -- train params -- //
        antigen_pop_fraction: 1.0,
        generations: 500,

        mutation_offset_weight: 3,
        mutation_multiplier_weight: 3,
        mutation_radius_weight: 2,
        mutation_value_type_weight: 1,

        mutation_label_weight: 0,

        offset_mutation_multiplier_range: 0.8..=1.2,
        multiplier_mutation_multiplier_range: 0.8..=1.2,
        radius_mutation_multiplier_range: 0.8..=1.2,
        value_type_valid_mutations: vec![DimValueType::Disabled,DimValueType::Circle],
        // value_type_valid_mutations: vec![DimValueType::Circle, DimValueType::Open, DimValueType::Disabled],

        label_valid_mutations: class_labels.clone().into_iter().collect::<Vec<usize>>(),

        //selection
        leak_fraction: 0.5,
        leak_rand_prob: 0.5,
        max_replacment_frac: 0.7,
        tournament_size: 5,
        n_parents_mutations: 10,

        b_cell_init_expand_radius: true,

        // -- B-cell from antigen initialization -- //
        b_cell_ag_init_multiplier_range: 0.8..=1.2,
        b_cell_ag_init_value_types: vec![DimValueType::Disabled ,DimValueType::Circle],
        // b_cell_ag_init_value_types: vec![DimValueType::Circle, DimValueType::Disabled, DimValueType::Open],
        b_cell_ag_init_range_range: 0.1..=0.4,

        // -- B-cell from random initialization -- //
        b_cell_rand_init_offset_range: 0.0..=1.0,
        b_cell_rand_init_multiplier_range: 0.8..=1.2,
        // b_cell_rand_init_value_types: vec![DimValueType::Circle, DimValueType::Disabled],
        b_cell_rand_init_value_types: vec![DimValueType::Circle, DimValueType::Disabled, DimValueType::Open],
        b_cell_rand_init_range_range: 0.1..=0.4,
    };




    println!("train size: {:?} test size: {:?}", train_slice.len(), test.len());





    let start = Instant::now();
    let train = train_slice.clone().to_vec();

    let mut ais = ArtificialImmuneSystem::new();
    let (train_acc_hist, train_score_hist,init_scored_pop) = ais.train(&train, &params);

    let duration = start.elapsed();
    plot_hist(train_acc_hist, "acuracy");
    plot_hist(train_score_hist, "score");

    let mut zero_reg_cells = 0;
    // display final

    let n_dims = antigens.get(0).unwrap().values.len();
    let mut bk: BucketKing<AntiGen> =
        BucketKing::new(n_dims, (0.0, 1.0), 10, |ag| ag.id, |ag| &ag.values);
    bk.add_values_to_index(&antigens);

    let pop = init_scored_pop.clone().into_iter().map(|(a, b, c)| c).collect();
    let evaluated_pop = evaluate_population(&bk, &params, pop, &antigens);

    let error_match_mask = gen_error_merge_mask(&evaluated_pop);
    let match_mask = gen_merge_mask(&evaluated_pop);

    let count_map: HashMap<usize,usize> = class_labels.clone()
        .iter()
        .map(|x| {
            (
                x.clone(),
                antigens
                    .iter()
                    .filter(|ag| ag.class_label == *x)
                    .collect::<Vec<&AntiGen>>()
                    .len()
            )
        })
        .collect();
    let scored_pop = score_b_cells(evaluated_pop,&match_mask, &error_match_mask, &count_map);


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

        if true {//registered_antigens.len() > 0 {
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
    println!("zero reg cells {}", zero_reg_cells);


    println!("########## error mask \n{:?}", error_match_mask);
    println!("########## match mask \n{:?}", match_mask);
    for n in (0..match_mask.len()){
        let wrong = error_match_mask.get(n).unwrap();
        let right = match_mask.get(n).unwrap();

        if wrong>right {
            println!("idx: {:2>?}  cor: {:2>?} - wrong {:2>?}", n, right, wrong);
        }
    }

    println!("=============================================================================");
    println!("      TRAIN");
    println!("=============================================================================");
    let mut n_corr = 0;
    let mut per_class_corr:HashMap<usize,usize> = HashMap::new();
    let mut n_wrong = 0;
    let mut n_no_detect = 0;
    for antigen in &train_slice {
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

    println!();
    println!("dataset size {:?}", train.len());
    println!(
        "corr {:?}, false {:?}, no_detect {:?}, frac: {:?}",
        n_corr,
        n_wrong,
        n_no_detect,
        n_corr as f64 / (train_slice.len() as f64)
    );
    println!("per class cor {:?}", per_class_corr);

    println!("=============================================================================");
    println!("      TEST");
    println!("=============================================================================");
    n_corr = 0;
    n_wrong = 0;
    per_class_corr = HashMap::new();
    n_no_detect = 0;
    for antigen in &test {
        let pred_class = ais.is_class_correct(&antigen);
        if let Some(v) = pred_class {
            if v {
                n_corr += 1;
                let class_count = per_class_corr.get(&antigen.class_label).unwrap_or(&0);
                per_class_corr.insert(antigen.class_label, *class_count + 1);
            } else {
                n_wrong += 1
            }
        } else {
            n_no_detect += 1
        }
    }
    println!();
    println!("dataset size {:?}", test.len());
    println!(
        "corr {:?}, false {:?}, no_detect {:?}, frac: {:?}",
        n_corr,
        n_wrong,
        n_no_detect,
        n_corr as f64 / (test.len() as f64)
    );
    println!("per class cor {:?}", per_class_corr);

    println!(
        "Total runtime: {:?}, \nPer iteration: {:?}",
        duration,
        duration.as_nanos() / params.generations as u128
    );
    // ais.pred_class(test.get(0).unwrap());
}
fn main() {
    ais_test()
}
