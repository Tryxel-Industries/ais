#![feature(fn_traits)]
#![feature(get_many_mut)]

extern crate core;

use std::collections::BinaryHeap;
use std::time::Instant;
use plotters::prelude::*;
use rand::prelude::SliceRandom;
use rand::Rng;
use crate::ais::{ArtificialImmuneSystem, ParamObj};
use crate::bucket_empire::BucketKing;
use crate::dataset_readers::{read_diabetes, read_iris};
use crate::evaluation::evaluate_b_cell;
use crate::model::{EmbeddingElement, NewsArticle};
use crate::mutations::mutate;
use crate::representation::AntiGen;
use crate::selection::selection;

mod model;
mod enums;
mod simple_rr_ais;
mod bucket_empire;
mod ais;
pub mod mutations;
pub mod representation;
mod evaluation;
mod selection;
mod dataset_readers;
#[derive(Clone)]
struct TestStruct{
    idx: i32,
    a: Vec<f64>,
}

fn bkt_test(){
    println!("Hello, world!");
    let dims = 500;
    let mut the_king: BucketKing<TestStruct> = BucketKing::new(
        dims,
        (-1.0,1.0),
        4,
        |x| x.idx as usize,
        |x1| &x1.a
    );

    let mut test_dat = Vec::new();

    let mut rng = rand::thread_rng();
    for i in 0..50000 {
        let vals: Vec<f64> = (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect();
        test_dat.push(TestStruct{
            a: vals,
            idx: i
        })

    }
    let vals: Vec<f64> = (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let check_val = TestStruct{
            a: vals,
            idx: 99999};

    let chk2 = check_val.clone();
    the_king.add_values_to_index(&test_dat);



    let res = the_king.get_potential_matches_indexes(&check_val).unwrap();
    println!("{:?}",res);


    let res2 = the_king.get_potential_matches_indexes(&test_dat.get(5).unwrap()).unwrap();
    println!("sanity {:?}",res2);

    // the_king.add_values_to_index(&vec![check_val]);
    // let res = the_king.get_potential_matches_indexes(&chk2).unwrap();
    // println!("{:?}",res);
}


fn plot_hist(hist: Vec<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("train_graph.png", (3000, 1000)).into_drawing_area();
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
            hist.iter().enumerate().map(|(y,x)| (y as f32,*x as f32)),
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
fn ais_test(){
    let params = ParamObj{
        b_cell_pop_size: 400,
        generations: 200,

        mutate_offset: true,
        offset_flip_prob: 0.0,
        offset_max_delta: 0.5,

        mutate_multiplier: true,
        multiplier_flip_prob: 0.0,
        multiplier_max_delta: 0.2,

        mutate_value_type: true,
    };

    // let mut antigens = read_iris();
    let mut antigens =  read_diabetes();

    // println!("antigens values    {:?}", antigens.iter().map(|v| &v.values).collect::<Vec<_>>());

    let mut rng = rand::thread_rng();
    antigens.shuffle(&mut rng);




    let (test, train_slice) = antigens.split_at(150);

        let start = Instant::now();
    let train = train_slice.clone().to_vec();
    let duration = start.elapsed();


    let mut ais = ArtificialImmuneSystem::new();
    let hist = ais.train(
        &train,
        &params,
        mutate,
        selection
    );
    plot_hist(hist);




    let mut zero_reg_cells = 0;
    // display final
    ais.b_cells.iter().for_each(| b_cell| {

        let registered_antigens = test.iter().filter(|ag| b_cell.test_antigen(ag)).collect::<Vec<_>>();
        let with_same_label = registered_antigens.iter().filter(|ag|ag.class_label==b_cell.class_label ).collect::<Vec<_>>();
        let num_wrong = registered_antigens.iter().filter(|ag|ag.class_label!=b_cell.class_label ).collect::<Vec<_>>();

        let score = with_same_label.len()as f64/(num_wrong.len() as f64+1.0);



        if registered_antigens.len() > 0{
            println!("genome dim values    {:?}", b_cell.dim_values.iter().map(|v| v.multiplier).collect::<Vec<_>>());
            println!("genome offset values {:?}", b_cell.dim_values.iter().map(|v| v.offset).collect::<Vec<_>>());
            println!("genome value type    {:?}", b_cell.dim_values.iter().map(|v| &v.value_type).collect::<Vec<_>>());
            println!("genome value radius    {:?}", b_cell.radius_constant);

            println!("num reg {:?} same label {:?} other label {:?}, score {:?}",registered_antigens.len(), with_same_label.len(), num_wrong.len(), score);
            println!()
        }else {
            zero_reg_cells+=1;
        }
    });
    println!("zero reg cells {}", zero_reg_cells);



    let mut n_corr = 0;
    let mut n_wrong = 0;
    let mut n_no_detect= 0;
    for antigen in test {
        let pred_class = ais.is_class_correct(antigen);
        if let Some(v) = pred_class {
            if v{
                n_corr+=1
            } else {
                n_wrong+=1
            }
        }else{
            n_no_detect+=1
        }
    }
    println!();
    println!("dataset size {:?}",antigens.len());
    println!("corr {:?}, false {:?}, no_detect {:?}, frac: {:?}",n_corr, n_wrong, n_no_detect, n_corr as f64/(test.len() as f64));


    println!("Total runtime: {:?}, \nPer iteration: {:?}", duration, duration/ params.generations as u32);
    // ais.pred_class(test.get(0).unwrap());


}
fn main() {

ais_test()

}

