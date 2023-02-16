#![feature(fn_traits)]

extern crate core;

use std::collections::BinaryHeap;
use rand::prelude::SliceRandom;
use rand::Rng;
use crate::ais::{ArtificialImmuneSystem, ParamObj};
use crate::bucket_empire::BucketKing;
use crate::dataset_readers::read_iris;
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

fn ais_test(){
    let params = ParamObj{
        b_cell_pop_size: 200,
        generations: 1000,
    };

    let mut antigens = read_iris();

    let mut rng = rand::thread_rng();
    antigens.shuffle(&mut rng);


        println!("dataset size {:?}",antigens.len());


    let (test, train_slice) = antigens.split_at(50);
    let train = train_slice.clone().to_vec();

    let mut ais = ArtificialImmuneSystem::new();
    ais.train(
        &train,
        &params,
        evaluate_b_cell,
        mutate,
        selection
    );

    let score: i32 = test.iter().map(|x| ais.pred_class(x)).sum::<usize>() as i32;

    println!("corr {:?}, false {:?}, frac: {:?}",score, test.len() as i32-score, score as f64/(test.len()as f64));
    // ais.pred_class(test.get(0).unwrap());


}
fn main() {

ais_test()

}

