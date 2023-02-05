#![feature(fn_traits)]

use std::collections::BinaryHeap;
use rand::Rng;
use crate::bucket_empire::BucketKing;
use crate::model::{EmbeddingElement, NewsArticle};

mod model;
mod enums;
mod simple_rr_ais;
mod bucket_empire;

#[derive(Clone)]
struct TestStruct{
    idx: i32,
    a: Vec<f64>,
}
fn main() {

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
