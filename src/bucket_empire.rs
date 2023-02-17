use crate::enums::TrueFalseLabel::True;
use std::borrow::BorrowMut;
use std::cmp::{max, min, Ordering};
use std::iter::Map;
use std::ops::{Add, Range};
use std::slice::SliceIndex;

fn roll_comparison(lists: Vec<Vec<usize>>) -> Option<Vec<usize>> {
    //  a = 1,4,6,7,8
    //  b = 1,2,7,9
    //
    //
    //

    // println!("in: {:?}", lists);

    let mut found_matches: Vec<usize> = Vec::new();

    let mut current_idx_list = vec![0 as usize; lists.len()];
    let mut max_idx_list: Vec<usize> = lists.iter().map(|x| x.len()).collect();

    let (first, rest_of_lists) = lists.split_at(1);
    let first_list_value = first.get(0)?;
    let mut first_idx: usize = 0;

    'outer: loop {
        // println!("outer");
        if first_idx >= *max_idx_list.get_mut(0)? {
            break 'outer;
        }
        let check_val = first_list_value.get(first_idx).unwrap();

        let mut found_match = true;

        'row: for (n, list) in rest_of_lists.iter().enumerate() {
            'internal: loop {
                let mut idx_val = current_idx_list.get_mut(n + 1).unwrap();
                let mut max_idx_val = max_idx_list.get_mut(n + 1).unwrap();

                if idx_val >= max_idx_val {
                    break 'outer;
                }

                let value = list.get(*idx_val).unwrap();
                if value > check_val {
                    // if on of the lower cols has a bigger value than
                    // the check val the check val needs to be increased
                    found_match = false;
                    break 'row;
                } else if value < check_val {
                    // if the value of the lover cols has a smaller value
                    // than the check val increase the lower col val
                    // println!("befor {}",idx_val);
                    *idx_val = idx_val.add(1);
                    // println!("after {}",idx_val);
                } else {
                    // if the values are equal continue the iteration of the cols
                    break 'internal;
                }
            }
        }

        if found_match {
            found_matches.push(check_val.clone());
        }

        first_idx += 1;
    }
    // println!("out: {:?}", found_matches);
    return Some(found_matches);
}
//
// Traits
//
pub trait Bucketable {
    fn get_dimensional_value(&self, dim: usize) -> &f64;
}

//
// structs
//

pub enum BucketEmpireOfficialRangeNotationSystemClasses {
    Open,
    Symmetric(f64),
    UpperBound(f64),
    LowerBound(f64),
}

#[derive(Debug, Copy, Clone)]
struct BucketValue {
    index: usize,
    value: f64,
}

pub struct Bucket {
    // The plebs of the bucket empire
    bucket_contents: Vec<BucketValue>,
    // inclusive
    start_value: f64,
    // exclusive
    end_value: f64,
}

pub struct BucketKnight {
    // The executive arm of the bucket empire
    buckets: Vec<Bucket>,
    dimension: usize,
}

pub struct BucketKing<T> {
    // he who rules the buckets
    dimensional_knights: Vec<BucketKnight>,
    index_fn: fn(&T) -> usize,
    value_fn: fn(&T) -> &Vec<f64>,
}

//
// impl
//

impl Bucket {
    pub fn add_items(&mut self, items: Vec<BucketValue>) {
        self.bucket_contents.extend(items);
    }
    pub fn add_item(&mut self, item: BucketValue) {
        self.bucket_contents.push(item);
    }

    pub fn sort(&mut self) {
        self.bucket_contents.sort_unstable_by_key(|k| k.index);
    }
}

impl BucketKnight {
    fn sort_buckets(&mut self) {
        self.buckets.iter_mut().for_each(|x| x.sort())
    }
    pub fn get_bucket(&self, dimensional_value: &f64) -> Option<&Bucket> {
        let value = dimensional_value;
        for bucket in self.buckets.iter() {
            if bucket.start_value < *value {
                if bucket.end_value > *value {
                    return Some(bucket);
                }
            }
        }
        return None;
    }

    pub fn get_index_in_range(
        &self,
        dimensional_value: &f64,
        range: &BucketEmpireOfficialRangeNotationSystemClasses,
    ) -> Vec<usize> {
        let mut ret: Vec<usize> = Vec::new();

        match range {
            BucketEmpireOfficialRangeNotationSystemClasses::Open => {
                ret = self
                    .buckets
                    .iter()
                    .map(|v| v.bucket_contents.iter())
                    .flat_map(|v| v)
                    .map(|v| v.index)
                    .collect();
            }
            BucketEmpireOfficialRangeNotationSystemClasses::Symmetric(v) => {
                let value_lb = dimensional_value.clone() - v;
                let value_ub = dimensional_value.clone() + v;

                for bucket in self.buckets.iter() {
                    if bucket.end_value > value_lb && bucket.end_value <= value_ub {
                        ret.extend(bucket.bucket_contents.iter().map(|x1| x1.index).clone())
                    } else if bucket.start_value > value_lb && bucket.start_value <= value_ub {
                        ret.extend(bucket.bucket_contents.iter().map(|x1| x1.index).clone())
                    }
                }
            }
            BucketEmpireOfficialRangeNotationSystemClasses::UpperBound(v) => {
                let value_ub = *v;
                for bucket in self.buckets.iter() {
                    if  bucket.end_value <= value_ub {
                        ret.extend(bucket.bucket_contents.iter().map(|x1| x1.index).clone())
                    }
                }
            }
            BucketEmpireOfficialRangeNotationSystemClasses::LowerBound(v) => {
                let value_lb = *v;
                for bucket in self.buckets.iter() {
                    if bucket.start_value > value_lb {
                        ret.extend(bucket.bucket_contents.iter().map(|x1| x1.index).clone())
                    }
                }
            }
        }

        ret.sort();
        // println!("ret size {:?}", ret.len());
        return ret;
    }
    pub fn get_bucket_mut(&mut self, dimensional_value: &f64) -> Option<&mut Bucket> {
        let value = dimensional_value;
        for bucket in self.buckets.iter_mut() {
            // println!("value {:?}, start  {:?}, end {:?}, ",*value, bucket.start_value,bucket.end_value);
            if bucket.start_value < *value {
                if bucket.end_value >= *value {
                    return Some(bucket);
                }
            }
        }
        return None;
    }
    pub fn add_items(&mut self, values: Vec<BucketValue>) {
        for value in values {
            let mut bucket = self.get_bucket_mut(&value.value);
            bucket.unwrap().add_item(value);
        }
        self.sort_buckets();
    }
}

impl<T> BucketKing<T> {
    pub fn new(
        n_dims: usize,
        bucket_range: (f64, f64),
        num_buckets: i32,
        index_fn: fn(&T) -> usize,
        value_fn: fn(&T) -> &Vec<f64>,
    ) -> BucketKing<T> {
        // let mut bucket_knights: Vec<BucketKnight> = Vec::new();
        let (r_from, r_to) = bucket_range;
        let bucket_step = (r_from.max(r_to) - r_from.min(r_to)) / num_buckets as f64;
        // println!("bkt step {}", bucket_step);
        // for i in 0..n_dims {
        //     let mut buckets: Vec<Bucket> = Vec::new();
        //     for j in 0..num_buckets {
        //
        //         let start_value = if j == 0 { f64::MIN } else { r_from + j as f64 * bucket_step };
        //         let mut end_value =  if j == (num_buckets-1) { f64::MAX } else { r_from + (j + 1) as f64 * bucket_step };
        //         // println!("{} to {}", start_value,end_value);
        //
        //         buckets.push(Bucket{
        //             start_value,
        //             end_value,
        //             bucket_contents: Vec::new(),
        //         })
        //
        //     }
        //
        //     bucket_knights.push(BucketKnight{
        //         dimension: i,
        //         buckets
        //     })
        // }

        let bucket_knights: Vec<BucketKnight> = (0..n_dims)
            .map(|i| {
                let mut buckets: Vec<Bucket> = Vec::new();
                for j in 0..num_buckets {
                    let start_value = if j == 0 {
                        f64::MIN
                    } else {
                        r_from + j as f64 * bucket_step
                    };
                    let mut end_value = if j == (num_buckets - 1) {
                        f64::MAX
                    } else {
                        r_from + (j + 1) as f64 * bucket_step
                    };
                    // println!("{} to {}", start_value,end_value);

                    buckets.push(Bucket {
                        start_value,
                        end_value,
                        bucket_contents: Vec::new(),
                    })
                }

                return BucketKnight {
                    dimension: i,
                    buckets,
                };
            })
            .collect();
        return BucketKing::<T> {
            value_fn,
            index_fn,
            dimensional_knights: bucket_knights,
        };
    }
    pub fn get_potential_matches_indexes(&self, value: &T) -> Option<Vec<usize>> {
        let value_vec = (self.value_fn)(value);
        let ret: Vec<Vec<usize>> = self
            .dimensional_knights
            .iter()
            // .map(|k| k.get_bucket(value_vec.get(k.dimension).unwrap()).unwrap())
            // .map(|k| k.get_buckets_in_range(value_vec.get(k.dimension).unwrap(),0.7)).flat_map(|x1| x1.into_iter())
            .map(|k| {
                k.get_index_in_range(
                    value_vec.get(k.dimension).unwrap(),
                    &BucketEmpireOfficialRangeNotationSystemClasses::Symmetric(0.6),
                )
            })
            .collect();
        // .inspect(|x2| println!("dim {:?} to {:?} <> {:?}",x2.start_value, x2.end_value, x2.bucket_contents))
        // .map(|bkt| bkt.bucket_contents.iter().map(|x| x.index.clone()).collect::<Vec<usize>>()).collect();
        return roll_comparison(ret);
    }
    pub fn get_potential_matches_indexes_with_ranges(
        &self,
        value: &T,
        ranges: &Vec<BucketEmpireOfficialRangeNotationSystemClasses>,
    ) -> Option<Vec<usize>> {
        let value_vec = (self.value_fn)(value);
        let ret: Vec<Vec<usize>> = self
            .dimensional_knights
            .iter()
            .map(|k| {
                k.get_index_in_range(
                    value_vec.get(k.dimension).unwrap(),
                    ranges.get(k.dimension).unwrap(),
                )
            })
            .collect();
        return roll_comparison(ret);
    }

    pub fn get_potential_matches_indexes_with_raw_values(
        &self,
        value: &Vec<f64>,
        ranges: &Vec<BucketEmpireOfficialRangeNotationSystemClasses>,
    ) -> Option<Vec<usize>> {
        let value_vec = value;
        let ret: Vec<Vec<usize>> = self
            .dimensional_knights
            .iter()
            .map(|k| {
                k.get_index_in_range(
                    value_vec.get(k.dimension).unwrap(),
                    ranges.get(k.dimension).unwrap(),
                )
            })
            .collect();
        return roll_comparison(ret);
    }
    pub fn add_values_to_index(&mut self, values: &Vec<T>) {
        let value_values: Vec<(usize, &Vec<f64>)> = values
            .iter()
            .map(|x| ((self.index_fn)(x), (self.value_fn)(x)))
            .collect();
        for (n, dimensional_knight) in self.dimensional_knights.iter_mut().enumerate() {
            let as_bucket_values: Vec<BucketValue> = value_values
                .iter()
                .map(|(index, value_vec)| {
                    return BucketValue {
                        value: value_vec.get(n).unwrap().clone(),
                        index: index.clone(),
                    };
                })
                .collect();
            dimensional_knight.add_items(as_bucket_values);
        }

        /*   self.dimensional_knights.iter().for_each(|x| {
            println!("dim {:?}", x.dimension);
            x.buckets.iter().for_each(|y|{
                println!("{:?}", y.bucket_contents);
                // println!("{:?}",y.bucket_contents.iter().map(|x1|x1.index).collect::<Vec<usize>>())
            })
        })*/
    }
}

impl Eq for BucketValue {}

impl PartialEq<Self> for BucketValue {
    fn eq(&self, other: &Self) -> bool {
        self.index.eq(&other.index)
    }
}

impl PartialOrd<Self> for BucketValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        return self.index.partial_cmp(&other.index);
    }
}

impl Ord for BucketValue {
    fn cmp(&self, other: &Self) -> Ordering {
        return self.index.cmp(&other.index);
    }
}
