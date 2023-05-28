use std::{collections::HashSet, f64::consts::PI, vec};

use crate::representation::{antibody::Antibody, antibody::AntibodyDim, antigen::AntiGen};
use core::{self, num};
use rand::{random, seq::IteratorRandom, Rng};
use rand_distr::{Distribution, Normal};

use crate::params::Params;
use crate::prediction::is_class_correct;
use crate::representation::antigen::AntiGenSplitShell;
use nalgebra::DMatrix;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufWriter, Write};

/// Given a vector `vec` and a positive integer `n`, returns a new vector that contains `n`
/// randomly chosen elements from `vec`. The elements in the resulting vector are in the same
/// order as they appear in `vec`.
///
/// # Arguments
///
/// * `vec` - A vector of type `T` that contains the elements to choose from.
/// * `n` - The number of elements to randomly choose from `vec`.
///
/// # Returns
///
/// A new vector of type `T` that contains `n` randomly chosen elements from `vec`.
///
/// # Panics
///
/// This function panics if `n` is greater than the length of `vec`.
///
/// # Example
///
/// ```
/// use rand::prelude::*;
///
/// let vec = vec![1, 2, 3, 4, 5];
/// let picks = pick_n_random(vec, 3);
/// assert_eq!(picks.len(), 3);
/// ```

// pub fn construct_orthonormal_vecs(population: &Vec<Antibody>) -> nalgebra::base::Matrix1<f64> {
pub fn construct_orthonormal_vecs(population: &[Antibody]) -> nalgebra::base::Matrix1<f64> {
    let nf: usize = 8;

    let t: Vec<f64> = population
        .iter()
        .map(|ab| &ab.dim_values)
        .map(|f| f.iter().map(|m| m.offset).collect::<Vec<f64>>())
        .flatten()
        .collect();
    let mut dm1 = DMatrix::from_vec(nf, population.len(), t.clone()).transpose();
    println!("{:?}", t.clone());
    println!("{}", dm1);
    println!("skadoosh");
    mutate_orientation(&mut dm1);
    todo!()
}

pub fn mutate_orientation(mut m: &mut DMatrix<f64>) {
    let q = m.shape().1;
    let mut rng = rand::thread_rng();
    let distr = Normal::new(0.0, PI / 2.0).unwrap();
    let theta = distr.sample(&mut rng);
    let column_idxs: Vec<usize> = m
        .column_iter()
        .enumerate()
        .choose_multiple(&mut rng, 2)
        .iter()
        .map(|f| f.0)
        .collect();

    let c1 = m.column(column_idxs[0]);
    let c2 = m.column(column_idxs[1]);
    let tmp = c1 * theta.cos() + c2 * theta.sin();
    let tmp2 = c1 * -theta.sin() + c2 * theta.cos();

    m.column_mut(column_idxs[0]).copy_from(&tmp);
    m.column_mut(column_idxs[1]).copy_from(&tmp2);

    println!("{:?}", theta);
    println!("{}", m);
    // println!("{:?}", q.clone());
    todo!()
}

pub fn pick_n_random<T>(vec: Vec<T>, n: usize) -> Vec<T> {
    let mut rng = rand::thread_rng();

    let mut idx_list = Vec::new();
    while idx_list.len() < n {
        let idx = rng.gen_range(0..vec.len());
        if !idx_list.contains(&idx) {
            idx_list.push(idx.clone());
        }
    }

    idx_list.sort();

    let picks = vec
        .into_iter()
        .enumerate()
        .filter(|(idx, _v)| idx_list.contains(idx))
        .map(|(_a, b)| b)
        .collect();
    return picks;
}

/// Splits a collection of AntiGen instances into training and testing sets, with the specified
/// proportion of instances reserved for testing. Returns a tuple containing the training set
/// followed by the test set.
///
/// # Arguments
///
/// * `antigens` - A vector containing the AntiGen instances to be split.
/// * `test_frac` - The fraction of instances to be reserved for testing, as a float between 0 and 1.
///
/// # Examples
///
/// ```
/// let antigens = vec![...];
/// let (train, test) = split_train_test(&antigens, 0.2);
/// ```
pub fn split_train_test(
    antigens: &Vec<AntiGenSplitShell>,
    test_frac: f64,
) -> (Vec<AntiGen>, Vec<AntiGen>) {
    let classes: HashSet<usize> = antigens.iter().map(|ag| ag.class_label).collect();

    println!("clasess: {:?}", classes);
    let mut train: Vec<_> = Vec::new();
    let mut test: Vec<_> = Vec::new();

    for class in classes {
        let mut of_class: Vec<_> = antigens
            .iter()
            .filter(|ag| ag.class_label == class)
            .cloned()
            .collect();
        let num_test = (of_class.len() as f64 * test_frac) as usize;
        println!(
            "in class {:?} -> train: {:?} - test: {:?}",
            of_class.len(),
            of_class.len() - num_test,
            num_test
        );
        let class_train = of_class.split_off(num_test);

        train.extend(class_train);
        test.extend(of_class);
    }

    let train_unpacked = train.into_iter().flat_map(|ag| ag.upack()).collect();
    let test_unpacked = test.into_iter().flat_map(|ag| ag.upack()).collect();
    return (train_unpacked, test_unpacked);
}

/// Splits a collection of AntiGen instances into train and test sets using a "n-fold" cross-validation
/// method, with the specified number of folds. Returns a vector of tuples, each containing a training
/// set and a corresponding test set.
///
/// # Arguments
///
/// * `antigens` - A vector containing the AntiGen instances to be split.
/// * `n_folds` - The number of folds to use in the cross-validation. Must be a positive integer.
///
/// # Examples
///
/// ```
/// let antigens = vec![...];
/// let folds = split_train_test_n_fold(&antigens, 5);
/// for (train, test) in folds {
///     // do something with the train and test sets
/// }
/// ```
pub fn split_train_test_n_fold(
    antigens: &Vec<AntiGenSplitShell>,
    n_folds: usize,
) -> Vec<(Vec<AntiGen>, Vec<AntiGen>)> {
    let classes: HashSet<usize> = antigens.iter().map(|ag| ag.class_label).collect();
    let fold_frac = 1.0 / n_folds as f64;

    let mut folds: Vec<Vec<AntiGenSplitShell>> = Vec::new();
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
        let train_fold_unpacked = train_fold.into_iter().flat_map(|ag| ag.upack()).collect();
        let test_fold_unpacked = test_fold.into_iter().flat_map(|ag| ag.upack()).collect();
        ret_folds.push((train_fold_unpacked, test_fold_unpacked))
    }

    // println!("folds {:?}", folds);
    return ret_folds;
}

pub fn read_csv(path: &str) -> Vec<Vec<String>> {
    let mut ret_vec: Vec<Vec<String>> = Vec::new();

    let f = File::open(path).unwrap();
    let mut reader = BufReader::new(f);
    let mut line = String::new();

    loop {
        let len = reader.read_line(&mut line).unwrap();

        if line.ends_with("\n") {
            line = line.strip_suffix("\n").unwrap().parse().unwrap();
        }
        if line.ends_with("\r") {
            line = line.strip_suffix("\r").unwrap().parse().unwrap();
        }

        if line.len() > 0 {
            let cols = line.split(",");
            ret_vec.push(cols.into_iter().map(|s| String::from(s)).collect());
            line.clear();
        } else {
            break;
        }
    }

    // println!("{:?}", ret_vec);
    return ret_vec;
}

pub fn get_pop_acc(
    antigens: &Vec<AntiGen>,
    antibodies: &Vec<Antibody>,
    params: &Params,
) -> (usize, usize, usize) {
    let mut n_corr = 0;
    let mut n_wrong = 0;
    let mut n_no_detect = 0;
    for antigen in antigens {
        let pred = is_class_correct(&antigen, &antibodies, &params.eval_method);
        if let Some(v) = pred {
            if v {
                n_corr += 1
            } else {
                n_wrong += 1
            }
        } else {
            n_no_detect += 1
        }
    }
    return (n_corr, n_wrong, n_no_detect);
}
