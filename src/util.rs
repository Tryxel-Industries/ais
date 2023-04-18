use std::collections::HashSet;

use rand::Rng;

use crate::representation::antigen::AntiGen;

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
pub fn split_train_test(antigens: &Vec<AntiGen>, test_frac: f64) -> (Vec<AntiGen>, Vec<AntiGen>) {
    let classes: HashSet<usize> = antigens.iter().map(|ag| ag.class_label).collect();

    let mut train: Vec<AntiGen> = Vec::new();
    let mut test: Vec<AntiGen> = Vec::new();

    for class in classes {
        let mut of_class: Vec<_> = antigens
            .iter()
            .filter(|ag| ag.class_label == class)
            .cloned()
            .collect();
        let num_test = (of_class.len() as f64 * test_frac) as usize;
        let class_train = of_class.split_off(num_test);

        train.extend(class_train);
        test.extend(of_class);
    }

    return (train, test);
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
