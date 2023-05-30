use std::cmp::Ordering;
use std::collections::HashSet;
use std::ops::Add;

/// Given a vector of vector of sorted usize values `lists`, this function compares the values of each list
/// and returns a new vector containing values that are found in every list.
/// If the given `lists` is an empty vector, an empty vector is returned.
/// If the given `lists` contains only one vector, that vector is returned as is.
///
/// # Arguments
///
/// * `lists` - A vector of vectors of usize values.
///
/// # Returns
///
/// An optional vector of usize values. If the function succeeds, it returns a vector containing
/// values that are found in every list. If the given `lists` is an empty vector,
/// an empty vector is returned. If the given `lists` contains only one vector, that vector is returned
/// as is. If the function fails to produce a result, it returns `None`.
fn roll_comparison(lists: Vec<Vec<usize>>) -> Option<Vec<usize>> {
    // todo: mabye improve speed by sorting by shortest and using that as start point

    let n_lists = lists.len();

    return if n_lists == 0 {
        // if no lists are given return a new empty vec
        Some(Vec::new())
    } else if n_lists == 1 {
        // if only one lists, this lists matches with itself and is therefore returned
        Some(lists.first().unwrap().clone())
    } else {
        let mut found_matches: Vec<usize> = Vec::new();

        let mut current_idx_list = vec![0 as usize; lists.len()];
        let mut max_idx_list: Vec<usize> = lists.iter().map(|x| x.len()).collect();
        if *max_idx_list.iter().min().unwrap() == 0 {
            // if any of the given lists are empty there wont be any matches so we return an empty vec
            return Some(Vec::new());
        }

        let (first, rest_of_lists) = lists.split_at(1);
        let first_list_value = first.get(0)?;
        let mut first_idx: usize = 0;

        'outer: loop {
            // println!("outer, cnt {:?} brk& {:?}", first_idx, *max_idx_list.get_mut(0)?);
            if first_idx >= *max_idx_list.get_mut(0)? {
                break 'outer;
            }
            let check_val = first_list_value.get(first_idx).unwrap();

            let mut found_match = true;

            'row: for (n, list) in rest_of_lists.iter().enumerate() {
                'internal: loop {
                    let idx_val = current_idx_list.get_mut(n + 1).unwrap();
                    let max_idx_val = max_idx_list.get_mut(n + 1).unwrap();

                    if idx_val >= max_idx_val {
                        break 'outer;
                    }

                    let value = list.get(*idx_val).unwrap();

                    // println!("internal, cnt {:?} brk& {:?}", value, check_val);
                    if value > check_val {
                        // if on of the lower cols has a bigger value than
                        // the check val the check val needs to be increased
                        found_match = false;
                        break 'row;
                    } else if value < check_val {
                        // if the value of the lover cols has a smaller value
                        // than the check val increase the lower col val
                        *idx_val = idx_val.add(1);
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
        Some(found_matches)
    }
}

//
// structs
//

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ValueRangeType {
    Open,
    Symmetric((f64, f64)),
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

// todo: the example here is not good (chatgpt used for docs)

/// A data structure for multi-dimensional indexing of data points. It uses a bucketing approach,
/// where each dimension is divided into several buckets and data points are placed into the
/// corresponding buckets according to their values in each dimension. This allows for fast
/// retrieval of potential matches based on a given query.
///
/// # Examples
///
/// ```
/// use my_crate::BucketKing;
///
/// // Define a simple struct to index
/// #[derive(Clone, PartialEq)]
/// struct Person {
///     name: String,
///     age: u8,
/// }
///
/// // Create a BucketKing that indexes Person objects by age
/// let bucket_range = (0.0, 100.0);
/// let num_buckets = 10;
/// let mut bucket_king = BucketKing::new(
///     1,
///     bucket_range,
///     num_buckets,
///     |person: &Person| person.age as usize,
///     |person: &Person| vec![person.age as f64],
/// );
///
/// // Add some Person objects to the index
/// let alice = Person {
///     name: "Alice".to_string(),
///     age: 25,
/// };
/// let bob = Person {
///     name: "Bob".to_string(),
///     age: 35,
/// };
/// bucket_king.add_values_to_index(&vec![alice, bob]);
///
/// // Get the indexes of potential matches for a given value
/// let potential_matches = bucket_king.get_potential_matches_indexes(&bob);
/// assert_eq!(potential_matches.unwrap(), vec![3, 4, 5]);
/// ```
pub struct BucketKing<T> {
    // he who rules the buckets
    dimensional_knights: Vec<BucketKnight>,
    index_fn: fn(&T) -> usize,
    value_fn: fn(&T) -> &Vec<f64>,
    full_index_set: HashSet<usize>,
}

//
// impl
//

impl Bucket {
    fn add_items(&mut self, items: Vec<BucketValue>) {
        self.bucket_contents.extend(items);
    }
    fn add_item(&mut self, item: BucketValue) {
        self.bucket_contents.push(item);
    }
    fn sort(&mut self) {
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

    pub fn get_index_in_range(&self, range: &ValueRangeType) -> Option<Vec<usize>> {
        let mut ret: Vec<usize> = Vec::new();

        match range {
            ValueRangeType::Open => return None,
            ValueRangeType::Symmetric((lb, ub)) => {
                let value_lb = *lb;
                let value_ub = *ub;

                for bucket in self.buckets.iter() {
                    let contains_left_border =
                        bucket.start_value < value_lb && bucket.end_value > value_lb;
                    let contains_right_border =
                        bucket.start_value < value_ub && bucket.end_value > value_ub;
                    let contains_center =
                        bucket.start_value > value_lb && bucket.end_value < value_ub;

                    if contains_left_border | contains_center | contains_right_border {
                        ret.extend(bucket.bucket_contents.iter().map(|x1| x1.index).clone())
                    }
                }
            }
        }

        ret.sort();
        // println!("ret size {:?}", ret.len());
        return Some(ret);
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
    fn add_items(&mut self, values: Vec<BucketValue>) {
        for value in values {
            let bucket = self.get_bucket_mut(&value.value);
            bucket.unwrap().add_item(value);
        }
        self.sort_buckets();
    }
}

impl<T> BucketKing<T> {
    /// Creates a new `BucketKing` instance with the given parameters.
    ///
    /// The `BucketKing` will use `n_dims` `BucketKnight`s to index elements based on their values.
    /// Each `BucketKnight` will contain `num_buckets` `Bucket`s, with each bucket representing a range of
    /// values in the range specified by `bucket_range`. The `index_fn` parameter is a function that takes an
    /// element of type `T` and returns an index that corresponds to a specific `Bucket` in the `BucketKing`.
    /// The `value_fn` parameter is a function that takes an element of type `T` and returns a vector of values
    /// that will be used to determine which `Bucket`s the element belongs to.
    ///
    /// # Arguments
    ///
    /// * `n_dims` - The number of dimensions that the `BucketKing` will index elements by.
    /// * `bucket_range` - A tuple specifying the range of values that will be used to create the `Bucket`s.
    /// * `num_buckets` - The number of `Bucket`s that will be created for each `BucketKnight`.
    /// * `index_fn` - A function that takes an element of type `T` and returns an index that corresponds to a
    ///                specific `Bucket` in the `BucketKing`.
    /// * `value_fn` - A function that takes an element of type `T` and returns a vector of values that will be
    ///                used to determine which `Bucket`s the element belongs to.
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

        let bucket_knights: Vec<BucketKnight> = (0..n_dims)
            .map(|i| {
                let mut buckets: Vec<Bucket> = Vec::new();
                for j in 0..num_buckets {
                    let start_value = if j == 0 {
                        f64::MIN
                    } else {
                        r_from + j as f64 * bucket_step
                    };
                    let end_value = if j == (num_buckets - 1) {
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
            full_index_set: HashSet::new(),
        };
    }

    pub fn get_potential_matches(&self, ranges: &Vec<ValueRangeType>) -> Option<Vec<usize>> {
        let mut ret: Vec<Vec<usize>> = self
            .dimensional_knights
            .iter()
            .filter_map(|k| k.get_index_in_range(ranges.get(k.dimension).unwrap()))
            .collect();

        if ret.len() == 0 {
            // all the dimensions are open
            return Some(self.full_index_set.clone().into_iter().collect());
        }

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
                    self.full_index_set.insert(index.clone());
                    return BucketValue {
                        value: value_vec.get(n).unwrap().clone(),
                        index: index.clone(),
                    };
                })
                .collect();
            dimensional_knight.add_items(as_bucket_values);
        }

        // self.dimensional_knights.iter().for_each(|x| {
        //     println!("dim {:?}", x.dimension);
        //     x.buckets.iter().for_each(|y|{
        //         println!("{:?}", y.bucket_contents);
        //         // println!("{:?}",y.bucket_contents.iter().map(|x1|x1.index).collect::<Vec<usize>>())
        //     })
        // })
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
