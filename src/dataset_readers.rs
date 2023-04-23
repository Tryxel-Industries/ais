use json::JsonValue;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

use crate::representation::antigen::AntiGen;

const REF_DATASET_DIR: &str = "./datasets/reference";
const FAKE_NEWS_DATASET_DIR: &str = "./datasets/fake_news";

fn normalize_features(feature_vec: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n_features = feature_vec.first().unwrap().len();

    let mut max_v = Vec::new();
    let mut min_v = Vec::new();

    for i in 0..n_features {
        max_v.push(
            feature_vec
                .iter()
                .map(|x| x.get(i).unwrap())
                .max_by(|a, b| a.total_cmp(b))
                .unwrap()
                .clone(),
        );

        min_v.push(
            feature_vec
                .iter()
                .map(|x| x.get(i).unwrap())
                .min_by(|a, b| a.total_cmp(b))
                .unwrap()
                .clone(),
        );
    }

    let ret: Vec<_> = feature_vec
        .into_iter()
        .map(|mut f_vals| {
            for i in 0..n_features {
                let min = min_v.get(i).unwrap();
                let max = max_v.get(i).unwrap();

                let val = f_vals[i].clone();
                let result = (val - min) / (max - min);
                f_vals[i] = if result.is_nan() {0.0f64} else {result}
            }
            return f_vals;
        })
        .collect();
    return ret;
}

fn normalize_features_ag(mut ag_vec: Vec<AntiGen>) -> Vec<AntiGen> {
    let mut feature_vec: Vec<_> = ag_vec.iter().map(|ag| ag.values.clone()).collect();
    let norm_features = normalize_features(feature_vec);
    ag_vec
        .iter_mut()
        .zip(norm_features)
        .for_each(|(ag, norm_f)| ag.values = norm_f);
    return ag_vec;
}

fn read_csv(path: &str) -> Vec<Vec<String>> {
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

pub fn read_iris() -> Vec<AntiGen> {
    let mut data_vec = read_csv(format!("{}/iris/iris.data", REF_DATASET_DIR).as_str());

    let (labels, dat): (Vec<String>, Vec<Vec<String>>) = data_vec
        .into_iter()
        .map(|mut row| (row.pop().unwrap(), row))
        .unzip();

    let mut transformed_dat: Vec<Vec<f64>> = dat
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<f64>>()
        })
        .collect();

    transformed_dat = normalize_features(transformed_dat);

    let antigens = labels
        .into_iter()
        .zip(transformed_dat.into_iter())
        .enumerate()
        .map(|(n, (label, features))| {
            let label_val = match label.as_str() {
                "Iris-setosa" => 0,
                "Iris-versicolor" => 1,
                "Iris-virginica" => 2,
                _ => {
                    panic!("parsing error ")
                }
            };
            return AntiGen::new(n, label_val, features)
        })
        .collect();

    return antigens;
}

pub fn read_iris_snipped() -> Vec<AntiGen> {
    let mut ag = read_iris();
    return ag
        .into_iter()
        .map(|mut ag| {
            let _ = ag.values.pop();
            return ag;
        })
        .collect::<Vec<_>>();
}

pub fn read_wine() -> Vec<AntiGen> {
    let mut data_vec = read_csv(format!("{}/wine/wine.data", REF_DATASET_DIR).as_str());

    let (labels, dat): (Vec<String>, Vec<Vec<String>>) = data_vec
        .into_iter()
        .map(|mut row| (row.remove(0), row))
        .unzip();

    let mut transformed_dat: Vec<Vec<f64>> = dat
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<f64>>()
        })
        .collect();

    transformed_dat = normalize_features(transformed_dat);

    let antigens = labels
        .into_iter()
        .zip(transformed_dat.into_iter())
        .enumerate()
        .map(|(n, (label, features))| {
            let label_val = label.parse().unwrap();


            return AntiGen::new(n, label_val, features)

        })
        .collect();

    return antigens;
}

pub fn read_diabetes() -> Vec<AntiGen> {
    let mut data_vec = read_csv(format!("{}/diabetes/diabetes.csv", REF_DATASET_DIR).as_str());

    data_vec.remove(0);
    let (labels, dat): (Vec<String>, Vec<Vec<String>>) = data_vec
        .into_iter()
        .map(|mut row| (row.pop().unwrap(), row))
        .unzip();

    let mut transformed_dat: Vec<Vec<f64>> = dat
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<f64>>()
        })
        .collect();

    transformed_dat = normalize_features(transformed_dat);

    let antigens = labels
        .into_iter()
        .zip(transformed_dat.into_iter())
        .enumerate()
        .map(|(n, (label, features))| {
            let label_val = label.parse().unwrap();

            return AntiGen::new(n, label_val, features)

        })
        .collect();

    return antigens;
}

pub fn read_sonar() -> Vec<AntiGen> {
    let mut data_vec = read_csv(format!("{}/sonar/sonar.all-data", REF_DATASET_DIR).as_str());

    let (labels, dat): (Vec<String>, Vec<Vec<String>>) = data_vec
        .into_iter()
        .map(|mut row| (row.pop().unwrap(), row))
        .unzip();

    let mut transformed_dat: Vec<Vec<f64>> = dat
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<f64>>()
        })
        .collect();

    transformed_dat = normalize_features(transformed_dat);

    let antigens = labels
        .into_iter()
        .zip(transformed_dat.into_iter())
        .enumerate()
        .map(|(n, (label, features))| {
            let label_val = match label.as_str() {
                "R" => 0,
                "M" => 1,
                _ => {
                    panic!("parsing error ")
                }
            };
             return AntiGen::new(n, label_val, features)

        })
        .collect();

    return antigens;
}

pub fn read_glass() -> Vec<AntiGen> {
    let mut data_vec = read_csv(format!("{}/glass/glass.data", REF_DATASET_DIR).as_str());

    let (labels, dat): (Vec<String>, Vec<Vec<String>>) = data_vec
        .into_iter()
        .map(|mut row| {
            //remove index
            row.remove(0);

            let label = row.pop().unwrap();
            return (label, row);
        })
        .unzip();

    let mut transformed_dat: Vec<Vec<f64>> = dat
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<f64>>()
        })
        .collect();

    transformed_dat = normalize_features(transformed_dat);

    let antigens = labels
        .into_iter()
        .zip(transformed_dat.into_iter())
        .enumerate()
        .map(|(n, (label, features))| {
            let label_val = label.parse().unwrap();
                   return AntiGen::new(n, label_val, features)

        })
        .collect();

    return antigens;
}

pub fn read_ionosphere() -> Vec<AntiGen> {
    let mut data_vec = read_csv(format!("{}/ionosphere/ionosphere.data", REF_DATASET_DIR).as_str());

    let (labels, dat): (Vec<String>, Vec<Vec<String>>) = data_vec
        .into_iter()
        .map(|mut row| {
            let label = row.pop().unwrap();
            return (label, row);
        })
        .unzip();

    let mut transformed_dat: Vec<Vec<f64>> = dat
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<f64>>()
        })
        .collect();

    transformed_dat = normalize_features(transformed_dat);

    let antigens = labels
        .into_iter()
        .zip(transformed_dat.into_iter())
        .enumerate()
        .map(|(n, (label, features))| {
            let label_val = match label.as_str() {
                "b" => 0,
                "g" => 1,
                _ => {
                    panic!("parsing error ")
                }
            };
                  return AntiGen::new(n, label_val, features)

        })
        .collect();
    return antigens;
}

pub fn read_pima_diabetes() -> Vec<AntiGen> {
    let mut data_vec =
        read_csv(format!("{}/pima_diabetes/pima_diabetes.csv", REF_DATASET_DIR).as_str());

    data_vec.remove(0);
    let (labels, dat): (Vec<String>, Vec<Vec<String>>) = data_vec
        .into_iter()
        .map(|mut row| {
            //remove index

            let label = row.pop().unwrap();
            return (label, row);
        })
        .unzip();

    let mut transformed_dat: Vec<Vec<f64>> = dat
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|v| {
                    let parsed = v.parse::<f64>();
                    if parsed.is_err() {
                        println!("err line {:?}", v)
                    }
                    return parsed.unwrap();
                })
                .collect::<Vec<f64>>()
        })
        .collect();

    transformed_dat = normalize_features(transformed_dat);

    let antigens = labels
        .into_iter()
        .zip(transformed_dat.into_iter())
        .enumerate()
        .map(|(n, (label, features))| {
            let label_val = label.parse().unwrap();
             return AntiGen::new(n, label_val, features)

        })
        .collect();

    return antigens;
}

pub fn read_spirals() -> Vec<AntiGen> {
    let mut data_vec = read_csv(format!("{}/spirals/spirals.csv", REF_DATASET_DIR).as_str());

    let (labels, dat): (Vec<String>, Vec<Vec<String>>) = data_vec
        .into_iter()
        .map(|mut row| {
            //remove index

            let label = row.pop().unwrap();
            return (label, row);
        })
        .unzip();

    let mut transformed_dat: Vec<Vec<f64>> = dat
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|v| {
                    let parsed = v.parse::<f64>();
                    if parsed.is_err() {
                        println!("err line {:?}", v)
                    }
                    return parsed.unwrap();
                })
                .collect::<Vec<f64>>()
        })
        .collect();

    transformed_dat = normalize_features(transformed_dat);

    let antigens = labels
        .into_iter()
        .zip(transformed_dat.into_iter())
        .enumerate()
        .map(|(n, (label, features))| {
            let label_val = label.parse().unwrap();
               return AntiGen::new(n, label_val, features)

        })
        .collect();

    return antigens;
}

//
//   Fake news
//

fn read_json(path: &str) -> JsonValue {
    let mut lines: Vec<String> = Vec::new();

    let f = File::open(path).unwrap();
    let mut reader = BufReader::new(f);
    let mut line = String::new();

    loop {
        let len = reader.read_line(&mut line).unwrap();

        if line.ends_with("\n") {
            line = line.strip_suffix("\n").unwrap().parse().unwrap();
        }

        if line.len() > 0 {
            lines.push(line.clone());
            line.clear();
        } else {
            break;
        }
    }
    let json_str = lines.join("");
    let js_obj = json::parse(&*json_str).unwrap();
    return js_obj;
}

pub fn read_kaggle_semantic() -> Vec<AntiGen> {
    let path = format!(
        "{}/kaggle/semantic_features_kaggle.json",
        FAKE_NEWS_DATASET_DIR
    );
    let mut json_value = read_json(path.as_str());

    json_value.entries().for_each(|(k, v)| println!("{:?}", k));

    let mut news_entries: Vec<AntiGen> = Vec::new();

    loop {
        let mut val = json_value.pop();
        if val.is_null() {
            break;
        } else {
            let id = val.remove("id").as_i32().unwrap();
            let title: String = val.remove("title").to_string();
            let pub_date: String = val.remove("publishDate").dump();
            let label: String = val.remove("label").to_string();
            let mut res_map = val.remove("resultMap");

            let mut feature_values: Vec<f64> = Vec::new();
            res_map.entries_mut().for_each(|(k, v)| {
                let f_val = v.remove("featureValue").as_f64().unwrap();
                feature_values.push(f_val);
            });



                        let ag = AntiGen::new(id as usize, label.parse().unwrap(), feature_values);

            news_entries.push(ag)
        }
    }

    news_entries = normalize_features_ag(news_entries);
    return news_entries;
}
