
use std::io::prelude::*;
use std::io::BufReader;
use std::fs::File;


use crate::representation::AntiGen;
use csv::StringRecord;

use serde::Deserialize;


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

            for i in 0..n_features{
                let min = min_v.get(i).unwrap();
                let max = max_v.get(i).unwrap();

                let val = f_vals[i].clone();
                f_vals[i] = (val - min) / (max - min);
            }
            return f_vals;
        })
        .collect();

    return ret;
}

fn read_csv(path: &str) -> Vec<Vec<String>> {
    let mut ret_vec : Vec<Vec<String>> = Vec::new();

    let f = File::open(path).unwrap();
    let mut reader = BufReader::new(f);
    let mut line = String::new();

    loop{
        let len = reader.read_line(&mut line).unwrap();

        if line.ends_with("\n"){
            line = line.strip_suffix("\n").unwrap().parse().unwrap();
        }
        if line.ends_with("\r"){
            line = line.strip_suffix("\r").unwrap().parse().unwrap();
        }

        if line.len() > 0{
            let cols = line.split(",");
            ret_vec.push(cols.into_iter().map(|s| String::from(s)).collect());
            line.clear();
        } else {
            break
        }
    }

    // println!("{:?}", ret_vec);
    return ret_vec;
    let mut reader = csv::Reader::from_path("./datasets/wine/wine.data").unwrap();
}




pub fn read_iris() -> Vec<AntiGen> {
    let mut data_vec = read_csv("./datasets/iris/iris.data");

    let (labels, dat): (Vec<String>, Vec<Vec<String>>) = data_vec.into_iter()
        .map(|mut row| (row.pop().unwrap(), row)).unzip();


    let mut transformed_dat: Vec<Vec<f64>> = dat
        .into_iter()
        .map(|v| {
             v.into_iter().map(|v| v.parse::<f64>().unwrap()).collect::<Vec<f64>>()
        }
           ).collect();

    transformed_dat = normalize_features(transformed_dat);

    let antigens = labels
        .into_iter()
        .zip(transformed_dat.into_iter())
        .enumerate()
        .map(|(n,(label, features))|{
            let label_val = match label.as_str() {
                "Iris-setosa" => 0,
                "Iris-versicolor" => 1,
                "Iris-virginica" => 2,
                _ => {
                    panic!("parsing error ")
                }
            };
            return AntiGen {
                id: n,
                class_label: label_val,
                values: features
            };
        }).collect();

    return antigens;

}

pub fn read_iris_snipped() -> Vec<AntiGen> {
    let mut ag = read_iris();
    return ag.into_iter().map(|mut ag| {
        let _ = ag.values.pop();
        return ag
    }).collect::<Vec<_>>();
}


pub fn read_wine() -> Vec<AntiGen> {

    let mut data_vec = read_csv("./datasets/wine/wine.data");

    let (labels, dat): (Vec<String>, Vec<Vec<String>>) = data_vec.into_iter()
        .map(|mut row| (row.remove(0), row)).unzip();


    let mut transformed_dat: Vec<Vec<f64>> = dat
        .into_iter()
        .map(|v| {
            v.into_iter().map(|v| v.parse::<f64>().unwrap()).collect::<Vec<f64>>()
        }
        ).collect();

    transformed_dat = normalize_features(transformed_dat);

    let antigens = labels
        .into_iter()
        .zip(transformed_dat.into_iter())
        .enumerate()
        .map(|(n,(label, features))|{
            let label_val = label.parse().unwrap();

            return AntiGen {
                id: n,
                class_label: label_val,
                values: features
            };
        }).collect();

    return antigens;

}


pub fn read_diabetes() -> Vec<AntiGen> {
    let mut data_vec = read_csv("./datasets/diabetes/diabetes.csv");

    data_vec.remove(0);
    let (labels, dat): (Vec<String>, Vec<Vec<String>>) = data_vec.into_iter()
        .map(|mut row| (row.pop().unwrap(), row)).unzip();


    let mut transformed_dat: Vec<Vec<f64>> = dat
        .into_iter()
        .map(|v| {
             v.into_iter().map(|v| v.parse::<f64>().unwrap()).collect::<Vec<f64>>()
        }
           ).collect();

    transformed_dat = normalize_features(transformed_dat);

    let antigens = labels
        .into_iter()
        .zip(transformed_dat.into_iter())
        .enumerate()
        .map(|(n,(label, features))|{
            let label_val = label.parse().unwrap();

            return AntiGen {
                id: n,
                class_label: label_val,
                values: features
            };
        }).collect();

    return antigens;


}


pub fn read_sonar() -> Vec<AntiGen> {
    let mut data_vec = read_csv("./datasets/sonar/sonar.all-data");

    let (labels, dat): (Vec<String>, Vec<Vec<String>>) = data_vec.into_iter()
        .map(|mut row| (row.pop().unwrap(), row)).unzip();


    let mut transformed_dat: Vec<Vec<f64>> = dat
        .into_iter()
        .map(|v| {
            v.into_iter().map(|v| v.parse::<f64>().unwrap()).collect::<Vec<f64>>()
        }
        ).collect();

    transformed_dat = normalize_features(transformed_dat);

    let antigens = labels
        .into_iter()
        .zip(transformed_dat.into_iter())
        .enumerate()
        .map(|(n,(label, features))|{
            let label_val = match label.as_str() {
                "R" => 0,
                "M" => 1,
                _ => {
                    panic!("parsing error ")
                }
            };
            return AntiGen {
                id: n,
                class_label: label_val,
                values: features
            };
        }).collect();

    return antigens;

}


pub fn read_glass() -> Vec<AntiGen> {
    let mut data_vec = read_csv("./datasets/glass/glass.data");

    let (labels, dat): (Vec<String>, Vec<Vec<String>>) = data_vec.into_iter()
        .map(|mut row| {
            //remove index
            row.remove(0);

            let label = row.pop().unwrap();
            return (label, row)
        }).unzip();


    let mut transformed_dat: Vec<Vec<f64>> = dat
        .into_iter()
        .map(|v| {
            v.into_iter().map(|v| v.parse::<f64>().unwrap()).collect::<Vec<f64>>()
        }
        ).collect();

    transformed_dat = normalize_features(transformed_dat);

    let antigens = labels
        .into_iter()
        .zip(transformed_dat.into_iter())
        .enumerate()
        .map(|(n,(label, features))|{
            let label_val = label.parse().unwrap();
            return AntiGen {
                id: n,
                class_label: label_val,
                values: features
            };
        }).collect();

    return antigens;

}




pub fn read_ionosphere() -> Vec<AntiGen> {

    let mut data_vec = read_csv("./datasets/ionosphere/ionosphere.data");

    let (labels, dat): (Vec<String>, Vec<Vec<String>>) = data_vec.into_iter()
        .map(|mut row| {
            //remove index
            row.remove(0);
            row.remove(0);

            let label = row.pop().unwrap();
            return (label, row)
        }).unzip();

    let mut transformed_dat: Vec<Vec<f64>> = dat
        .into_iter()
        .map(|v| {
            v.into_iter().map(|v| v.parse::<f64>().unwrap()).collect::<Vec<f64>>()
        }
        ).collect();

    transformed_dat = normalize_features(transformed_dat);

    let antigens = labels
        .into_iter()
        .zip(transformed_dat.into_iter())
        .enumerate()
        .map(|(n,(label, features))|{
            let label_val = match label.as_str() {
                "b" => 0,
                "g" => 1,
                _ => {
                    panic!("parsing error ")
                }
            };
            return AntiGen {
                id: n,
                class_label: label_val,
                values: features
            };
        }).collect();

    return antigens;

}
