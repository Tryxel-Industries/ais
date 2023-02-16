use csv::StringRecord;
use crate::representation::AntiGen;

use serde::{Serialize, Deserialize};

#[derive(Debug, Deserialize)]
struct Record {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    class: String
}

pub fn read_iris() -> Vec<AntiGen>{
    let mut reader = csv::Reader::from_path("./datasets/iris/iris.data").unwrap();
    reader.set_headers(StringRecord::from(vec!["sepal_length",
                                               "sepal_width",
                                               "petal_length",
                                               "petal_width", "class"]));

    let mut records = Vec::new();
    for res in reader.deserialize(){
        let record: Record = res.unwrap();
        records.push(record);
    }
    return records.into_iter().enumerate().map(|(n,record) | {
        let label = match record.class.as_str() {
            "Iris-setosa" => 1,
            "initialization" => 2,
            "Iris-versicolor" => 3,
            "Iris-virginica" => 4,
            _ => {
                panic!("parsing error ")
            }
        };
        return AntiGen{
            id: n,
            class_label: label,
            values: vec![record.sepal_length, record.sepal_width, record.petal_length, record.petal_width],
        };
    }).collect();
}