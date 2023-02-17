use crate::representation::AntiGen;
use csv::StringRecord;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct RecordIris {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    class: String,
}

pub fn read_iris() -> Vec<AntiGen> {
    let mut reader = csv::Reader::from_path("./datasets/iris/iris.data").unwrap();
    reader.set_headers(StringRecord::from(vec![
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]));

    let mut records = Vec::new();
    for res in reader.deserialize() {
        let record: RecordIris = res.unwrap();
        records.push(record);
    }
    return records
        .into_iter()
        .enumerate()
        .map(|(n, record)| {
            let label = match record.class.as_str() {
                "Iris-setosa" => 1,
                "initialization" => 2,
                "Iris-versicolor" => 3,
                "Iris-virginica" => 4,
                _ => {
                    panic!("parsing error ")
                }
            };
            return AntiGen {
                id: n,
                class_label: label,
                values: vec![
                    record.sepal_length,
                    record.sepal_width,
                    record.petal_length,
                    record.petal_width,
                ],
            };
        })
        .collect();
}
#[derive(Debug, Deserialize)]
struct RecordDiabetes {
    Pregnancies: f64,
    Glucose: f64,
    BloodPressure: f64,
    SkinThickness: f64,
    Insulin: f64,
    BMI: f64,
    DiabetesPedigreeFunction: f64,
    Age: f64,
    Outcome: usize,
}

pub fn read_diabetes() -> Vec<AntiGen> {
    let mut reader = csv::Reader::from_path("./datasets/diabetes/diabetes.csv").unwrap();

    let mut records = Vec::new();
    for res in reader.deserialize() {
        let record: RecordDiabetes = res.unwrap();
        records.push(record);
    }
    let mut antigens: Vec<AntiGen> = records
        .into_iter()
        .enumerate()
        .map(|(n, r)| {
            return AntiGen {
                id: n,
                class_label: r.Outcome,
                values: vec![
                    r.Pregnancies,
                    r.Glucose,
                    r.BloodPressure,
                    r.SkinThickness,
                    r.Insulin,
                    r.BMI,
                    r.DiabetesPedigreeFunction,
                    r.Age,
                ],
            };
        })
        .collect();

    let mut max_v = Vec::new();
    let mut min_v = Vec::new();
    for i in 0..=7 {
        max_v.push(
            antigens
                .iter()
                .map(|x| x.values.get(i).unwrap())
                .max_by(|a, b| a.total_cmp(b))
                .unwrap()
                .clone(),
        );

        min_v.push(
            antigens
                .iter()
                .map(|x| x.values.get(i).unwrap())
                .min_by(|a, b| a.total_cmp(b))
                .unwrap()
                .clone(),
        );
    }

    antigens = antigens
        .into_iter()
        .map(|mut ag| {
            for i in 0..=7 {
                let min = min_v.get(i).unwrap();
                let max = max_v.get(i).unwrap();

                let val = ag.values[i].clone();
                ag.values[i] = ((val - min) / (max - min)) * 10.0;
            }
            return ag;
        })
        .collect();

    return antigens;
}
