use crate::datasets::REF_DATASET_DIR;
use crate::datasets::util::normalize_features;
use crate::representation::antigen::AntiGen;
use crate::util::read_csv;

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
