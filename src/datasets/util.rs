use crate::representation::antigen::AntiGen;

pub fn normalize_features(feature_vec: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

pub fn normalize_features_ag(mut ag_vec: Vec<AntiGen>) -> Vec<AntiGen> {
    let mut feature_vec: Vec<_> = ag_vec.iter().map(|ag| ag.values.clone()).collect();
    let norm_features = normalize_features(feature_vec);
    ag_vec
        .iter_mut()
        .zip(norm_features)
        .for_each(|(ag, norm_f)| ag.values = norm_f);
    return ag_vec;
}
