use std::fs::File;
use std::io::{BufWriter, Write};

use rayon::prelude::*;

use crate::representation::antibody::{Antibody, AntibodyDim, DimValueType, InitType};
use crate::representation::antigen::AntiGen;
use crate::util::read_csv;
use std::str::FromStr;

fn vec_of_vec_to_csv(dump: Vec<Vec<String>>, path: &str) {
    let f = File::create(path).unwrap();
    let mut writer = BufWriter::new(f);
    let mut line = String::new();
    for dump_vec in dump {
        line = dump_vec
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
            + "\n";
        let _ = writer.write(line.as_ref());
    }
    writer.flush();
}

pub fn dump_to_csv(antigens: &Vec<AntiGen>, antibodies: &Vec<Antibody>) {
    let csv_formatted_antigens = antigens
        .iter()
        .map(|ag| {
            let mut ret_vec = vec![ag.class_label.to_string(), ag.id.to_string()];
            ret_vec.extend(ag.values.clone().iter().map(|v| v.to_string()));
            return ret_vec;
        })
        .collect::<Vec<_>>();
    vec_of_vec_to_csv(csv_formatted_antigens, "out/antigens.csv");

    let csv_formatted_antibodies = antibodies
        .iter()
        .map(|cell| {
            let mut ret_vec = vec![
                cell.class_label.to_string(),
                cell.radius_constant.to_string(),
                cell.boosting_model_alpha.to_string(),
                cell.final_train_label_membership.unwrap().0.to_string(),
                cell.final_train_label_affinity.unwrap().0.to_string(),
            ];
            ret_vec.extend(cell.dim_values.clone().iter().flat_map(|d| {
                vec![
                    d.value_type.to_string(),
                    d.offset.to_string(),
                    d.multiplier.to_string(),
                ]
            }));
            return ret_vec;
        })
        .collect::<Vec<_>>();
    vec_of_vec_to_csv(csv_formatted_antibodies, "out/antibodies.csv");
}

pub fn read_ab_csv(filepath: String) -> Vec<Antibody>{

    let mut data_vec = read_csv(filepath.as_str());

    let dim_feature_cols = (data_vec.get(0).unwrap().len()-5);
    let num_dims = if dim_feature_cols % 3 == 0{
        dim_feature_cols/3
    }else {
        panic!("incorrect ammount of dim cols");
    };

    let antibodies: Vec<Antibody> = data_vec
        .into_iter()
        .map(|mut row| {
            // TODO: if needed this is just inefficient beyond comprehension
            let class_label: usize = row.remove(0).parse().unwrap();
            let radius_constant: f64 = row.remove(0).parse().unwrap();
            let boosting_model_alpha: f64 = row.remove(0).parse().unwrap();
            let final_train_membership: f64 = row.remove(0).parse().unwrap();
            let final_train_affinity: f64 = row.remove(0).parse().unwrap();

            let mut row_values = Vec::new();
            for n in 0..num_dims{
                let base_idx = n * 3;

                let value_type: DimValueType = DimValueType::from_str(&*row.get(base_idx).unwrap()).unwrap();
                let offset: f64 = row.get(base_idx+1).unwrap().parse().unwrap();
                let multiplier: f64 = row.get(base_idx+2).unwrap().parse().unwrap();

                row_values.push(AntibodyDim{
                    multiplier,
                    offset,
                    value_type,
                })
            }
            // row_values.reverse();

            return Antibody{
                dim_values: row_values,
                radius_constant,
                class_label,
                boosting_model_alpha,
                final_train_label_membership: Some((final_train_membership, 1.0-final_train_membership)),
                final_train_label_affinity: Some((final_train_affinity, 1.0-final_train_affinity)),
                init_type: InitType::NA,
                mutation_counter: Default::default(),
                clone_count: 0,
            }

        }).collect();


    return antibodies;


}