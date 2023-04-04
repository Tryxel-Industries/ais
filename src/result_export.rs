use std::fs::File;
use std::io::{BufWriter, Write};

use crate::representation::antibody::Antibody;
use crate::representation::antigen::AntiGen;

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
