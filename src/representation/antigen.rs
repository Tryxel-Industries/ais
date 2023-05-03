use crate::representation::antibody::Antibody;
use arrayfire::{af_print, le, pow, sum, tile, transpose, Array, DType, Dim4};
use core::f64;
use prost::encoding::bool;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct AntiGen {
    pub id: usize,
    pub class_label: usize,
    pub values: Vec<f64>,
    pub boosting_weight: f64,
}

impl AntiGen {
    pub fn new(id: usize, class_label: usize, values: Vec<f64>) -> Self {
        Self {
            id,
            class_label,
            values,
            boosting_weight: 1.0,
        }
    }
}

pub struct AntiGenPop {
    pub antigens: Vec<AntiGen>,
    pub full_ag_array: Array<f64>,
    pub labeled_ag_arrays: HashMap<usize, Array<f64>>,
    pub label_count_map: HashMap<usize, usize>,
}

fn get_ag_as_array(antigens: &Vec<AntiGen>) -> Array<f64> {
    let num_antigens = antigens.len() as u64;
    let num_dims = antigens.get(0).unwrap().values.len() as u64;

    let ag_arrays = antigens
        .iter()
        .flat_map(|ag| ag.values.as_slice())
        .cloned()
        .collect::<Vec<_>>();

    return transpose(
        &Array::<f64>::new(
            ag_arrays.as_slice(),
            Dim4::new(&[num_dims, num_antigens, 1, 1]),
        ),
        false,
    );
}

fn get_ag_ref_as_array(antigens: &Vec<&AntiGen>) -> Array<f64> {
    // fuckit
    let num_antigens = antigens.len() as u64;
    let num_dims = antigens.get(0).unwrap().values.len() as u64;

    let ag_arrays = antigens
        .iter()
        .flat_map(|ag| ag.values.as_slice())
        .cloned()
        .collect::<Vec<_>>();

    return transpose(
        &Array::<f64>::new(
            ag_arrays.as_slice(),
            Dim4::new(&[num_dims, num_antigens, 1, 1]),
        ),
        false,
    );
}
impl AntiGenPop {
    pub fn new(antigens: Vec<AntiGen>) -> AntiGenPop {
        let num_antigens = antigens.len() as u64;
        let num_dims = antigens.get(0).unwrap().values.len() as u64;

        let class_labels = antigens
            .iter()
            .map(|x| x.class_label)
            .collect::<HashSet<_>>();

        let count_map: HashMap<usize, usize> = class_labels
            .iter()
            .map(|x| {
                (
                    x.clone(),
                    antigens
                        .iter()
                        .filter(|ag| ag.class_label == *x)
                        .collect::<Vec<&AntiGen>>()
                        .len(),
                )
            })
            .collect();

        let mut label_array_map = HashMap::new();

        let ag_array = get_ag_as_array(&antigens);

        for label in &class_labels {
            let filtered: Vec<_> = antigens
                .iter()
                .filter(|ag| ag.class_label == *label)
                .collect();
            let label_array = get_ag_ref_as_array(&filtered);
            label_array_map.insert(label.clone(), label_array);
        }

        return AntiGenPop {
            antigens,
            full_ag_array: ag_array,
            labeled_ag_arrays: label_array_map,
            label_count_map: count_map,
        };
    }

    fn get_antibody_ag_values(
        &self,
        antibody: &Antibody,
        label: &Option<usize>,
    ) -> (Array<f64>, Array<f64>, Array<u8>) {
        let vector_vals = antibody.get_as_vector_values();

        let dim_size = match label {
            None => self.antigens.len().clone(),
            Some(lbl) => self.label_count_map.get(&lbl).unwrap().clone(),
        };

        let offset = tile(&vector_vals.offset, Dim4::new(&[dim_size as u64, 1, 1, 1]));
        let multiplier = tile(
            &vector_vals.multiplier,
            Dim4::new(&[dim_size as u64, 1, 1, 1]),
        );
        let exponent = tile(
            &vector_vals.exponent,
            Dim4::new(&[dim_size as u64, 1, 1, 1]),
        );

        return (offset, multiplier, exponent);
    }

    pub fn get_antibody_required_range(
        &self,
        antibody: &Antibody,
        label: &Option<usize>,
    ) -> Vec<f64> {
        let (offset, multiplier, exponent) = self.get_antibody_ag_values(antibody, label);

        let ag_array = match label {
            None => &self.full_ag_array,
            Some(lbl) => &self.labeled_ag_arrays.get(&lbl).unwrap(),
        };

        let ag_offset = (ag_array + &offset);
        let ag_multi = &ag_offset * multiplier;
        let ag_vals = pow(&ag_multi, &exponent, false);

        let range_sums = &sum(&ag_vals, 1);

        let mut buffer = vec![f64::default(); range_sums.elements()];
        range_sums.host(&mut buffer);
        return buffer;
    }

    pub fn get_registered_antigens(&self, antibody: &Antibody, label: &Option<usize>) -> Vec<bool> {
        let (offset, multiplier, exponent) = self.get_antibody_ag_values(antibody, label);

        let ag_array = match label {
            None => &self.full_ag_array,
            Some(lbl) => &self.labeled_ag_arrays.get(&lbl).unwrap(),
        };

        let ag_offset = (ag_array + &offset);
        let ag_multi = &ag_offset * multiplier;
        let ag_vals = pow(&ag_multi, &exponent, false);

        // af_print!("ag vals ", ag_vals);

        let b = le(&sum(&ag_vals, 1), &antibody.radius_constant, true);

        let mut buffer = vec![bool::default(); b.elements()];
        b.host(&mut buffer);
        return buffer;
    }
}
