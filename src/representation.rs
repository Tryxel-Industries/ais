use rayon::prelude::*;

use crate::evaluation::evaluate_antibody;
use crate::representation::antibody::{Antibody, DimValueType};
use crate::representation::antigen::AntiGen;
use crate::BucketKing;
use crate::params::Params;
use crate::representation::evaluated_antibody::EvaluatedAntibody;

pub mod antibody;
pub mod antibody_factory;
pub mod antigen;
pub mod news_article_mapper;
pub mod evaluated_antibody;

pub fn expand_antibody_radius_until_hit(
    mut cell: Antibody,
    antigens: &Vec<AntiGen>,
) -> Antibody {
    /*
    ha en middels høy range
    finn om ikke hit på feil øk til du får det
    iterer deretter gjennom de som er feil og for hver som er innenfor reduser rangen til den er lavere en den
     */

    if cell
        .dim_values
        .iter()
        .map(|v| v.value_type)
        .all(|v| v == DimValueType::Disabled)
    {
        // abort if all dims are disabled
        return cell;
    }

    let min_range = antigens
        .par_iter()
        .map(|ag| cell.get_affinity_dist(ag))
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();


    cell.radius_constant = min_range * 0.99;
    return cell;
}

pub fn expand_antibody_radius_until_hit__(
    mut cell: Antibody,
    bk: &BucketKing<AntiGen>,
    antigens: &Vec<AntiGen>,
) -> Antibody {
    /*
    ha en middels høy range
    finn om ikke hit på feil øk til du får det
    iterer deretter gjennom de som er feil og for hver som er innenfor reduser rangen til den er lavere en den
     */

    if !cell
        .dim_values
        .iter()
        .map(|v| v.value_type)
        .any(|v| v != DimValueType::Disabled)
    {
        // abort if all dims are disabled
        return cell;
    }
    let mut evaluation = loop {
        let evaluation = evaluate_antibody(antigens, &cell);

        if evaluation.wrongly_matched.len() > 0 {
            break evaluation;
        } else {
            cell.radius_constant *= 2.0;
        }
    };
    evaluation.wrongly_matched.sort();

    let min_range = antigens
        .par_iter()
        .filter(|ag| evaluation.wrongly_matched.binary_search(&ag.id).is_ok())
        .map(|ag| cell.get_affinity_dist(ag))
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    cell.radius_constant = min_range * 0.99;

    // todo: inneficeient
    // let wrong_ags : Vec<_>= antigens.iter().filter(|ag| evaluation.wrongly_matched.binary_search(&ag.id).is_ok()).collect();
    //
    // for err_ag in wrong_ags{
    //
    //     let affinity_dist = cell.get_affinity_dist(err_ag);
    //
    //     if affinity_dist < cell.radius_constant{
    //         cell.radius_constant = affinity_dist * 0.99;
    //     }
    // }
    //

    return cell;
}



pub fn evaluate_population(
    params: &Params,
    population: Vec<Antibody>,
    antigens: &Vec<AntiGen>,
) -> Vec<EvaluatedAntibody> {
    return population
        .into_par_iter() // TODO: set parallel
        // .into_iter()
        .map(|antibody| {
            // evaluate antibodies
            let eval  = evaluate_antibody(antigens, &antibody);
            return EvaluatedAntibody{
                evaluation: eval,
                antibody,
            };
        })
        .collect();
}

