use rayon::prelude::*;
use statrs::statistics::Statistics;

use crate::evaluation::evaluate_antibody;
use crate::representation::antibody::{Antibody, DimValueType};
use crate::representation::antigen::{AntiGen, AntiGenPop};
use crate::BucketKing;
use crate::params::Params;

pub mod antibody;
pub mod antibody_factory;
pub mod antigen;
pub mod news_article_mapper;

pub fn expand_antibody_radius_until_hit(
    params: &Params,
    mut cell: Antibody,
    antigen_pop: &AntiGenPop,
) -> Antibody {
    /*
    ha en middels høy range
    finn om ikke hit på feil øk til du får det
    iterer deretter gjennom de som er feil og for hver som er innenfor reduser rangen til den er lavere en den
     */

    let min_range = if params.gpu_accelerate{
        let antigens = &antigen_pop.antigens;
        antigen_pop.get_antibody_required_range(&cell, &Some(cell.class_label)).min()
    }else {

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
            let evaluation = evaluate_antibody(params,antigen_pop, &cell);

            if evaluation.wrongly_matched.len() > 0 {
                break evaluation;
            } else {
                cell.radius_constant *= 2.0;
            }
        };
        evaluation.wrongly_matched.sort();

  antigen_pop.antigens
            .par_iter()
            .filter(|ag| evaluation.wrongly_matched.binary_search(&ag.id).is_ok())
            .map(|ag| cell.get_affinity_dist(ag))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    };




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
