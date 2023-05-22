use crate::ais::ArtificialImmuneSystem;
use crate::params::Params;
use crate::prediction::EvaluationMethod;
use crate::representation::antibody::Antibody;
use crate::representation::antigen::AntiGen;
use crate::representation::news_article_mapper::NewsArticleAntigenTranslator;
use std::collections::HashMap;

//
//  Antibody info
//
pub fn show_ab_dim_multipliers(antibody: &Antibody) {
    println!(
        "genome dim values    {:?}",
        antibody
            .dim_values
            .iter()
            .map(|v| v.multiplier)
            .collect::<Vec<_>>()
    );
}

pub fn show_ab_dim_offsets(antibody: &Antibody) {
    println!(
        "genome offset values {:?}",
        antibody
            .dim_values
            .iter()
            .map(|v| v.offset)
            .collect::<Vec<_>>()
    );
}
pub fn show_ab_dim_value_types(antibody: &Antibody) {
    println!(
        "genome value type    {:?}",
        antibody
            .dim_values
            .iter()
            .map(|v| &v.value_type)
            .collect::<Vec<_>>()
    );
}

//
//  Pop testing
//

struct EvalCounter {
    true_count: usize,
    false_count: usize,
    no_reg_count: usize,
}

impl EvalCounter {
    fn new() -> EvalCounter {
        EvalCounter {
            true_count: 0,
            false_count: 0,
            no_reg_count: 0,
        }
    }
}

pub fn eval_display(
    eval_ag_pop: &Vec<AntiGen>,
    ais: &ArtificialImmuneSystem,
    translator: &NewsArticleAntigenTranslator,
    display_headline: String,
    verbose: bool,
    return_eval_type: Option<&EvaluationMethod>,
) -> f64 {
    let eval_types = vec![
        EvaluationMethod::Count,
        EvaluationMethod::Fraction,
        EvaluationMethod::AffinitySum,
    ];

    let mut translator_vals = Vec::new();
    let mut eval_vals = Vec::new();

    for eval_method in &eval_types {
        let mut eval_translator_vals = Vec::new();
        let mut eval_count = EvalCounter::new();
        for antigen in eval_ag_pop {
            let pred = ais.is_class_correct(antigen, &eval_method);
            eval_translator_vals.push(pred.clone());
            if let Some(corr_pred) = pred {
                if corr_pred {
                    eval_count.true_count += 1;
                } else {
                    eval_count.false_count += 1;
                }
            } else {
                eval_count.no_reg_count += 1;
            }
        }

        translator_vals.push(eval_translator_vals);
        eval_vals.push(eval_count);
    }

    if verbose {
        println!();
        println!("==========================");
        println!("      MUT info");
        println!("==========================");
        ais.print_ab_mut_info();

        println!("=============================================================================");
        println!("      {:}", display_headline);
        println!("=============================================================================");
        println!("dataset size {:?}", eval_ag_pop.len());
    } else {
        println!("## {:}", display_headline);
    }

    let mut return_eval = 0.0;

    for ((eval_method, evals), eval_counts) in eval_types.iter().zip(translator_vals).zip(eval_vals)
    {
        let translator_formatted = evals.into_iter().zip(eval_ag_pop).collect();

        let acc = eval_counts.true_count as f64 / (eval_ag_pop.len() as f64);
        let precession = eval_counts.true_count as f64
            / (eval_counts.false_count as f64 + eval_counts.true_count as f64).max(1.0);

        if let Some(eval_type) = return_eval_type {
            if eval_type == eval_method {
                return_eval = acc;
            }
        } else {
            if acc > return_eval {
                return_eval = acc;
            }
        }

        println!(
            "{:<11}: corr {:>2?}, false {:>3?}, no_detect {:>3?}, presission: {:>2.3?}, frac: {:2.3?}",
            eval_method.to_string(),eval_counts.true_count, eval_counts.false_count, eval_counts.no_reg_count,precession, acc
        );
        translator.get_show_ag_acc(translator_formatted, true);
        if verbose {
            println!()
        }
    }

    println!();
    return return_eval;

    /*           println!(
               "Total runtime: {:?}, \nPer iteration: {:?}",
               duration,
               duration.as_nanos() / params.generations as u128
           );

    */
}
