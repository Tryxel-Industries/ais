use std::collections::HashSet;
use strum_macros::{Display, EnumString};
use crate::datasets::fake_news_datasets::{read_buzfeed_embeddings, read_fnn_embeddings, read_kaggle_embeddings};
use crate::datasets::reference_datasets::{read_diabetes, read_glass, read_ionosphere, read_iris, read_iris_snipped, read_pima_diabetes, read_sonar, read_spirals, read_wine};
use crate::params::{Params, PopSizeType, ReplaceFractionType};
use crate::prediction::EvaluationMethod;
use crate::representation::antibody::DimValueType;
use crate::representation::antigen::{AntiGen, AntiGenSplitShell};
use crate::representation::news_article_mapper::NewsArticleAntigenTranslator;

mod reference_datasets;
mod util;
mod fake_news_datasets;


const REF_DATASET_DIR: &str = "./datasets/reference";
const FAKE_NEWS_DATASET_DIR: &str = "./datasets/fake_news";

#[derive(Clone, EnumString, Display)]
pub enum Datasets{
    Iris,
    IrisSnipped,
    Wine,
    Diabetes,
    Spirals,
    PrimaDiabetes,
    Sonar,
    Glass,
    Ionosphere,

    EmbeddingKaggle,
    EmbeddingBuzfeed,
    EmbeddingFnn
}

impl Datasets {
    pub fn is_embedding_set(&self)-> bool{
        match self {
            Datasets::EmbeddingKaggle | Datasets::EmbeddingFnn => {true}
            _ => {false}
        }
    }

}



pub fn get_dataset(dataset: Datasets, num_to_read: Option<usize>, sentence_limit: Option<usize>,translator: &mut NewsArticleAntigenTranslator, use_whitened: bool) -> Vec<AntiGenSplitShell>{
    match dataset {
        Datasets::Iris => {AntiGenSplitShell::build_from_entry_list(read_iris())}
        Datasets::IrisSnipped => {AntiGenSplitShell::build_from_entry_list(read_iris_snipped())}
        Datasets::Wine => {AntiGenSplitShell::build_from_entry_list(read_wine())}
        Datasets::Diabetes => {AntiGenSplitShell::build_from_entry_list(read_diabetes())}
        Datasets::Spirals => {AntiGenSplitShell::build_from_entry_list(read_spirals())}
        Datasets::PrimaDiabetes => {AntiGenSplitShell::build_from_entry_list(read_pima_diabetes())}
        Datasets::Sonar => {AntiGenSplitShell::build_from_entry_list(read_sonar())}
        Datasets::Glass => {AntiGenSplitShell::build_from_entry_list(read_glass())}
        Datasets::Ionosphere => {AntiGenSplitShell::build_from_entry_list(read_ionosphere())}

        // embedding sets
        Datasets::EmbeddingKaggle => {read_kaggle_embeddings(num_to_read, sentence_limit, translator,use_whitened)},
        Datasets::EmbeddingBuzfeed => {read_buzfeed_embeddings(num_to_read, sentence_limit, translator,use_whitened)},
        Datasets::EmbeddingFnn => {read_fnn_embeddings(num_to_read, sentence_limit, translator,use_whitened)},
    }

}

pub fn get_dataset_optimal_params(dataset: Datasets, class_labels: HashSet<usize>) -> Params{

    match dataset {
        Datasets::Wine => {get_wine_params(class_labels)}
        Datasets::Sonar => {get_sonar_params(class_labels)}
        _ => {
            panic!("eror")
        }

    }
}


// fn get_wine_params()-> Params{
//     return Params{
//
//     }
// }


fn get_diabetes_params(class_labels: HashSet<usize>)-> Params{
    return Params {
        eval_method: EvaluationMethod::Count,
            boost: 5,
            // -- train params -- //
            // antigen_pop_size: PopSizeType::Fraction(0.7),
            antigen_pop_size: PopSizeType::Number(200),
            generations: 500,

            mutation_offset_weight: 5,
            mutation_multiplier_weight: 5,
            mutation_multiplier_local_search_weight: 1,
            mutation_radius_weight: 5,
            mutation_value_type_weight: 3,

            mutation_label_weight: 0,

            mutation_value_type_local_search_dim: true,

            // -- reduction -- //
            membership_required: 0.75,

            offset_mutation_multiplier_range: -0.5..=0.5,
            multiplier_mutation_multiplier_range: -0.5..=0.5,
            radius_mutation_multiplier_range: -0.5..=0.5,

            value_type_valid_mutations: vec![
                DimValueType::Circle,
                DimValueType::Disabled,
                DimValueType::Open,
            ],

            label_valid_mutations: class_labels.clone().into_iter().collect::<Vec<usize>>(),

            correctness_weight: 1.0,
            coverage_weight: 1.0,
            uniqueness_weight: 0.5,
            good_afin_weight: 0.0,
            bad_afin_weight: 2.0,

            //selection
            leak_fraction: 0.5,
            leak_rand_prob: 0.5,
            // replace_frac_type: ReplaceFractionType::Linear(0.5..0.01),
            // replace_frac_type: ReplaceFractionType::Linear(0.8..0.3),
            replace_frac_type: ReplaceFractionType::MaxRepFrac(0.8),
            tournament_size: 1,
            n_parents_mutations: 40,

            antibody_init_expand_radius: true,

            // -- B-cell from antigen initialization -- //
            antibody_ag_init_multiplier_range: 0.8..=1.2,
            antibody_ag_init_value_types: vec![
                (DimValueType::Circle, 1),
                (DimValueType::Disabled, 1),
                (DimValueType::Open, 1),
            ],
            antibody_ag_init_range_range: 0.1..=0.4,

            // -- B-cell from random initialization -- //
            antibody_rand_init_offset_range: 0.0..=1.0,
            antibody_rand_init_multiplier_range: 0.8..=1.2,
            antibody_rand_init_value_types: vec![
                (DimValueType::Circle, 1),
                (DimValueType::Disabled, 1),
                (DimValueType::Open, 1),
            ],
            antibody_rand_init_range_range: 0.1..=0.4,
    };
}

fn get_wine_params(class_labels: HashSet<usize>)-> Params{
    return Params {
        eval_method: EvaluationMethod::Count,
            boost: 0,
            // -- train params -- //
            // antigen_pop_size: PopSizeType::Fraction(0.6),
            antigen_pop_size: PopSizeType::Number(200),
            generations: 200,

            mutation_offset_weight: 5,
            mutation_multiplier_weight: 5,
            mutation_multiplier_local_search_weight: 1,
            mutation_radius_weight: 5,
            mutation_value_type_weight: 3,

            mutation_label_weight: 0,

            mutation_value_type_local_search_dim: true,

            // -- reduction -- //
            membership_required: 0.75,

            offset_mutation_multiplier_range: -0.5..=0.5,
            multiplier_mutation_multiplier_range: -0.5..=0.5,
            radius_mutation_multiplier_range: -0.5..=0.5,

            value_type_valid_mutations: vec![
                DimValueType::Circle,
                DimValueType::Disabled,
                DimValueType::Open,
            ],

            label_valid_mutations: class_labels.clone().into_iter().collect::<Vec<usize>>(),

            correctness_weight: 1.0,
            coverage_weight: 1.0,
            uniqueness_weight: 0.5,
            good_afin_weight: 0.0,
            bad_afin_weight: 1.0,

            //selection
            leak_fraction: 0.5,
            leak_rand_prob: 0.5,
            replace_frac_type: ReplaceFractionType::Linear(0.8..0.3),
            // replace_frac_type: ReplaceFractionType::MaxRepFrac(0.6),
            tournament_size: 1,
            n_parents_mutations: 40,

            antibody_init_expand_radius: true,

            // -- B-cell from antigen initialization -- //
            antibody_ag_init_multiplier_range: 0.8..=1.2,
            antibody_ag_init_value_types: vec![
                (DimValueType::Circle, 1),
                (DimValueType::Disabled, 1),
                (DimValueType::Open, 1),
            ],
            antibody_ag_init_range_range: 0.1..=0.4,

            // -- B-cell from random initialization -- //
            antibody_rand_init_offset_range: 0.0..=1.0,
            antibody_rand_init_multiplier_range: 0.8..=1.2,
            antibody_rand_init_value_types: vec![
                (DimValueType::Circle, 1),
                (DimValueType::Disabled, 1),
                (DimValueType::Open, 1),
            ],
            antibody_rand_init_range_range: 0.1..=0.4,
        };
}

fn get_sonar_params(class_labels: HashSet<usize>)-> Params{
    return Params{

        eval_method: EvaluationMethod::Count,
        correctness_weight: 0.2,
        coverage_weight: 0.5,
        uniqueness_weight: 0.8,
        good_afin_weight: 0.4,
        bad_afin_weight: 1.0,


        generations: 1000,
        membership_required: 0.0,

        boost: 3,
        // -- train params -- //
        antigen_pop_size: PopSizeType::Fraction(0.8),
        // antigen_pop_size: PopSizeType::Number(200),

        mutation_offset_weight: 5,
        mutation_multiplier_weight: 5,
        mutation_multiplier_local_search_weight: 1,
        mutation_radius_weight: 5,
        mutation_value_type_weight: 3,

        mutation_label_weight: 0,

        mutation_value_type_local_search_dim: true,

        // -- reduction -- //

        offset_mutation_multiplier_range: -0.5..=0.5,
        multiplier_mutation_multiplier_range: -0.5..=0.5,
        radius_mutation_multiplier_range: -0.5..=0.5,

        value_type_valid_mutations: vec![
            DimValueType::Circle,
            DimValueType::Disabled,
            DimValueType::Open,
        ],

        label_valid_mutations: class_labels.clone().into_iter().collect::<Vec<usize>>(),




        //selection
        leak_fraction: 0.45,
        leak_rand_prob: 0.5,
        replace_frac_type: ReplaceFractionType::Linear(0.8..0.3),
        // replace_frac_type: ReplaceFractionType::MaxRepFrac(0.6),
        tournament_size: 1,
        n_parents_mutations: 40,

        antibody_init_expand_radius: true,

        // -- B-cell from antigen initialization -- //
        antibody_ag_init_multiplier_range: 0.8..=1.2,
        antibody_ag_init_value_types: vec![
            (DimValueType::Circle,1),
            (DimValueType::Disabled,1),
            (DimValueType::Open,1),
        ],
        antibody_ag_init_range_range: 0.1..=0.4,

        // -- B-cell from random initialization -- //
        antibody_rand_init_offset_range: 0.0..=1.0,
        antibody_rand_init_multiplier_range: 0.8..=1.2,
        antibody_rand_init_value_types: vec![
            (DimValueType::Circle,1),
            (DimValueType::Disabled,1),
            (DimValueType::Open,1),
        ],
        antibody_rand_init_range_range: 0.1..=0.4,

    }
}
