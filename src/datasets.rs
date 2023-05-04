use crate::datasets::fake_news_datasets::{read_fnn_embeddings, read_kaggle_embeddings};
use crate::datasets::reference_datasets::{read_diabetes, read_glass, read_ionosphere, read_iris, read_iris_snipped, read_pima_diabetes, read_sonar, read_spirals, read_wine};
use crate::representation::antigen::{AntiGen, AntiGenSplitShell};
use crate::representation::news_article_mapper::NewsArticleAntigenTranslator;

mod reference_datasets;
mod util;
mod fake_news_datasets;


const REF_DATASET_DIR: &str = "./datasets/reference";
const FAKE_NEWS_DATASET_DIR: &str = "./datasets/fake_news";


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
        Datasets::EmbeddingFnn => {read_fnn_embeddings(num_to_read, sentence_limit, translator,use_whitened)},
    }

}

pub fn get_dataset_optimal_params(){}