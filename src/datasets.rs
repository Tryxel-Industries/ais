use crate::datasets::fake_news_datasets::{read_fnn_embeddings, read_kaggle_embeddings};
use crate::datasets::reference_datasets::{read_diabetes, read_glass, read_ionosphere, read_iris, read_iris_snipped, read_pima_diabetes, read_sonar, read_spirals, read_wine};
use crate::representation::antigen::AntiGen;
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

pub fn get_dataset(dataset: Datasets, num_to_read: Option<usize>, translator: &mut NewsArticleAntigenTranslator, use_whitened: bool) -> Vec<AntiGen>{
    match dataset {
        Datasets::Iris => {read_iris()}
        Datasets::IrisSnipped => {read_iris_snipped()}
        Datasets::Wine => {read_wine()}
        Datasets::Diabetes => {read_diabetes()}
        Datasets::Spirals => {read_spirals()}
        Datasets::PrimaDiabetes => {read_pima_diabetes()}
        Datasets::Sonar => {read_sonar()}
        Datasets::Glass => {read_glass()}
        Datasets::Ionosphere => {read_ionosphere()}

        // embedding sets
        Datasets::EmbeddingKaggle => {read_kaggle_embeddings(num_to_read, translator,use_whitened)},
        Datasets::EmbeddingFnn => {read_fnn_embeddings(num_to_read, translator,use_whitened)},
    }

}

pub fn get_dataset_optimal_params(){}