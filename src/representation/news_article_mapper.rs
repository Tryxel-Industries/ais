use std::collections::HashSet;
use std::vec;
use rayon::prelude::*;

use crate::entities::NewsEntryEmbeddings;
use crate::representation::antibody::Antibody;
use crate::representation::antigen::AntiGen;


struct SentenceKey{
    global_id: usize,
    local_id: usize,
}

struct NewsArticleKey{
    pub article_id: usize,
    pub article_label: usize,
    pub sentence_id_list: Vec<usize>,
}

//TODO: implement a hash check of the embeddings to avoid running multiple simmelar ones


pub struct NewsArticleAntigenTranslator{
    news_article_keys: Vec<NewsArticleKey>,
    id_counter: usize,
}

impl NewsArticleAntigenTranslator {
    pub fn new() -> NewsArticleAntigenTranslator{
       return NewsArticleAntigenTranslator{
           news_article_keys: Vec::new(),
           id_counter: 0,
       } 
    }
    
    fn get_next_local_id(&mut self) -> usize{
        let next_id = self.id_counter.clone();
        self.id_counter += 1;
        return next_id;
    }
    
    fn embedding_to_antigen(&self, embedding_id: usize, embedding_label:usize, embedding: Vec<f64>) -> AntiGen{
        return AntiGen::new(embedding_id, embedding_label, embedding)
    }
    pub fn translate_article(&mut self, article: NewsEntryEmbeddings) -> Vec<AntiGen>{
        let mut antigens = Vec::new();
        let mut article_ids = Vec::new();
        
        let id = article.id as usize;
        let label = article.label.parse().unwrap();
        let pub_date = article.publish_date;
        
        for embedding in article.embeddings {
            let new_id = self.get_next_local_id();
            article_ids.push(new_id);
                
            let ag = self.embedding_to_antigen(new_id,label,embedding.embedding_value);
            antigens.push(ag);
        }
        
        let key = NewsArticleKey{
            article_id: id,
            article_label: label,
            sentence_id_list: article_ids,
        };
        
        self.news_article_keys.push(key);
        
        return antigens
    }

    pub fn get_show_ag_acc(&self, mut pred_res: Vec<(Option<bool>, &AntiGen)>){
        pred_res.sort_by_key(|(_, ag)| ag.id);



        let (true_positive, false_positive,true_negative, false_negative, nodetect_positive, nodetect_negative) = self.news_article_keys
            .par_iter()
            .map(|article_key| {

            let mut true_sentences =0;
            let mut false_sentences =0;
            for n in &article_key.sentence_id_list{
                let idx = pred_res.binary_search_by_key(&n,|(_, ag)| &ag.id).unwrap();
                let (pred_class, ag) = pred_res.get(idx).unwrap();
                if let Some(c) = pred_class{
                    // if we have a predicted class
                    if *c {
                        true_sentences += 1;
                    } else {
                        false_sentences += 1;
                    }
                }
            }

            let mut true_positives = 0;
            let mut false_positives = 0;
                let mut nodetect_positives = 0;

            let mut true_negatives = 0;
            let mut false_negatives = 0;
                let mut nodetect_negatives = 0;

            if true_sentences > false_sentences{
                if article_key.article_label == 0{
                    true_positives += 1;
                } else {
                    true_negatives += 1;
                }
            } else if false_sentences > true_sentences {
                if article_key.article_label == 0{
                    false_negatives += 1;
                } else {
                    false_positives += 1;
                }
            } else {
                if article_key.article_label == 0{
                    nodetect_positives += 1
                } else {
                    nodetect_negatives += 1
                }
            }
            return (true_positives, false_positives,true_negatives, false_negatives , nodetect_positives, nodetect_negatives)
    }).reduce(|| (0,0,0,0,0,0), |a,b| (a.0+b.0, a.1+b.1,a.2+b.2, a.3+b.3, a.4+b.4, a.5+b.5));


       /* let mut num_cor = 0;
        let mut num_false = 0;
        let mut num_no_detect = 0;

        for article_key in &self.news_article_keys{
            let mut true_sentences = 0;
            let mut false_sentences = 0;
            for n in &article_key.sentence_id_list{
                let idx = pred_res.binary_search_by_key(&n,|(_, ag)| &ag.id).unwrap();
                let (pred_class, ag) = pred_res.get(idx).unwrap();
                if let Some(c) = pred_class{
                    // if we have a predicted class
                    if *c {
                        true_sentences += 1;
                    } else {
                        false_sentences += 1;
                    }
                }
            }
            if true_sentences > false_sentences{
                num_cor += 1
            } else if false_sentences > true_sentences {
                num_false += 1
            } else {
                num_no_detect += 1
            }
        }*/
        let num_cor = true_positive+true_negative;
        let num_false = false_positive + false_negative;
        let num_no_detect = nodetect_positive +nodetect_negative;
        println!("| num cor {:?}, num false {:?}, num no detect {:?}, accuracy: {:.4}%\n| true/false/nodetect positive: {:?}/{:?}/{:?} ,true/false/nodetect negative: {:?}/{:?}/{:?}", num_cor, num_false, num_no_detect, (num_cor as f64/(num_cor+num_false+num_no_detect) as f64), true_positive, false_positive, nodetect_positive, true_negative, false_negative, nodetect_negative)

    }
    
}