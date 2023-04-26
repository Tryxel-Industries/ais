use std::vec;
use crate::entities::NewsEntryEmbeddings;
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
    
}