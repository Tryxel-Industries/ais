use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::net::Shutdown::Read;
use bytes::buf;
use json::JsonValue;
use prost::{DecodeError, Message};
use prost_types::field_descriptor_proto::Type::Bytes;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use crate::datasets::util::normalize_features_ag;
use crate::entities;
use crate::entities::{DatasetEmbeddings, NewsEntryEmbeddings};
use crate::representation::antibody::Antibody;
use crate::representation::antigen::{AntiGen, AntiGenSplitShell};
use crate::representation::news_article_mapper::NewsArticleAntigenTranslator;





pub fn read_kaggle_embeddings(num_to_read: Option<usize>, sentence_limit: Option<usize>, translator: &mut NewsArticleAntigenTranslator, s_type: SemanticType) -> Vec<AntiGenSplitShell>{
    read_embeddings("kaggle", num_to_read, sentence_limit,translator, s_type)
}

pub fn read_politifact_embeddings(num_to_read: Option<usize>, sentence_limit: Option<usize>,translator: &mut NewsArticleAntigenTranslator, s_type: SemanticType) -> Vec<AntiGenSplitShell>{
    read_embeddings("politifact", num_to_read, sentence_limit,translator, s_type)
}

pub fn read_gosipcop_embeddings(num_to_read: Option<usize>, sentence_limit: Option<usize>,translator: &mut NewsArticleAntigenTranslator, s_type: SemanticType) -> Vec<AntiGenSplitShell>{
    read_embeddings("gosipcop", num_to_read, sentence_limit,translator, s_type)
}

pub fn read_buzfeed_embeddings(num_to_read: Option<usize>, sentence_limit: Option<usize>,translator: &mut NewsArticleAntigenTranslator, s_type: SemanticType) -> Vec<AntiGenSplitShell>{
    read_embeddings("buzfeed", num_to_read, sentence_limit,translator, s_type)
}

fn read_embeddings(dir: &str, num_to_read: Option<usize>,sentence_limit: Option<usize>, translator: &mut NewsArticleAntigenTranslator, s_type: SemanticType) -> Vec<AntiGenSplitShell>{

    let postfix = match s_type {
        SemanticType::Full => {""}
        SemanticType::Whiten90 => {".whitened_90"}
        SemanticType::Whiten265 => {".whitened_256"}
    };

    let path = format!("./datasets/fake_news/{}/embeddings_proto.bin{}", dir, postfix);

    let f = File::open(path.clone()).unwrap();
    let b = match std::fs::read(path) {
        Ok(bytes) => {bytes}
        Err(e) => {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                eprintln!("please run again with appropriate permissions.");
                // return;
            }
            panic!("{}", e);
        }
    };

    let embed = match entities::DatasetEmbeddings::decode(b.as_slice())  {
        Ok(r) => {r}
        Err(e) => {
            panic!("{}", e);

        }
    };
    let mut return_vec = Vec::new();


    let mut to_decode = if let Some(n) = num_to_read{
        let mut embedings = embed.news_entries;
        pick_n_fairly(embedings, n)
    } else {
        embed.news_entries
    };

    for entry in to_decode{
        let ags = translator.translate_article(entry, sentence_limit);
        return_vec.push(AntiGenSplitShell::build_from_article(ags));
    }

    return return_vec;


}

fn pick_n_fairly(embeddings: Vec<NewsEntryEmbeddings>, n_to_pick: usize) -> Vec<NewsEntryEmbeddings> {


    let class_labels = embeddings
        .iter()
        .map(|x| x.label.clone())
        .collect::<HashSet<_>>();

    let frac_map: HashMap<String, f64> = class_labels
        .iter()
        .map(|x| {
            (
                x.clone(),
                embeddings
                    .iter()
                    .filter(|ag| ag.label == *x)
                    .collect::<Vec<_>>()
                    .len() as f64
                    / embeddings.len() as f64,
            )
        })
        .collect();
    let count_map: HashMap<String, usize> = class_labels
        .iter()
        .map(|x| {
            (
                x.clone(),
                embeddings
                    .iter()
                    .filter(|ag| ag.label == *x)
                    .collect::<Vec<_>>()
                    .len(),
            )
        })
        .collect();


    println!("proto test -> frac map:  {:?}", frac_map);
    println!("proto test -> count map: {:?}", count_map);

    let mut out_vec: Vec<NewsEntryEmbeddings> = Vec::new();
    for label in class_labels{
        let label_max = count_map.get(&*label).unwrap();
        let label_count = ((n_to_pick as f64 * frac_map.get(&*label).unwrap()).ceil() as usize).min(label_max.clone());
        let filtered: Vec<_> = embeddings.iter().filter(|x1| x1.label == label).cloned().collect();
        if *label_max == label_count{
            out_vec.extend(filtered)
        }else {
            let picked: Vec<NewsEntryEmbeddings> = filtered.choose_multiple(&mut thread_rng(), label_count).cloned().collect();
            out_vec.extend(picked)
        }
    }

    return out_vec;


}

//
//   Fake news
//

pub enum SemanticType{
    Full,
    Whiten90,
    Whiten265,

}
pub fn read_kaggle_semantic() -> Vec<AntiGenSplitShell>{
    read_semantic_dataset("kaggle")
}

pub fn read_politifact_semantic() -> Vec<AntiGenSplitShell>{
    read_semantic_dataset("politifact")
}

pub fn read_gosipcop_semantic() -> Vec<AntiGenSplitShell>{
    read_semantic_dataset("gosipcop")
}
pub fn read_buzfeed_semantic() -> Vec<AntiGenSplitShell>{
    read_semantic_dataset("buzfeed")
}


fn read_json(path: &str) -> JsonValue {
    let mut lines: Vec<String> = Vec::new();

    let f = File::open(path).unwrap();
    let mut reader = BufReader::new(f);
    let mut line = String::new();

    loop {
        let len = reader.read_line(&mut line).unwrap();

        if line.ends_with("\n") {
            line = line.strip_suffix("\n").unwrap().parse().unwrap();
        }

        if line.len() > 0 {
            lines.push(line.clone());
            line.clear();
        } else {
            break;
        }
    }
    let json_str = lines.join("");
    let js_obj = json::parse(&*json_str).unwrap();
    return js_obj;
}


fn read_semantic_dataset(dir: &str) -> Vec<AntiGenSplitShell>{



    let path = format!("./datasets/fake_news/{}/semantic_features_{}.json", dir,dir);

    let mut json_value = read_json(path.as_str());

    json_value.entries().for_each(|(k, v)| println!("{:?}", k));

    let mut news_entries: Vec<AntiGen> = Vec::new();

    loop {
        let mut val = json_value.pop();
        if val.is_null() {
            break;
        } else {
            let id = val.remove("id").as_i32().unwrap();
            let title: String = val.remove("title").to_string();
            let pub_date: String = val.remove("publishDate").dump();
            let label: String = val.remove("label").to_string();
            let mut res_map = val.remove("resultMap");

            let mut feature_values: Vec<f64> = Vec::new();
            res_map.entries_mut().for_each(|(k, v)| {
                let f_val = v.remove("featureValue").as_f64().unwrap_or(0.0);
                feature_values.push(f_val);
            });



            let ag = AntiGen::new(id as usize, label.parse().unwrap(), feature_values);

            news_entries.push(ag)
        }
    }

    news_entries = normalize_features_ag(news_entries);
    return AntiGenSplitShell::build_from_entry_list(news_entries);
}