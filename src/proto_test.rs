use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::net::Shutdown::Read;
use bytes::buf;
use prost::{DecodeError, Message};
use prost_types::field_descriptor_proto::Type::Bytes;
use crate::entities;
use crate::entities::DatasetEmbeddings;
use crate::representation::antibody::Antibody;
use crate::representation::antigen::AntiGen;
use crate::representation::news_article_mapper::NewsArticleAntigenTranslator;

pub fn read_kaggle_embeddings() -> Vec<AntiGen>{

    let path= "./datasets/fake_news/kaggle/embeddings_proto.bin";
    let f = File::open(path).unwrap();
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

    let mut translator = NewsArticleAntigenTranslator::new();
    for entry in embed.news_entries{
        let ags = translator.translate_article(entry);
        return_vec.extend(ags);
    }

    return return_vec;
    // println!("{:?}", return_vec.len());
    // return_vec.extend(embed.news_entries.into_iter().map(|v| AntiGen{
    //     id: v.id as usize,
    //     class_label: v.label.parse().unwrap(),
    //     values: v.embeddings.into_iter().map(|e| e.embedding_value),
    // }))
    //
    // println!("{:?}", embed)



    return return_vec;



    // prost_build::compile_protos(&["src/items.proto"], &["src/"])?;
    // let proto_in = NewsEntryEmbeddings::;
    //
    // let mut ret_vec: Vec<Vec<String>> = Vec::new();
    //
    // let f = File::open("./datasets/fake_news/kaggle/embeddings_proto.bin").unwrap();
    // let mut reader = CodedInputStream::new(Read::new(f));
    //
    // let mut line = String::new();
    //
    // loop {
    //     let len = reader.read_line(&mut line).unwrap();
    //
    //     if line.ends_with("\n") {
    //         line = line.strip_suffix("\n").unwrap().parse().unwrap();
    //     }
    //     if line.ends_with("\r") {
    //         line = line.strip_suffix("\r").unwrap().parse().unwrap();
    //     }
    //
    //     if line.len() > 0 {
    //         let cols = line.split(",");
    //         ret_vec.push(cols.into_iter().map(|s| String::from(s)).collect());
    //         line.clear();
    //     } else {
    //         break;
    //     }
    // }



}