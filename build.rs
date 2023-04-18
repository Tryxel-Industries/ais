extern crate prost_build;
use std::io::Result;

fn main() -> Result<()> {
    prost_build::compile_protos(&["./src/proto/news_dataset_embeddings.proto"], &["src/"])?;
    Ok(())
}