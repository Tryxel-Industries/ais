
#[derive(Clone, Debug)]
pub struct AntiGen {
    pub id: usize,
    pub class_label: usize,
    pub values: Vec<f64>,
    pub boosting_weight: f64,
}

/// Used to split train/test data by article
#[derive(Clone, Debug)]
pub struct AntiGenSplitShell {
    pub class_label: usize,
    sub_antigens: Vec<AntiGen>,
}

impl AntiGenSplitShell {
    pub fn upack(self) -> Vec<AntiGen>{
        return self.sub_antigens
    }

    pub fn build_from_article(sentences: Vec<AntiGen>) -> AntiGenSplitShell{
        return AntiGenSplitShell{
            class_label: sentences.first().unwrap().class_label.clone(),
            sub_antigens: sentences,
        }
    }


    pub fn build_from_entry(ag: AntiGen) -> AntiGenSplitShell{
        return AntiGenSplitShell{
            class_label: ag.class_label.clone(),
            sub_antigens: vec![ag],
        }
    }

    pub fn build_from_entry_list(entry_list:  Vec<AntiGen>) -> Vec<AntiGenSplitShell>{
        return entry_list.into_iter().map(|ag| AntiGenSplitShell::build_from_entry(ag)).collect();
    }

}

impl AntiGen {
    pub fn new(id: usize, class_label: usize, values: Vec<f64>) -> Self {
        Self {
            id,
            class_label,
            values,
            boosting_weight: 1.0,
        }
    }
}
