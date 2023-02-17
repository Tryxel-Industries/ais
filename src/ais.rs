
use crate::evaluation::{evaluate_b_cell, score_b_cells, Evaluation};
use crate::representation::{AntiGen, BCell, BCellFactory, DimValueType};
use rayon::prelude::*;
use std::collections::HashSet;
use crate::bucket_empire::BucketKing;
use crate::selection::elitism_selection;

pub struct ParamObj {
    pub b_cell_pop_size: usize,
    pub generations: usize,

    pub mutate_offset: bool,
    pub offset_max_delta: f64,
    pub offset_flip_prob: f64,

    pub mutate_multiplier: bool,
    pub multiplier_max_delta: f64,
    pub multiplier_flip_prob: f64,

    pub mutate_value_type: bool,
}
pub struct ArtificialImmuneSystem {
    pub b_cells: Vec<BCell>,
}

//
//  AIS
//

fn evaluate_population(
    bk: &BucketKing<AntiGen>,
    params: &ParamObj,
    population: Vec<BCell>,
    antigens: &Vec<AntiGen>,
) -> Vec<(Evaluation, BCell)> {
    return population
        .into_par_iter()
        .map(|b_cell| {
            // evaluate b_cells
            let score = evaluate_b_cell(bk,params, antigens, &b_cell);
            return (score, b_cell);
        })
        .collect();
}

enum BCellStates {
    New(BCell),
    Evaluated(Evaluation, BCell),
    Scored(f64, Evaluation, BCell),
}
impl ArtificialImmuneSystem {
    pub fn new() -> ArtificialImmuneSystem {
        return Self {
            b_cells: Vec::new(),
        };
    }

    pub fn train(
        &mut self,
        antigens: &Vec<AntiGen>,
        params: &ParamObj,
        mutate_fn: fn(&ParamObj, f64, BCell) -> BCell,
        selector_fn: fn(&ParamObj, Vec<(f64, Evaluation, BCell)>) -> Vec<(f64, Evaluation, BCell)>,
    ) -> Vec<f64> {
        let mut train_hist = Vec::new();
        let n_dims = antigens.get(0).unwrap().values.len();
        let class_labels = antigens
            .iter()
            .map(|x| x.class_label)
            .collect::<HashSet<_>>();

        let mut bk: BucketKing<AntiGen> = BucketKing::new(
            n_dims,
            (0.0,10.0),
            10,
            |ag| ag.id,
            |ag| &ag.values,
        );

        bk.add_values_to_index(antigens);

        let cell_factory = BCellFactory::new(
            n_dims,
            vec![0.5..=2.0; n_dims],
            vec![0.0..=10.0; n_dims],
            3.0..=20.0,
            vec![DimValueType::Circle, DimValueType::Disabled],
            true,
            true,
            Vec::from_iter(class_labels.into_iter()),
        );

        // gen inital pop
        let mut initial_population: Vec<BCell> = Vec::with_capacity(params.b_cell_pop_size);
        let mut evaluated_pop: Vec<(Evaluation, BCell)> =
            Vec::with_capacity(params.b_cell_pop_size);
        let mut scored_pop: Vec<(f64, Evaluation, BCell)> =
            Vec::with_capacity(params.b_cell_pop_size);

        antigens
            .iter()
            .for_each(|ag| initial_population.push(cell_factory.generate_from_antigen(ag)));
        if params.b_cell_pop_size > initial_population.len() {
            let missing = params.b_cell_pop_size - initial_population.len();
            (0..missing)
                .for_each(|_| initial_population.push(cell_factory.generate_random_genome()));
        }

        evaluated_pop = evaluate_population(&bk,params, initial_population, antigens);
        scored_pop = score_b_cells(evaluated_pop);

        for i in 0..params.generations {
            let max_score = scored_pop
                .iter()
                .map(|(score, _, _)| score)
                .max_by(|a, b| a.total_cmp(b))
                .unwrap();
            train_hist.push(
                scored_pop.iter().map(|(a, _b, _)| a).sum::<f64>() / scored_pop.len() as f64,
            );
            println!(
                "iter: {:<5} avg score {:.6}, max score {:.6}",
                i,
                scored_pop.iter().map(|(a, _b, _)| a).sum::<f64>() / scored_pop.len() as f64,
                max_score
            );
            // println!("pop size {:} ",scored_pop.len());

            // find the matches for the new B-cells
            //

            // let new_scored: Vec<(f64, BCell)> = scored_pop.clone().into_par_iter().map(|(score ,b_cell)| {
            let new_evaluated: Vec<(Evaluation, BCell)> = scored_pop
                .clone()
                .into_par_iter()
                .map(|(score, _evaluation, b_cell)| {
                    let frac_score = score / max_score;
                    let mutated = mutate_fn(params, frac_score, b_cell);
                    let evaluation = evaluate_b_cell(&bk,params, antigens, &mutated);
                    return (evaluation, mutated);
                })
                .collect();
            evaluated_pop = scored_pop.into_iter().map(|(_a, b, c)| (b, c)).collect();
            evaluated_pop.extend(new_evaluated);

            // let new_scored = score_population(params, mutated, antigens, eval_fn);

            // tweak by overmatching factor
            scored_pop = score_b_cells(evaluated_pop);

            // select a subset of the pop
            scored_pop = selector_fn(params, scored_pop);

            // gen new b_cells to fill the hole
            if params.b_cell_pop_size > scored_pop.len() {
                let missing = params.b_cell_pop_size - scored_pop.len();
                let new_pop = (0..missing)
                    .map(|_| cell_factory.generate_random_genome())
                    .map(|cell| (evaluate_b_cell(&bk,params, antigens, &cell), cell))
                    .collect::<Vec<(Evaluation, BCell)>>();

                evaluated_pop = scored_pop.into_iter().map(|(_a, b, c)| (b, c)).collect();
                evaluated_pop.extend(new_pop);
                scored_pop = score_b_cells(evaluated_pop);
            }
        }
        let (scored_pop, _drained) = elitism_selection(scored_pop,&150);
        self.b_cells = scored_pop
            .into_iter()
            .map(|(_score, _ev, cell)| cell)
            .collect();
        return train_hist;
    }

    pub fn is_class_correct(&self, antigen: &AntiGen) -> Option<bool> {
        let matching_cells = self
            .b_cells
            .iter()
            .filter(|b_cell| b_cell.test_antigen(antigen))
            .collect::<Vec<_>>();

        if matching_cells.len() == 0 {
            return None;
        }

        let class_true = matching_cells
            .iter()
            .filter(|x| x.class_label == antigen.class_label)
            .collect::<Vec<_>>();
        let class_false = matching_cells
            .iter()
            .filter(|x| x.class_label != antigen.class_label)
            .collect::<Vec<_>>();

        if class_true.len() > class_false.len() {
            return Some(true);
        } else {
            return Some(false);
        }
    }
}
