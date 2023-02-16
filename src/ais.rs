use crate::representation::{AntiGen, BCell, BCellFactory, DimValueType};
use std::collections::HashSet;
use rayon::prelude::*;

pub struct ParamObj {
    pub b_cell_pop_size: usize,
    pub generations: usize,
}
pub struct ArtificialImmuneSystem {
    b_cells: Vec<BCell>,
}

//
//  AIS
//


fn score_population(params: &ParamObj, population: Vec<BCell>, antigens: &Vec<AntiGen>, eval_fn: fn(&ParamObj, &Vec<AntiGen>, &BCell) -> f64) -> Vec<(f64,BCell)>{
    return population.into_par_iter().map(|b_cell| {
        // evaluate b_cells
        let score = eval_fn(params,antigens, &b_cell);
        return (score, b_cell)
    }).collect();
}

impl ArtificialImmuneSystem {
    pub fn new() -> ArtificialImmuneSystem{
        return Self{
            b_cells: Vec::new()
        };
    }

    pub fn train(
        &mut self,
        antigens: &Vec<AntiGen>,
        params: &ParamObj,
        eval_fn: fn(&ParamObj, &Vec<AntiGen>, &BCell) -> f64,
        mutate_fn: fn(&ParamObj, f64, BCell) -> BCell,
        selector_fn: fn(&ParamObj, Vec<(f64, BCell)>) -> Vec<(f64, BCell)>,
    ) {
        let n_dims = antigens.get(0).unwrap().values.len();
        let class_labels = antigens
            .iter()
            .map(|x| x.class_label)
            .collect::<HashSet<_>>();

        let cell_factory = BCellFactory::new(
            n_dims,
            vec![-2.0..=2.0; n_dims],
            vec![-10.0..=10.0; n_dims],
            3.0..=20.0,
            vec![DimValueType::Circle],
            true,
            true,
            Vec::from_iter(class_labels.into_iter()),
        );

        // gen inital pop
        let mut initial_population: Vec<BCell> = Vec::with_capacity(params.b_cell_pop_size);
        let mut scored_pop: Vec<(f64, BCell)> = Vec::with_capacity(params.b_cell_pop_size);
        antigens
            .iter()
            .for_each(|ag| initial_population.push(cell_factory.generate_from_antigen(ag)));

        if params.b_cell_pop_size < initial_population.len() {
            let missing = params.b_cell_pop_size - initial_population.len();
            (0..missing).for_each(|_| initial_population.push(cell_factory.generate_random_genome()));
        }

        scored_pop = score_population(params, initial_population, antigens, eval_fn);

        for i in 0..params.generations {
            let max_score = scored_pop.iter().map(|(score, _)| score).max_by(|a, b| a.total_cmp(b)).unwrap();

            println!("avg score {:?}", max_score);
            // println!("on generation {} ",i);
            let mutated: Vec<BCell> = scored_pop.clone().into_par_iter().map(|(score ,b_cell)| {
                let frac_score = score/max_score;
                let mutated = mutate_fn(params,frac_score, b_cell);

                return mutated
            }).collect();


            let new_scored = score_population(params, mutated, antigens, eval_fn);

            scored_pop.extend(new_scored.into_iter());

            // println!("avg score {:?}", scored_pop.iter().map(|(a,b)| a).sum::<f64>()/scored_pop.len()as f64);
            // select a subset of the pop
            scored_pop = selector_fn(params,scored_pop);
            // gen new b_cells to fill the hole
            if params.b_cell_pop_size > scored_pop.len() {
                let missing = params.b_cell_pop_size - scored_pop.len();
                let new_pop = (0..missing).map(|_| cell_factory.generate_random_genome()).collect::<Vec<BCell>>();
                scored_pop.extend(score_population(params, new_pop, antigens, eval_fn).into_iter());

            }

        }



        self.b_cells = scored_pop.into_iter().map(|(_,x)| x).collect();
    }

    pub fn pred_class(&self, antigen: &AntiGen) -> usize {

        let matching_cells =  self.b_cells.iter().filter(|b_cell| b_cell.test_antigen(antigen)).collect::<Vec<_>>();
        let class_true = matching_cells.iter().filter(|x| x.class_label==antigen.class_label).collect::<Vec<_>>();
        let class_false = matching_cells.iter().filter(|x| x.class_label!=antigen.class_label).collect::<Vec<_>>();

        if class_true.len()>class_false.len(){
            return 1
        }else {
            return 0
        }
    }
}