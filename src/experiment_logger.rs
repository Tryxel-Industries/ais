use std::collections::HashMap;
use std::hash::Hash;


use std::fs::File;
use std::io::{BufWriter, Write};
use crate::datasets::Datasets;

use strum_macros::{Display, EnumString};

use serde::{Serialize, Serializer};
use serde_json::{json, Value};
use statrs::statistics::Statistics;
use crate::params::Params;
use crate::representation::antibody::DimValueType;

#[derive(Clone, Copy, PartialEq, Debug, Display, EnumString)]
enum FillMode{
    Ffill,
    Nan,
}


#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash, Serialize, EnumString, Display)]
pub enum ExperimentProperty{
    // per iter logging
    TestAccuracy,
    TrainAccuracy,
    AvgTrainScore,
    PopLabelMemberships,
    PopDimTypeMemberships,

    ScoreComponents,

    BoostAccuracy,
    BoostAccuracyTest,
    Runtime,


    FoldAccuracy,
}





impl ExperimentProperty {
    pub fn is_step_prop(&self) -> bool{
        return match self {
            ExperimentProperty::TestAccuracy => {true}
            ExperimentProperty::TrainAccuracy => {true}
            ExperimentProperty::AvgTrainScore => {true}
            ExperimentProperty::PopLabelMemberships => {true}
            ExperimentProperty::PopDimTypeMemberships => {true}
            ExperimentProperty::ScoreComponents => {true}
            ExperimentProperty::BoostAccuracy => {false}
            ExperimentProperty::BoostAccuracyTest => {false}
            ExperimentProperty::Runtime => {false}
            ExperimentProperty::FoldAccuracy => {false}
        }

    }
}



#[derive(Clone, PartialEq, Debug, Serialize, EnumString, Display)]
pub enum LoggedValue{
    MappedFloats(HashMap<String, f64>),
    MappedInts(HashMap<String, usize>),
    SingleValue(f64),
    LabelMembership(HashMap<usize,usize>),
    DimTypeMembership(HashMap<DimValueType,f64>),
    NoValue
}

impl LoggedValue {
    pub fn gen_corr_wrong_no_reg(cor: usize, wrong: usize, no_reg: usize) -> LoggedValue{
        let map = HashMap::from([("cor".to_string(), cor),("wrong".to_string(), wrong),("no_reg".to_string(), no_reg)]);
        return LoggedValue::MappedInts(map)
    }
    pub fn gen_train_test(train: f64, test: f64) -> LoggedValue{
        let map = HashMap::from([("train".to_string(), train),("test".to_string(), test)]);
        return LoggedValue::MappedFloats(map)
    }

}

#[derive(Clone, Debug, Serialize)]
struct TrackedProperty{
    property_type: ExperimentProperty,
    prop_values: Vec<LoggedValue>,

    #[serde(skip_serializing)]
    fill_method: FillMode
}

impl TrackedProperty {
    fn new(property_type: ExperimentProperty) -> TrackedProperty {
        return TrackedProperty {
            property_type,
            prop_values: Vec::new(),
            fill_method: FillMode::Ffill,
        }
    }

    pub fn get_current_step(&self) -> usize {
        return self.prop_values.len();
    }

    pub fn set_fill_mode(&mut self, fill_mode: FillMode) {
        self.fill_method = fill_mode
    }


    fn step(&mut self) {
        let value = if let Some(v) = self.prop_values.get(self.prop_values.len()) {
            v.clone()
        } else {
            LoggedValue::NoValue
        };
        self.prop_values.push(value);
    }

    fn add_val(&mut self, value: LoggedValue) {
        self.prop_values.push(value)
    }

}

pub struct ExperimentLogger{
    dataset_used: Datasets,

    step_properties: Vec<ExperimentProperty>,
    meta_properties: Vec<ExperimentProperty>,

    full_properties: Vec<ExperimentProperty>,

    meta_properties_map: HashMap<ExperimentProperty, TrackedProperty>,

    step_properties_map: Option<HashMap<ExperimentProperty, TrackedProperty>>,
    map_hist: Vec<HashMap<ExperimentProperty, TrackedProperty>>,

    current_step: usize,
    current_fold: usize,
    current_run: usize,

    run_every_n_step: usize,
    params: Option<Params>,

}
fn build_trackers(active_properties: &Vec<ExperimentProperty>) -> HashMap<ExperimentProperty, TrackedProperty>{
        let properties_map: HashMap<_,_> = active_properties.iter()
            .map(|prop_type| (prop_type.clone(),TrackedProperty::new(prop_type.clone())))
            .collect();
    return properties_map;
}

impl  ExperimentLogger {
    pub fn new(dataset_used: Datasets, tracked_props: Vec<ExperimentProperty>, run_every_n_step: usize) -> ExperimentLogger {

        let step_properties = tracked_props.clone().into_iter().filter(|prop| prop.is_step_prop()).collect();
        let meta_properties = tracked_props.clone().into_iter().filter(|prop| !prop.is_step_prop()).collect();

        let meta_properties_map = build_trackers(&meta_properties);

        return ExperimentLogger{
            dataset_used,
            step_properties,
            meta_properties,
            full_properties: tracked_props,
            meta_properties_map,
            step_properties_map: None,
            map_hist: Vec::new(),
            current_step: 0,
            current_fold: 0,
            current_run: 0,
            run_every_n_step,
            params: None,
        }
    }



    pub fn should_run(&self, experiment_property: ExperimentProperty) -> bool{
        if experiment_property.is_step_prop(){
            if self.current_step % self.run_every_n_step == 0{
                self.full_properties.contains(&experiment_property)
            }else {
                false
            }
        }else {
            self.full_properties.contains(&experiment_property)
        }
    }

    pub fn iter_step(&mut self){
        self.current_step += 1;
        if let Some(prop_map) = &mut self.step_properties_map{
            for (_, prop) in prop_map.iter_mut(){
                if prop.get_current_step() < self.current_step{
                    prop.step()
                }
            }
        } else { panic!("not initialized") }

    }

    pub fn end_train(&mut self){
        if let Some(old_map) = &self.step_properties_map{
            if old_map.len() > 0{
                self.map_hist.push(old_map.clone())
            }
        }
        self.current_fold += 1;
        self.step_properties_map = None;
    }

    pub fn end_run(&mut self){
        self.current_run += 1;
    }

    pub fn init_train(&mut self){
        self.end_train();
        self.step_properties_map = Some(build_trackers(&self.step_properties))
    }

    pub fn log_prop(&mut self, prop: ExperimentProperty, val: LoggedValue){
        if self.should_run(prop){
            if prop.is_step_prop(){
                if let Some(prop_map) = &mut self.step_properties_map{
                    if let Some(prop) = prop_map.get_mut(&prop){
                        prop.add_val(val)
                    }else {
                        panic!("tried logging to uninitialized step prop")
                    }
                }
            } else {
                if let Some(prop) = self.meta_properties_map.get_mut(&prop){
                        prop.add_val(val)
                    }else {
                        panic!("tried logging to uninitialized meta prop")
                    }
            }
        }
    }

    pub fn dump_to_json_file(&self, file_path: String){
        let f = File::create(file_path).unwrap();
        let mut writer = BufWriter::new(f);
        let mut line = String::new();


        let meta_props = get_as_sorted_json_string(&self.meta_properties_map);
        let iter_props = self.map_hist.iter().map(|hist_elm| get_as_sorted_json_string(hist_elm)).collect::<Vec<_>>();

        let json = json!({
            "dataset": self.dataset_used.to_string(),
            "meta_props": meta_props,
            "iter_props": iter_props,
            "params": self.params.as_ref().map(|p| serde_json::to_value(&p).unwrap()).unwrap_or(serde_json::Value::Null),
        });

        serde_json::to_writer_pretty(writer, &json);
    }
    pub fn log_params(&mut self, params: &Params){
        self.params = Some(params.clone());
    }
    
    pub fn log_multi_run_acc(&self){
        if let Some(tp) = self.meta_properties_map.get(&ExperimentProperty::FoldAccuracy){
            let (test_acc_vec, train_acc_vec): (Vec<f64>, Vec<f64>) = tp.prop_values.iter().map(|lv| {
                if let LoggedValue::MappedFloats(m) = lv{
                    (m.get("test").unwrap(), m.get("train").unwrap())
                }else { 
                    panic!()
                }
            }).unzip();
            println!("with vec {:?},",  test_acc_vec);
            println!("with dataset {:?}, over {:?} runs", self.dataset_used, test_acc_vec.len());
            println!("train acc mean: {:?}, std: {:?}", (&train_acc_vec).mean(), train_acc_vec.std_dev());
            println!("test acc mean: {:?}, std: {:?}", (&test_acc_vec).mean(), test_acc_vec.std_dev());
        }
    }
}


fn get_as_sorted_json_string(prop_map: &HashMap<ExperimentProperty, TrackedProperty>) -> Value{
    let mut meta_props: Vec<_> = prop_map.values().collect();
    meta_props.sort_by_key(|tp| tp.property_type.to_string());

    return serde_json::to_value(&meta_props).unwrap()
}