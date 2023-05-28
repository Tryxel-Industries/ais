use std::collections::HashMap;
use std::hash::Hash;


use std::fs::File;
use std::io::{BufWriter, Write};
use crate::datasets::Datasets;

use strum_macros::{Display, EnumString};

use serde::Serialize;
use serde_json::{json, Value};

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
    TrainScore,
    PopLabelMemberships,

    TrainLabel,

    BoostAccuracy,
    BoostAccuracyTest,
    Runtime,

}





impl ExperimentProperty {
    const STEP_PROPS: [ExperimentProperty; 2] = [
        ExperimentProperty::TestAccuracy,
        ExperimentProperty::PopLabelMemberships
    ];
    pub fn is_step_prop(&self) -> bool{
        return ExperimentProperty::STEP_PROPS.contains(self);

    }
}



#[derive(Clone, PartialEq, Debug, Serialize, EnumString, Display)]
pub enum LoggedValue{
    TrainTest(f64,f64),
    CorWrongNoReg(usize,usize,usize),
    SingleValue(f64),
    LabelMembership(HashMap<usize,usize>),
    NoValue
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

}
fn build_trackers(active_properties: &Vec<ExperimentProperty>) -> HashMap<ExperimentProperty, TrackedProperty>{
        let properties_map: HashMap<_,_> = active_properties.iter()
            .map(|prop_type| (prop_type.clone(),TrackedProperty::new(prop_type.clone())))
            .collect();
    return properties_map;
}

impl  ExperimentLogger {
    pub fn new(dataset_used: Datasets, tracked_props: Vec<ExperimentProperty>) -> ExperimentLogger {

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
        }
    }



    pub fn should_run(&self, experiment_property: ExperimentProperty) -> bool{
        self.full_properties.contains(&experiment_property)
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
        self.step_properties_map = None;

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

        });

        serde_json::to_writer_pretty(writer, &json);
    }
}


fn get_as_sorted_json_string(prop_map: &HashMap<ExperimentProperty, TrackedProperty>) -> Value{
    let mut meta_props: Vec<_> = prop_map.values().collect();
    meta_props.sort_by_key(|tp| tp.property_type.to_string());

    return serde_json::to_value(&meta_props).unwrap()
}