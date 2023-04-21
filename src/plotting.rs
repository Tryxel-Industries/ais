extern crate core;

use plotters::prelude::*;

pub fn plot_hist(hist: Vec<f64>, file_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("train_graph{:?}.png", file_name);
    let root = BitMapBackend::new(&path, (3000, 1000)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_y = *hist.iter().max_by(|a, b| a.total_cmp(b)).unwrap() as f32;
    let min_y = *hist.iter().min_by(|a, b| a.total_cmp(b)).unwrap() as f32;
    // let min_y = hist.get(hist.len()/2).unwrap().clone() as f32;
    let mut chart = ChartBuilder::on(&root)
        .caption("y=x^2", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0f32..(hist.len() as f32), min_y..max_y)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            hist.iter().enumerate().map(|(y, x)| (y as f32, *x as f32)),
            &RED,
        ))?
        .label("y = x^2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}
