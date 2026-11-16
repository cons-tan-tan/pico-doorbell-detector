use doorbell_firmware::{
    config,
    decision::Chime,
    frontend::{AudioWindow, FEATURES, FRAME, SAMPLE_RATE},
    model::Model,
    rules::Detector,
};
use std::io::{self, Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let patterns_only = match arguments.as_slice() {
        [] => false,
        [argument] if argument == "--patterns-only" => true,
        _ => return Err("Usage: replay [--patterns-only]".into()),
    };
    let mut bytes = Vec::new();
    io::stdin().read_to_end(&mut bytes)?;
    if bytes.len() % 4 != 0 {
        return Err("Expected mono little-endian float32 PCM at 8000 Hz".into());
    }
    let mut detector = Detector::new(SAMPLE_RATE as f32);
    let mut tolerant = Detector::tolerant(SAMPLE_RATE as f32);
    let mut model = if patterns_only {
        None
    } else {
        Some(Model::take().map_err(|code| format!("TFLM initialization: {code}"))?)
    };
    let mut window = AudioWindow::new();
    let mut features = [0; FEATURES];
    let mut frame = [0.0; FRAME];
    println!("model,{}", config::MODEL_SHA256);
    println!("hop_samples,{}", config::INFERENCE_HOP);
    println!("trace_version,3");
    for (index, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let sample = f32::from_le_bytes(*chunk);
        if !sample.is_finite() {
            return Err("Non-finite PCM sample".into());
        }
        let adc = (sample * 2048.0 + 2048.0).round().clamp(0.0, 4095.0) as u16;
        frame[index % FRAME] = (adc as f32 - 2048.0) / 2048.0;
        let seconds = (index + 1) as f64 / SAMPLE_RATE as f64;
        if (index + 1).is_multiple_of(FRAME)
            && let Some(event) = tolerant.feed(&frame)
        {
            let label = match event {
                Chime::Entrance => "doorbell_type1",
                Chime::Door => "doorbell_type2",
            };
            println!("pattern,{seconds:.6},{label}");
        }
        if (index + 1).is_multiple_of(FRAME)
            && let Some(event) = detector.feed(&frame)
        {
            let label = match event {
                Chime::Entrance => "doorbell_type1",
                Chime::Door => "doorbell_type2",
            };
            println!("candidate,{seconds:.6},{label}");
        }
        if !patterns_only && window.push(adc) {
            window.features(&mut features);
            let scores = model
                .as_mut()
                .unwrap()
                .predict(&features)
                .map_err(|code| format!("TFLM inference: {code}"))?;
            println!(
                "prediction,{seconds:.6},{},{},{}",
                scores[0], scores[1], scores[2]
            );
        }
    }
    Ok(())
}
