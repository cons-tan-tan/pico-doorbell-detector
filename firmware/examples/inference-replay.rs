use doorbell_firmware::{config, frontend::FEATURES, model::Model};
use std::io::{self, Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    io::stdin().read_to_end(&mut bytes)?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(FEATURES) {
        return Err("Expected complete int8 feature windows".into());
    }
    let mut model = Model::take().map_err(|code| format!("TFLM initialization: {code}"))?;
    println!("model,{}", config::MODEL_SHA256);
    println!("arena,{}", model.arena_used());
    for bytes in bytes.as_chunks::<FEATURES>().0 {
        let input = core::array::from_fn(|index| bytes[index] as i8);
        let output = model
            .predict(&input)
            .map_err(|code| format!("TFLM inference: {code}"))?;
        println!("{},{},{}", output[0], output[1], output[2]);
    }
    Ok(())
}
