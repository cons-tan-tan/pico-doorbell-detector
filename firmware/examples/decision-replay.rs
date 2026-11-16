use doorbell_firmware::{
    decision::Chime,
    fusion::{Decision, Policy},
};
use std::io::{self, BufRead};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(4..=7).contains(&args.len()) {
        return Err(
            "Usage: decision-replay LOW HIGH CONFIRMATION_FRAMES USE_RULES [COOLDOWN_SECONDS [RULE_HOLD_MS [MINIMUM_SUPPORT]]]"
                .into(),
        );
    }
    let mut policy = Policy::new(
        args[0].parse()?,
        args[1].parse()?,
        args[2].parse()?,
        args[3].parse()?,
    );
    if let Some(seconds) = args.get(4) {
        let milliseconds = seconds.parse::<f64>()? * 1000.0;
        if !milliseconds.is_finite() || !(0.0..=u32::MAX as f64).contains(&milliseconds) {
            return Err("Invalid notification cooldown".into());
        }
        policy.cooldown_ms = milliseconds.round() as u64;
    }
    if let Some(milliseconds) = args.get(5) {
        let milliseconds = milliseconds.parse()?;
        if milliseconds > 2000 {
            return Err("Pattern hold exceeds the inference window".into());
        }
        policy.rule_hold_ms = milliseconds;
    }
    if let Some(count) = args.get(6) {
        policy.minimum_support = count.parse()?;
    }
    let mut decision = Decision::new(policy);
    let mut clip = 0usize;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let values: Vec<_> = line.split(',').collect();
        match values.as_slice() {
            ["reset", index] => {
                clip = index.parse()?;
                decision = Decision::new(policy);
            }
            ["pattern", time, label] => {
                let kind = match *label {
                    "doorbell_type1" => Chime::Entrance,
                    "doorbell_type2" => Chime::Door,
                    _ => return Err("Invalid chime label".into()),
                };
                let time = time.parse()?;
                if let Some(event) = decision.observe_pattern(kind, time) {
                    print_event(clip, time, event);
                }
            }
            ["prediction", time, background, entrance, door] => {
                let time = time.parse()?;
                if let Some(event) = decision.update(
                    [background.parse()?, entrance.parse()?, door.parse()?],
                    time,
                ) {
                    print_event(clip, time, event);
                }
            }
            _ => return Err("Invalid decision trace".into()),
        }
    }
    Ok(())
}

fn print_event(clip: usize, time: u64, event: Chime) {
    let label = match event {
        Chime::Entrance => "doorbell_type1",
        Chime::Door => "doorbell_type2",
    };
    println!("{clip},{time},{label}");
}
