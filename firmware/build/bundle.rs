use sha2::{Digest, Sha256};
use std::{format, fs, path::Path, string::String, vec::Vec};

fn read_floats(path: &Path, count: usize) -> Vec<f32> {
    let data = fs::read(path).unwrap();
    assert_eq!(
        data.len(),
        count * 4,
        "{}: wrong array length",
        path.display()
    );
    data.as_chunks::<4>()
        .0
        .iter()
        .map(|b| {
            let f = f32::from_le_bytes(*b);
            assert!(f.is_finite());
            f
        })
        .collect()
}

fn sparse_mel(model: &Path) -> String {
    let values = read_floats(&model.join("mel.f32"), 20 * 129);
    let mut result = String::from("pub static MEL: [(usize, &[f32]); 20] = [\n");
    for filter in values.as_chunks::<129>().0 {
        assert!(filter.iter().all(|v| *v >= 0.0));
        let start = filter.iter().position(|v| *v != 0.0).unwrap();
        let end = filter.iter().rposition(|v| *v != 0.0).unwrap() + 1;
        result += &format!("({start}, &{:?}),\n", &filter[start..end]);
    }
    result + "];\n"
}

pub fn configuration(model: &Path, manifest: &serde_json::Value) -> String {
    let files = [
        "model.tflite",
        "hann.f32",
        "mel.f32",
        "reference-pcm.f32",
        "reference-features.f32",
        "reference-input.i8",
        "reference-output.i8",
    ];
    let checksums = manifest["sha256"]
        .as_object()
        .expect("bundle checksums are required");
    assert_eq!(
        checksums.len(),
        files.len(),
        "unexpected bundle checksum entries"
    );
    for name in files {
        let digest = checksums
            .get(name)
            .and_then(|value| value.as_str())
            .unwrap_or_else(|| panic!("{name}: missing bundle checksum"));
        let data = fs::read(model.join(name)).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(data)),
            digest,
            "{name}: bundle checksum mismatch"
        );
    }
    let frontend = &manifest["frontend"];
    for (key, value) in [
        ("sample_rate", 8000),
        ("window_samples", 16000),
        ("frame_samples", 256),
        ("hop_samples", 128),
        ("mel_bins", 20),
    ] {
        assert_eq!(frontend[key].as_u64(), Some(value), "unsupported {key}");
    }
    assert_eq!(
        manifest["input"]["shape"],
        serde_json::json!([1, 124, 20, 1])
    );
    assert_eq!(manifest["output"]["shape"], serde_json::json!([1, 3]));
    for (key, value) in [("low_hz", 80.0), ("high_hz", 3800.0), ("log_floor", -10.0)] {
        assert_eq!(frontend[key].as_f64(), Some(value), "unsupported {key}");
    }
    assert_eq!(
        manifest["labels"],
        serde_json::json!(["background", "doorbell_type1", "doorbell_type2"])
    );
    let mut constants = format!(
        "pub const MODEL_SHA256: &str = {:?};\n",
        manifest["sha256"]["model.tflite"].as_str().unwrap()
    );
    let compression = frontend["compression"].as_str().unwrap_or("log");
    assert!(matches!(compression, "log" | "pcen"));
    constants += &format!("pub const PCEN: bool = {};\n", compression == "pcen");
    for (key, default) in [
        ("pcen_alpha", 0.98),
        ("pcen_delta", 2.0),
        ("pcen_root", 0.5),
        ("pcen_smooth", 0.04),
        ("pcen_epsilon", 1e-10),
        ("pcen_scale", 4.0),
    ] {
        let value = frontend[key].as_f64().unwrap_or(default);
        assert_eq!(value, default, "unsupported {key}");
        constants += &format!(
            "pub const {}: f32 = {:?};\n",
            key.to_uppercase(),
            value as f32
        );
    }
    for (prefix, key) in [("INPUT", "input"), ("OUTPUT", "output")] {
        assert_eq!(manifest[key]["dtype"], "int8");
        let scale = manifest[key]["scale"].as_f64().unwrap();
        let zero = manifest[key]["zero_point"].as_i64().unwrap();
        assert!(scale.is_finite() && scale > 0.0 && (-128..=127).contains(&zero));
        constants += &format!(
            "pub const {prefix}_SCALE: f32 = {:?};\npub const {prefix}_ZERO_POINT: i32 = {zero};\n",
            scale as f32
        );
    }
    let policy = &manifest["policy"];
    let threshold = policy["threshold"].as_f64().unwrap();
    let consecutive = policy["consecutive"].as_u64().unwrap();
    let hop = policy["hop_seconds"].as_f64().unwrap() * 8000.0;
    let cooldown = policy["cooldown_seconds"].as_f64().unwrap() * 1000.0;
    assert!(threshold > 0.0 && threshold <= 1.0 && consecutive > 0);
    assert!((1.0..=16000.0).contains(&hop) && hop.fract() == 0.0);
    assert_eq!(
        hop as usize % 128,
        0,
        "prediction hop must align with FFT frames"
    );
    assert!(cooldown.is_finite() && cooldown >= 0.0 && cooldown <= u32::MAX as f64);
    constants += &format!(
        "pub const THRESHOLD: f32 = {threshold:?};\npub const CONSECUTIVE: usize = {consecutive};\npub const INFERENCE_HOP: usize = {};\npub const COOLDOWN_MS: u64 = {};\n",
        hop as usize,
        cooldown.round() as u64
    );
    if let Some(fusion) = policy.get("fusion") {
        let low = fusion["low"].as_f64().unwrap();
        let high = fusion["high"].as_f64().unwrap();
        let frames = fusion["confirmation_frames"].as_u64().unwrap();
        let rules = fusion["use_rules"].as_bool().unwrap();
        let rule_hold = fusion
            .get("rule_hold_ms")
            .map_or(768, |value| value.as_u64().unwrap());
        assert!(rule_hold <= 2000);
        assert!(0.0 < low && low <= high && high <= 1.0 && (2..=4).contains(&frames));
        assert_eq!(hop, 2048.0);
        if let Some(seconds) = fusion.get("cooldown_seconds") {
            assert_eq!(seconds.as_f64().unwrap() * 1000.0, cooldown);
        }
        let support = fusion
            .get("minimum_support")
            .map_or(2, |value| value.as_u64().unwrap());
        assert!((2..=frames).contains(&support));
        constants += &format!(
            "pub const FUSION_POLICY: Option<crate::fusion::Policy> = Some(crate::fusion::Policy::new({low:?}, {high:?}, {frames}, {rules}).with_rule_hold_ms({rule_hold}).with_minimum_support({support}));\n"
        );
    } else {
        constants += "pub const FUSION_POLICY: Option<crate::fusion::Policy> = None;\n";
    }
    constants += &format!(
        "pub static HANN: [f32; 256] = {:?};\n",
        read_floats(&model.join("hann.f32"), 256)
    );
    constants += &sparse_mel(model);
    constants
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{panic::catch_unwind, path::PathBuf};

    fn fixture() -> (PathBuf, serde_json::Value) {
        let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("model");
        let manifest =
            serde_json::from_slice(&fs::read(model.join("manifest.json")).unwrap()).unwrap();
        (model, manifest)
    }

    #[test]
    fn every_embedded_file_requires_a_matching_checksum() {
        let (model, manifest) = fixture();
        assert!(configuration(&model, &manifest).contains("pub const MODEL_SHA256"));
        for name in manifest["sha256"].as_object().unwrap().keys() {
            let mut missing = manifest.clone();
            missing["sha256"].as_object_mut().unwrap().remove(name);
            assert!(catch_unwind(|| configuration(&model, &missing)).is_err());

            let mut stale = manifest.clone();
            stale["sha256"][name] = serde_json::json!("invalid checksum");
            assert!(catch_unwind(|| configuration(&model, &stale)).is_err());
        }
    }

    #[test]
    fn unknown_checksum_paths_are_rejected() {
        let (model, mut manifest) = fixture();
        manifest["sha256"]["../outside"] = serde_json::json!("invalid checksum");
        assert!(catch_unwind(|| configuration(&model, &manifest)).is_err());
    }

    #[test]
    fn cooldown_rounding_matches_decision_replay() {
        let (model, mut manifest) = fixture();
        manifest["policy"]["cooldown_seconds"] = serde_json::json!(0.0016);
        manifest["policy"]["fusion"]["cooldown_seconds"] = serde_json::json!(0.0016);
        assert!(configuration(&model, &manifest).contains("pub const COOLDOWN_MS: u64 = 2;"));
    }
}
