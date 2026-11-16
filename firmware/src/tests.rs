use crate::{
    config,
    decision::{Chime, Decision},
    frontend::*,
    model::Model,
};
use std::vec::Vec;

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[test]
fn sparse_mel_matches_dense_reference() {
    let dense_mel = floats(include_bytes!(concat!(
        env!("DOORBELL_MODEL_DIR"),
        "/mel.f32"
    )));
    let mut seed = 123u32;
    for level in [
        0.0,
        1.0 / 2048.0,
        0.001,
        0.002,
        0.003,
        0.005,
        0.01,
        0.1,
        1.0,
    ] {
        for pattern in 0..32 {
            let samples = core::array::from_fn(|i| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let value = match pattern {
                    0 => 1.0,
                    1 => {
                        if i % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    2 => {
                        if i == FRAME / 2 {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    3 => libm::sinf(i as f32 * 0.7),
                    _ => ((seed >> 16) as f32 - 32768.0) / 32768.0,
                };
                value * level
            });
            let mean = samples.iter().sum::<f32>() / FRAME as f32;
            let mut windowed = core::array::from_fn(|i| (samples[i] - mean) * config::HANN[i]);
            let spectrum = microfft::real::rfft_256(&mut windowed);
            let mut power = [0.0; FRAME / 2 + 1];
            power[0] = spectrum[0].re * spectrum[0].re / (FRAME * FRAME) as f32;
            power[FRAME / 2] = spectrum[0].im * spectrum[0].im / (FRAME * FRAME) as f32;
            for i in 1..FRAME / 2 {
                power[i] = (spectrum[i].re * spectrum[i].re + spectrum[i].im * spectrum[i].im)
                    / (FRAME * FRAME) as f32;
            }
            let mut actual = [0.0; BINS];
            frame_features(&samples, &mut actual);
            for (bin, value) in actual.into_iter().enumerate() {
                let energy: f32 = power
                    .iter()
                    .zip(&dense_mel[bin * 129..(bin + 1) * 129])
                    .map(|(p, w)| p * w)
                    .sum();
                let expected = (libm::log10f(energy.max(1e-10)).clamp(-10.0, 0.0) + 10.0) / 10.0;
                assert!(
                    (value - expected).abs() < 2e-5,
                    "level={level} pattern={pattern} bin={bin}: {value} vs {expected}"
                );
                assert!((quantize(value) as i16 - quantize(expected) as i16).abs() <= 1);
            }
        }
    }
}

#[test]
fn python_frontend_and_litert_reference() {
    let pcm = floats(include_bytes!(concat!(
        env!("DOORBELL_MODEL_DIR"),
        "/reference-pcm.f32"
    )));
    let expected = floats(include_bytes!(concat!(
        env!("DOORBELL_MODEL_DIR"),
        "/reference-features.f32"
    )));
    let input =
        include_bytes!(concat!(env!("DOORBELL_MODEL_DIR"), "/reference-input.i8")).map(|b| b as i8);
    let output = include_bytes!(concat!(env!("DOORBELL_MODEL_DIR"), "/reference-output.i8"))
        .map(|b| b as i8);
    let mut features = [0; FEATURES];
    let mut bins = [0.0; BINS];
    let mut frontend = FrameFrontend::new();
    for frame in 0..124 {
        frontend.process(
            pcm[frame * HOP..frame * HOP + FRAME].try_into().unwrap(),
            &mut bins,
        );
        for (bin, value) in bins.iter().enumerate() {
            let index = frame * BINS + bin;
            assert!(
                (value - expected[index]).abs() < 2e-5,
                "feature {index}: {value} vs {}",
                expected[index]
            );
            features[index] = quantize(*value);
            assert!((features[index] as i16 - input[index] as i16).abs() <= 1);
        }
    }
    let mut model = Model::take().unwrap();
    assert_eq!(model.predict(&input).unwrap(), output);
    let computed = model.predict(&features).unwrap();
    for (a, b) in computed.into_iter().zip(output) {
        assert!((a as i16 - b as i16).abs() <= 1);
    }
    assert!(model.arena_used() > 0 && model.arena_used() <= 32768);
    assert!(Model::take().is_err());
    std::println!("TFLM reference passed; arena={} bytes", model.arena_used());
}

#[test]
fn streaming_window_timing_and_alignment() {
    let mut window = AudioWindow::new();
    let count = WINDOW + config::INFERENCE_HOP * 20 + 57;
    let mut seed = 42u32;
    let adc: Vec<u16> = (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 16) as u16
        })
        .collect();
    let mut actual = [0; FEATURES];
    let mut frontend = FrameFrontend::new();
    let expected: Vec<i8> = (0..=(count - FRAME) / HOP)
        .flat_map(|frame| {
            let samples = core::array::from_fn(|index| {
                ((adc[frame * HOP + index] & 0x0fff) as f32 - 2048.0) / 2048.0
            });
            let mut bins = [0.0; BINS];
            frontend.process(&samples, &mut bins);
            bins.map(quantize)
        })
        .collect();
    for (i, sample) in adc.iter().enumerate() {
        let ready = i + 1 >= WINDOW && (i + 1 - WINDOW).is_multiple_of(config::INFERENCE_HOP);
        assert_eq!(window.push(*sample), ready);
        if ready {
            window.features(&mut actual);
            let start = i + 1 - WINDOW;
            let offset = start / HOP * BINS;
            assert_eq!(actual, expected[offset..offset + FEATURES]);
        }
    }
    window.reset();
    for _ in 0..WINDOW - 1 {
        assert!(!window.push(2048));
    }
    assert!(window.push(2048));
    window.features(&mut actual);
    assert!(actual.iter().all(|x| *x == quantize(0.0)));
    assert!(core::mem::size_of::<AudioWindow>() < 4096);
}

#[test]
fn decisions_reject_background_and_preserve_cooldown_across_gaps() {
    let mut state = Decision::new();
    assert_eq!(state.update([127, -128, -128], 0), None);
    assert_eq!(state.update([-128, 0, 0], 100), None);
    for i in 0..config::CONSECUTIVE {
        let expected = if i + 1 == config::CONSECUTIVE {
            Some(Chime::Entrance)
        } else {
            None
        };
        assert_eq!(state.update([-128, 127, -128], 200), expected);
    }
    state.reset_history();
    for _ in 0..config::CONSECUTIVE {
        assert_eq!(state.update([-128, -128, 127], 201), None);
    }
    assert_eq!(
        state.update([-128, -128, 127], 200 + config::COOLDOWN_MS),
        Some(Chime::Door)
    );
}

#[test]
fn audio_levels_separate_dc_from_sound_and_reset_between_reports() {
    let mut levels = crate::levels::AudioLevels::default();
    assert!(levels.take().is_none());
    for _ in 0..WINDOW {
        levels.push(1489);
    }
    let quiet = levels.take().unwrap();
    assert_eq!(quiet.samples, WINDOW as u32);
    assert_eq!((quiet.min, quiet.max), (1489, 1489));
    assert_eq!(quiet.mean, 1489.0);
    assert_eq!(quiet.ac_rms, 0.0);
    assert_eq!(quiet.rails, 0);
    assert!(levels.take().is_none());

    for _ in 0..WINDOW / 2 {
        levels.push(4093);
        levels.push(4094);
    }
    let small_signal = levels.take().unwrap();
    assert_eq!((small_signal.min, small_signal.max), (4093, 4094));
    assert_eq!(small_signal.mean, 4093.5);
    assert_eq!(small_signal.ac_rms, 0.5);
    assert_eq!(small_signal.rails, 0);

    for _ in 0..WINDOW / 2 {
        levels.push(0);
        levels.push(4095);
    }
    let full_scale = levels.take().unwrap();
    assert_eq!((full_scale.min, full_scale.max), (0, 4095));
    assert_eq!(full_scale.mean, 2047.5);
    assert_eq!(full_scale.ac_rms, 2047.5);
    assert_eq!(full_scale.rails, WINDOW as u32);
}

#[test]
fn legacy_rules_detect_both_patterns_across_gain_and_dc_offset() {
    fn replay(segments: &[(f32, usize)], gain: f32, dc: f32) -> Vec<Chime> {
        let mut detector = crate::rules::Detector::new(SAMPLE_RATE as f32);
        let mut events = Vec::new();
        let mut index = 0;
        for &(frequency, frames) in segments {
            for _ in 0..frames {
                let samples = core::array::from_fn(|_| {
                    let value = dc
                        + gain
                            * libm::sinf(
                                2.0 * core::f32::consts::PI * frequency * index as f32
                                    / SAMPLE_RATE as f32,
                            );
                    index += 1;
                    value
                });
                if let Some(event) = detector.feed(&samples) {
                    events.push(event);
                }
            }
        }
        events
    }
    for gain in [0.01, 0.2, 0.6] {
        for dc in [0.0, 0.3] {
            assert_eq!(replay(&[(850.0, 16), (662.0, 8)], gain, dc), [Chime::Door]);
            let trill: Vec<_> = (0..7)
                .map(|i| (if i % 2 == 0 { 850.0 } else { 662.0 }, 4))
                .collect();
            assert_eq!(replay(&trill, gain, dc), [Chime::Entrance]);
        }
    }
    for segments in [
        &[(440.0, 100)][..],
        &[(850.0, 64)][..],
        &[(850.0, 8), (0.0, 48), (662.0, 16)][..],
        &[
            (850.0, 4),
            (662.0, 4),
            (850.0, 4),
            (0.0, 4),
            (662.0, 4),
            (850.0, 4),
            (662.0, 4),
        ][..],
    ] {
        assert!(replay(segments, 0.2, 0.3).is_empty());
    }
}

fn fusion_policy() -> crate::fusion::Policy {
    crate::fusion::Policy {
        low: 0.6,
        high: 0.95,
        confirmation_frames: 3,
        minimum_support: 2,
        rule_hold_ms: 768,
        cooldown_ms: 3000,
        use_rules: true,
    }
}

#[test]
fn notification_timeout_does_not_delay_the_first_event_or_extend_with_more_sound() {
    let mut policy = fusion_policy();
    policy.cooldown_ms = 10000;
    let mut decision = crate::fusion::Decision::new(policy);
    let strong = scores([0.0, 1.0, 0.0]);
    assert_eq!(decision.update(strong, 1000), None);
    assert_eq!(
        decision.observe_pattern(Chime::Entrance, 1100),
        Some(Chime::Entrance)
    );
    for time in (1300..11100).step_by(256) {
        assert_eq!(decision.update(strong, time), None);
        assert_eq!(decision.observe_pattern(Chime::Entrance, time), None);
    }
    decision.reset_history();
    assert_eq!(decision.observe_pattern(Chime::Entrance, 11100), None);
    assert_eq!(decision.update(strong, 11100), Some(Chime::Entrance));
}

#[test]
fn pattern_hold_allows_delayed_model_evidence_but_expires() {
    let policy = fusion_policy().with_rule_hold_ms(1280);
    let mut decision = crate::fusion::Decision::new(policy);
    assert_eq!(decision.observe_pattern(Chime::Door, 2000), None);
    assert_eq!(decision.update(scores([0.7, 0.0, 0.3]), 2512), None);
    assert_eq!(
        decision.update(scores([0.3, 0.0, 0.7]), 3072),
        Some(Chime::Door)
    );
    let mut expired = crate::fusion::Decision::new(policy);
    assert_eq!(expired.observe_pattern(Chime::Door, 2000), None);
    assert_eq!(expired.update(scores([0.3, 0.0, 0.7]), 3281), None);
}

fn scores(probabilities: [f32; 3]) -> [i8; 3] {
    probabilities.map(|p| {
        (libm::roundf(p / config::OUTPUT_SCALE) as i32 + config::OUTPUT_ZERO_POINT).clamp(-128, 127)
            as i8
    })
}

#[test]
fn moderate_door_scores_need_a_matching_pattern_even_when_repeated() {
    let policy = crate::fusion::Policy::new(0.9, 0.98, 3, true).with_rule_hold_ms(1280);
    let mut decision = crate::fusion::Decision::new(policy);
    for (index, door) in [0.7, 0.8, 0.9, 0.95, 0.97, 0.95].into_iter().enumerate() {
        assert_eq!(
            decision.update(scores([1.0 - door, 0.0, door]), 1000 + index as u64 * 256),
            None
        );
    }
    let status = decision.status(2280);
    assert_eq!(status.pattern, None);
    assert_eq!((status.frames, status.support), (0, [0; 2]));
    assert!(!status.cooling_down);
    assert_eq!(decision.observe_pattern(Chime::Entrance, 2300), None);
    assert_eq!(decision.status(2300).pattern, Some(Chime::Entrance));
    assert_eq!(
        decision.observe_pattern(Chime::Door, 2301),
        Some(Chime::Door)
    );
    assert!(decision.status(2301).cooling_down);
    decision.reset_history();
    assert_eq!(decision.status(2302).pattern, None);
    assert!(decision.status(2302).cooling_down);
    assert!(!decision.status(2301 + policy.cooldown_ms).cooling_down);
}

#[test]
fn pattern_status_and_quantized_threshold_share_the_acceptance_boundary() {
    let policy = crate::fusion::Policy::new(0.9, 0.98, 3, true).with_rule_hold_ms(1280);
    for (score, time, accepted) in [(102, 2280, false), (103, 2280, true), (103, 2281, false)] {
        let mut decision = crate::fusion::Decision::new(policy);
        assert_eq!(decision.observe_pattern(Chime::Door, 1000), None);
        assert_eq!(
            decision.status(time).pattern,
            if time <= 2280 {
                Some(Chime::Door)
            } else {
                None
            }
        );
        assert_eq!(
            decision.update([-score, -128, score], time),
            if accepted { Some(Chime::Door) } else { None }
        );
    }
    let mut disabled = policy;
    disabled.use_rules = false;
    let mut decision = crate::fusion::Decision::new(disabled);
    assert_eq!(decision.observe_pattern(Chime::Door, 1000), None);
    assert_eq!(decision.status(1000).pattern, None);
    assert_eq!(decision.update([-103, -128, 103], 1000), None);
}

#[test]
fn ml_confirmation_status_counts_high_scores_without_starting_cooldown_early() {
    let policy = crate::fusion::Policy::new(0.9, 0.98, 3, true);
    for (last, expected) in [(122, None), (123, Some(Chime::Door))] {
        let mut decision = crate::fusion::Decision::new(policy);
        assert_eq!(decision.update([-123, -128, 123], 1000), None);
        let status = decision.status(1000);
        assert_eq!((status.frames, status.support), (1, [0, 1]));
        assert!(!status.cooling_down);
        assert_eq!(decision.update([-122, -128, 122], 1256), None);
        let status = decision.status(1256);
        assert_eq!((status.frames, status.support), (2, [0, 1]));
        assert_eq!(decision.update([-last, -128, last], 1512), expected);
        let status = decision.status(1512);
        assert_eq!((status.frames, status.support), (0, [0; 2]));
        assert_eq!(status.cooling_down, expected.is_some());
    }
}

#[test]
fn temporal_confirmation_requires_repeated_support_for_the_same_kind() {
    let mut policy = fusion_policy();
    policy.confirmation_frames = 2;
    let mut decision = crate::fusion::Decision::new(policy);
    assert_eq!(decision.update(scores([0.0, 1.0, 0.0]), 1000), None);
    assert_eq!(decision.update(scores([0.0, 0.0, 1.0]), 1256), None);
    assert_eq!(decision.update(scores([0.0, 1.0, 0.0]), 1512), None);
    assert_eq!(
        decision.update(scores([0.0, 1.0, 0.0]), 1768),
        Some(Chime::Entrance)
    );

    let mut policy = fusion_policy().with_minimum_support(3);
    policy.confirmation_frames = 4;
    let mut decision = crate::fusion::Decision::new(policy);
    for (time, value) in [
        (1000, [0.0, 1.0, 0.0]),
        (1256, [0.0, 0.0, 1.0]),
        (1512, [0.0, 0.97, 0.03]),
    ] {
        assert_eq!(decision.update(scores(value), time), None);
    }
    assert_eq!(
        decision.update(scores([0.0, 0.97, 0.03]), 1768),
        Some(Chime::Entrance)
    );
}

#[test]
fn confirmation_defers_type_and_cooldown_until_more_audio_arrives() {
    let mut decision = crate::fusion::Decision::new(fusion_policy());
    assert_eq!(decision.update(scores([0.0, 0.96, 0.04]), 1000), None);
    assert_eq!(decision.update(scores([0.0, 0.02, 0.98]), 1256), None);
    assert_eq!(
        decision.update(scores([0.0, 0.0, 1.0]), 1512),
        Some(Chime::Door)
    );
    decision.reset_history();
    for time in [1768, 2024, 2280, 4500] {
        assert_eq!(decision.update(scores([0.0, 1.0, 0.0]), time), None);
    }
    for time in [4512, 4768] {
        assert_eq!(decision.update(scores([0.0, 1.0, 0.0]), time), None);
    }
    assert_eq!(
        decision.update(scores([0.0, 1.0, 0.0]), 5024),
        Some(Chime::Entrance)
    );
}

#[test]
fn strong_ml_bypasses_patterns_but_moderate_scores_need_recent_matching_evidence() {
    let mut decision = crate::fusion::Decision::new(fusion_policy());
    for time in [1000, 1256, 1512] {
        assert_eq!(decision.update(scores([0.2, 0.7, 0.1]), time), None);
    }
    assert_eq!(decision.observe_pattern(Chime::Door, 1600), None);
    for time in [1700, 1956, 2212] {
        assert_eq!(decision.update(scores([0.2, 0.7, 0.1]), time), None);
    }
    for time in [2400, 2656, 2912] {
        assert_eq!(decision.update(scores([0.2, 0.1, 0.7]), time), None);
    }
    for time in [3200, 3456] {
        assert_eq!(decision.update(scores([0.0, 0.0, 1.0]), time), None);
    }
    assert_eq!(
        decision.update(scores([0.0, 0.0, 1.0]), 3712),
        Some(Chime::Door)
    );
}

#[test]
fn rejected_spike_and_capture_gap_do_not_start_cooldown() {
    let mut decision = crate::fusion::Decision::new(fusion_policy());
    assert_eq!(decision.update(scores([0.0, 1.0, 0.0]), 1000), None);
    assert_eq!(decision.update(scores([1.0, 0.0, 0.0]), 1256), None);
    assert_eq!(decision.update(scores([1.0, 0.0, 0.0]), 1512), None);
    assert_eq!(decision.update(scores([0.2, 0.7, 0.1]), 1768), None);
    decision.reset_history();
    assert_eq!(decision.observe_pattern(Chime::Entrance, 1900), None);
    decision.reset_history();
    for time in [2024, 2280, 2536] {
        assert_eq!(decision.update(scores([0.2, 0.7, 0.1]), time), None);
    }
    assert_eq!(
        decision.observe_pattern(Chime::Entrance, 2600),
        Some(Chime::Entrance)
    );
}

#[test]
fn pattern_confirmation_does_not_reuse_stale_predictions_or_backdate_events() {
    let mut decision = crate::fusion::Decision::new(fusion_policy());
    assert_eq!(decision.update(scores([0.2, 0.1, 0.7]), 1000), None);
    assert_eq!(decision.observe_pattern(Chime::Door, 1300), None);
    assert_eq!(
        decision.update(scores([0.2, 0.1, 0.7]), 1512),
        Some(Chime::Door)
    );
    assert_eq!(decision.observe_pattern(Chime::Door, 1600), None);
}

#[test]
fn tolerant_patterns_cover_pitch_shift_without_accepting_a_single_tone() {
    for speed in [0.96, 0.98, 1.0, 1.02, 1.04] {
        for kind in [Chime::Door, Chime::Entrance] {
            let mut detector = crate::rules::Detector::tolerant(SAMPLE_RATE as f32);
            let mut events = Vec::new();
            for frame_index in 0..32 {
                let hz = match kind {
                    Chime::Door => {
                        if frame_index < 16 {
                            850.0
                        } else {
                            662.0
                        }
                    }
                    Chime::Entrance => {
                        if frame_index / 4 % 2 == 0 {
                            850.0
                        } else {
                            662.0
                        }
                    }
                };
                let frame = core::array::from_fn(|i| {
                    0.3 + 0.02
                        * libm::sinf(
                            2.0 * core::f32::consts::PI
                                * hz
                                * speed
                                * (frame_index * FRAME + i) as f32
                                / SAMPLE_RATE as f32,
                        )
                });
                if let Some(event) = detector.feed(&frame) {
                    events.push(event);
                }
            }
            assert_eq!(events, [kind]);
        }
    }
    let mut detector = crate::rules::Detector::tolerant(SAMPLE_RATE as f32);
    let frame = core::array::from_fn(|i| {
        0.02 * libm::sinf(2.0 * core::f32::consts::PI * 850.0 * i as f32 / SAMPLE_RATE as f32)
    });
    for _ in 0..100 {
        assert_eq!(detector.feed(&frame), None);
    }
    detector.reset();
    for _ in 0..100 {
        assert_eq!(detector.feed(&[0.3; FRAME]), None);
    }
}

#[test]
fn ntfy_request_counts_utf8_bytes_and_validates_http_status() {
    let mut request = std::string::String::new();
    crate::notification::request(&mut request, "/debug", Chime::Door).unwrap();
    let (headers, body) = request.split_once("\r\n\r\n").unwrap();
    assert!(headers.starts_with("POST /debug HTTP/1.1\r\n"));
    assert!(headers.contains(&std::format!("Content-Length: {}\r\n", body.len())));
    assert_eq!(
        crate::notification::status(b"HTTP/1.1 200 OK\r\n"),
        Some(200)
    );
    assert_eq!(
        crate::notification::status(b"HTTP/1.0 503 Unavailable\r\n"),
        Some(503)
    );
    for line in [
        b"HTTP/1.1 20x OK\r\n".as_slice(),
        b"HTTP/1.1 200",
        b"200 OK\r\n",
        b"HTTP/1.1 2000 OK\r\n",
    ] {
        assert_eq!(crate::notification::status(line), None);
    }
}
