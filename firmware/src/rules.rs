use crate::{decision::Chime, frontend::FRAME};

const HIGH_PASS_HZ: f32 = 250.0;

fn ac_power(frame: &[f32; FRAME]) -> f32 {
    let mean = frame.iter().sum::<f32>() / FRAME as f32;
    frame
        .iter()
        .map(|sample| (sample - mean) * (sample - mean))
        .sum::<f32>()
        / FRAME as f32
}

struct HighPass {
    gain: f32,
    a1: f32,
    a2: f32,
    state: Option<[f32; 2]>,
}

impl HighPass {
    fn new(sample_rate: f32, cutoff: f32) -> Self {
        let k = libm::tanf(core::f32::consts::PI * cutoff / sample_rate);
        let gain = 1.0 / (1.0 + core::f32::consts::SQRT_2 * k + k * k);
        Self {
            gain,
            a1: 2.0 * (k * k - 1.0) * gain,
            a2: (1.0 - core::f32::consts::SQRT_2 * k + k * k) * gain,
            state: None,
        }
    }

    fn process(&mut self, frame: &[f32; FRAME]) -> [f32; FRAME] {
        frame.map(|sample| {
            let state = self
                .state
                .get_or_insert([-self.gain * sample, self.gain * sample]);
            let output = self.gain * sample + state[0];
            state[0] = -2.0 * self.gain * sample - self.a1 * output + state[1];
            state[1] = self.gain * sample - self.a2 * output;
            output
        })
    }
}

fn ratios<const N: usize>(frame: &[f32; FRAME], coefficients: [f32; N]) -> [f32; N] {
    let mean = frame.iter().sum::<f32>() / FRAME as f32;
    let centered = frame.map(|x| x - mean);
    let rms = libm::sqrtf(centered.iter().map(|x| x * x).sum::<f32>() / FRAME as f32);
    if rms < 0.001 {
        return [0.0; N];
    }
    coefficients.map(|coefficient| {
        let (mut s1, mut s2) = (0.0, 0.0);
        for sample in centered {
            let s0 = sample + coefficient * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        let power = (s1 * s1 + s2 * s2 - coefficient * s1 * s2).max(0.0);
        libm::sqrtf(power) / (rms * FRAME as f32)
    })
}

pub struct Detector {
    coefficients: [f32; 10],
    tolerant: bool,
    high_pass: Option<HighPass>,
    ding_streak: usize,
    since_ding: Option<usize>,
    dong_window: [bool; 5],
    dong_index: usize,
    dong_frames: usize,
    tone: Option<usize>,
    tone_frames: usize,
    alternations: usize,
    gap_frames: usize,
}

impl Detector {
    pub fn new(sample_rate: f32) -> Self {
        assert!(sample_rate.is_finite() && sample_rate > 1800.0);
        Self {
            coefficients: core::array::from_fn(|i| {
                let hz = [850.0, 662.0][i / 5] * [0.96, 0.98, 1.0, 1.02, 1.04][i % 5];
                2.0 * libm::cosf(2.0 * core::f32::consts::PI * hz / sample_rate)
            }),
            tolerant: false,
            high_pass: None,
            ding_streak: 0,
            since_ding: None,
            dong_window: [false; 5],
            dong_index: 0,
            dong_frames: 0,
            tone: None,
            tone_frames: 0,
            alternations: 0,
            gap_frames: 0,
        }
    }

    pub fn tolerant(sample_rate: f32) -> Self {
        Self {
            tolerant: true,
            high_pass: Some(HighPass::new(sample_rate, HIGH_PASS_HZ)),
            ..Self::new(sample_rate)
        }
    }

    pub fn reset(&mut self) {
        if let Some(filter) = &mut self.high_pass {
            filter.state = None;
        }
        self.reset_pattern();
    }

    fn reset_pattern(&mut self) {
        self.ding_streak = 0;
        self.since_ding = None;
        self.dong_window.fill(false);
        self.dong_index = 0;
        self.dong_frames = 0;
        self.reset_trill();
    }

    fn reset_trill(&mut self) {
        self.tone = None;
        self.tone_frames = 0;
        self.alternations = 0;
        self.gap_frames = 0;
    }

    pub fn feed(&mut self, frame: &[f32; FRAME]) -> Option<Chime> {
        let filtered;
        let frame = if let Some(filter) = &mut self.high_pass {
            filtered = filter.process(frame);
            // Unconditional filtering made bass-heavy music look like a chime.
            if ac_power(&filtered) < 0.25 * ac_power(frame) {
                &filtered
            } else {
                frame
            }
        } else {
            frame
        };
        let ratios = if self.tolerant {
            let bands = ratios(frame, self.coefficients);
            [
                bands[..5].iter().copied().fold(0.0, f32::max),
                bands[5..].iter().copied().fold(0.0, f32::max),
            ]
        } else {
            ratios(frame, [self.coefficients[2], self.coefficients[7]])
        };
        let tone = if ratios[0] > 0.30 && ratios[0] > ratios[1] * 1.5 {
            Some(0)
        } else if ratios[1] > 0.30 && ratios[1] > ratios[0] * 1.5 {
            Some(1)
        } else {
            None
        };
        if self.feed_trill(tone) {
            self.reset_pattern();
            return Some(Chime::Entrance);
        }
        self.dong_window[self.dong_index] = ratios[1] > 0.20;
        self.dong_index = (self.dong_index + 1) % self.dong_window.len();
        self.dong_frames = (self.dong_frames + 1).min(self.dong_window.len());
        if let Some(frames) = self.since_ding {
            let frames = frames + 1;
            self.since_ding = Some(frames);
            if frames > 47 {
                self.since_ding = None;
                self.ding_streak = 0;
            } else if frames >= 10
                && self.dong_frames == 5
                && self.dong_window.iter().filter(|x| **x).count() >= 3
            {
                self.reset_pattern();
                return Some(Chime::Door);
            }
        } else {
            self.ding_streak = if ratios[0] > 0.25 {
                self.ding_streak + 1
            } else {
                0
            };
            if self.ding_streak >= 8 {
                self.since_ding = Some(0);
            }
        }
        None
    }

    fn feed_trill(&mut self, tone: Option<usize>) -> bool {
        if tone.is_none() {
            self.gap_frames += 1;
            if self.gap_frames > 3 {
                self.reset_trill();
            }
            return false;
        }
        self.gap_frames = 0;
        if tone == self.tone {
            self.tone_frames = self.tone_frames.saturating_add(1);
            return false;
        }
        if self.tone.is_some() && self.tone_frames >= 2 {
            self.alternations += 1;
        }
        self.tone = tone;
        self.tone_frames = 1;
        self.alternations >= 6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_pass_matches_scipy_and_starts_at_the_input_dc_level() {
        let mut filter = HighPass::new(8000.0, 250.0);
        let mut frame = [0.3; FRAME];
        frame[..8].copy_from_slice(&[0.3, 0.31, 0.28, 0.32, 0.3, 0.26, 0.34, 0.27]);
        let expected = [
            0.0,
            0.008703308,
            -0.019810677,
            0.020180818,
            -0.002425116,
            -0.03687489,
            0.042712778,
            -0.02898789,
        ];
        let result = filter.process(&frame);
        for (actual, expected) in result.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-6);
        }
        for dc in [-0.75, 0.3, 0.9] {
            filter.state = None;
            for _ in 0..32 {
                assert!(filter.process(&[dc; FRAME]).iter().all(|x| x.abs() < 2e-6));
            }
        }
    }

    #[test]
    fn both_chimes_survive_hum_and_hum_alone_does_not_trigger() {
        for hum_hz in [50.0, 60.0] {
            for dc in [-0.75, 0.3] {
                for kind in [None, Some(Chime::Door), Some(Chime::Entrance)] {
                    let mut filtered = Detector::tolerant(8000.0);
                    let mut legacy = Detector::new(8000.0);
                    let mut detected = false;
                    for frame_index in 0..100 {
                        let hz = if !(16..72).contains(&frame_index) {
                            0.0
                        } else {
                            match kind {
                                Some(Chime::Door) => {
                                    if frame_index < 32 {
                                        850.0
                                    } else {
                                        662.0
                                    }
                                }
                                Some(Chime::Entrance) => {
                                    if frame_index / 4 % 2 == 0 {
                                        850.0
                                    } else {
                                        662.0
                                    }
                                }
                                None => 0.0,
                            }
                        };
                        let frame = core::array::from_fn(|i| {
                            let phase =
                                2.0 * core::f32::consts::PI * (frame_index * FRAME + i) as f32
                                    / 8000.0;
                            (dc + 0.4 * libm::sinf(phase * hum_hz) + 0.025 * libm::sinf(phase * hz))
                                .clamp(-1.0, 4095.0 / 2048.0 - 1.0)
                        });
                        assert_eq!(legacy.feed(&frame), None);
                        if let Some(event) = filtered.feed(&frame) {
                            assert_eq!(Some(event), kind);
                            detected = true;
                        }
                    }
                    assert_eq!(
                        detected,
                        kind.is_some(),
                        "hum={hum_hz} dc={dc} kind={kind:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn only_capture_gaps_reset_the_filter_state() {
        let mut detector = Detector::tolerant(8000.0);
        let frame = core::array::from_fn(|i| 0.3 + 0.01 * libm::sinf(i as f32 * 0.3));
        detector.feed(&frame);
        let previous = detector.high_pass.as_ref().unwrap().state;
        assert!(previous.is_some());
        detector.reset_pattern();
        assert_eq!(detector.high_pass.as_ref().unwrap().state, previous);
        detector.reset();
        assert_eq!(detector.high_pass.as_ref().unwrap().state, None);
    }

    #[test]
    fn bass_with_target_tones_does_not_gain_an_unconditional_filter_false_alarm() {
        let mut detector = Detector::tolerant(8000.0);
        let mut legacy = Detector::new(8000.0);
        let mut always_filtered = Detector::new(8000.0);
        let mut filter = HighPass::new(8000.0, HIGH_PASS_HZ);
        let mut naive_alarm = false;
        for frame_index in 0..100 {
            let frame = core::array::from_fn(|i| {
                let phase = 2.0 * core::f32::consts::PI * (frame_index * FRAME + i) as f32 / 8000.0;
                0.6 * libm::sinf(phase * 200.0)
                    + 0.17 * libm::sinf(phase * 850.0)
                    + 0.15 * libm::sinf(phase * 662.0)
            });
            assert_eq!(detector.feed(&frame), None);
            assert_eq!(legacy.feed(&frame), None);
            naive_alarm |= always_filtered.feed(&filter.process(&frame)).is_some();
        }
        assert!(naive_alarm);
    }
}
