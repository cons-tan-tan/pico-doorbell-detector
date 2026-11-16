use crate::config;

pub const SAMPLE_RATE: usize = 8000;
pub const WINDOW: usize = 16000;
pub const FRAME: usize = 256;
pub const HOP: usize = 128;
pub const BINS: usize = 20;
pub const FRAMES: usize = 1 + (WINDOW - FRAME) / HOP;
pub const FEATURES: usize = FRAMES * BINS;

pub fn frame_features(samples: &[f32; FRAME], output: &mut [f32; BINS]) {
    frame_energy(samples, output);
    for value in output {
        *value = log_energy(*value);
    }
}

fn log_energy(energy: f32) -> f32 {
    (libm::log10f(energy.max(1e-10)).clamp(-10.0, 0.0) + 10.0) / 10.0
}

fn frame_energy(samples: &[f32; FRAME], output: &mut [f32; BINS]) {
    let mean = samples.iter().sum::<f32>() / FRAME as f32;
    let mut windowed = [0.0; FRAME];
    for (i, sample) in windowed.iter_mut().enumerate() {
        *sample = (samples[i] - mean) * config::HANN[i];
    }
    let spectrum = microfft::real::rfft_256(&mut windowed);
    let mut power = [0.0; FRAME / 2 + 1];
    power[0] = spectrum[0].re * spectrum[0].re / 65536.0;
    power[128] = spectrum[0].im * spectrum[0].im / 65536.0;
    for i in 1..128 {
        power[i] = (spectrum[i].re * spectrum[i].re + spectrum[i].im * spectrum[i].im) / 65536.0;
    }
    for (bin, value) in output.iter_mut().enumerate() {
        let (start, weights) = config::MEL[bin];
        *value = power[start..start + weights.len()]
            .iter()
            .zip(weights)
            .map(|(p, weight)| p * weight)
            .sum();
    }
}

pub struct FrameFrontend {
    smooth: [f32; BINS],
}

impl Default for FrameFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameFrontend {
    pub const fn new() -> Self {
        Self {
            smooth: [0.0; BINS],
        }
    }

    pub fn process(&mut self, samples: &[f32; FRAME], output: &mut [f32; BINS]) {
        frame_energy(samples, output);
        for (index, value) in output.iter_mut().enumerate() {
            if config::PCEN {
                self.smooth[index] += config::PCEN_SMOOTH * (*value - self.smooth[index]);
                let normalized = *value
                    / libm::powf(
                        config::PCEN_EPSILON + self.smooth[index],
                        config::PCEN_ALPHA,
                    );
                *value = ((libm::powf(normalized + config::PCEN_DELTA, config::PCEN_ROOT)
                    - libm::powf(config::PCEN_DELTA, config::PCEN_ROOT))
                    / config::PCEN_SCALE)
                    .clamp(0.0, 1.0);
            } else {
                *value = log_energy(*value);
            }
        }
    }
}

pub fn quantize(value: f32) -> i8 {
    (libm::rintf(value / config::INPUT_SCALE) as i32 + config::INPUT_ZERO_POINT).clamp(-128, 127)
        as i8
}

pub struct AudioWindow {
    samples: [i16; FRAME],
    features: [i8; FEATURES],
    next_sample: usize,
    next_feature: usize,
    until_frame: usize,
    until_prediction: usize,
    frontend: FrameFrontend,
}

impl Default for AudioWindow {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioWindow {
    pub const fn new() -> Self {
        Self {
            samples: [0; FRAME],
            features: [0; FEATURES],
            next_sample: 0,
            next_feature: 0,
            until_frame: FRAME,
            until_prediction: WINDOW,
            frontend: FrameFrontend::new(),
        }
    }

    pub fn reset(&mut self) {
        self.next_sample = 0;
        self.next_feature = 0;
        self.until_frame = FRAME;
        self.until_prediction = WINDOW;
        self.frontend = FrameFrontend::new();
    }

    pub fn push(&mut self, adc: u16) -> bool {
        self.samples[self.next_sample] = (adc & 0x0fff) as i16 - 2048;
        self.next_sample = (self.next_sample + 1) % FRAME;
        self.until_frame -= 1;
        if self.until_frame == 0 {
            self.until_frame = HOP;
            let frame = core::array::from_fn(|i| {
                self.samples[(self.next_sample + i) % FRAME] as f32 / 2048.0
            });
            let mut bins = [0.0; BINS];
            self.frontend.process(&frame, &mut bins);
            for (dest, value) in self.features[self.next_feature..self.next_feature + BINS]
                .iter_mut()
                .zip(bins)
            {
                *dest = quantize(value);
            }
            self.next_feature = (self.next_feature + BINS) % FEATURES;
        }
        self.until_prediction -= 1;
        if self.until_prediction == 0 {
            self.until_prediction = config::INFERENCE_HOP;
            true
        } else {
            false
        }
    }

    pub fn features(&self, output: &mut [i8; FEATURES]) {
        let (tail, head) = output.split_at_mut(FEATURES - self.next_feature);
        tail.copy_from_slice(&self.features[self.next_feature..]);
        head.copy_from_slice(&self.features[..self.next_feature]);
    }
}
