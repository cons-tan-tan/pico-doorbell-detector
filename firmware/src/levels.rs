#[derive(Default)]
pub struct AudioLevels {
    count: u32,
    sum: u64,
    sum_squares: u64,
    min: u16,
    max: u16,
    rails: u32,
}

pub struct LevelReport {
    pub samples: u32,
    pub min: u16,
    pub max: u16,
    pub mean: f32,
    pub ac_rms: f32,
    pub rails: u32,
}

impl AudioLevels {
    pub fn push(&mut self, adc: u16) {
        debug_assert!(self.count < crate::frontend::WINDOW as u32);
        let sample = adc & 0x0fff;
        if self.count == 0 {
            self.min = sample;
        }
        self.count += 1;
        self.sum += sample as u64;
        self.sum_squares += sample as u64 * sample as u64;
        self.min = self.min.min(sample);
        self.max = self.max.max(sample);
        self.rails += u32::from(sample == 0 || sample == 4095);
    }

    pub fn take(&mut self) -> Option<LevelReport> {
        if self.count == 0 {
            return None;
        }
        let count = self.count as u64;
        // Subtract before converting to float to retain small signals on a large DC bias.
        let variance =
            (self.sum_squares * count - self.sum * self.sum) as f32 / (count * count) as f32;
        let report = LevelReport {
            samples: self.count,
            min: self.min,
            max: self.max,
            mean: self.sum as f32 / self.count as f32,
            ac_rms: libm::sqrtf(variance),
            rails: self.rails,
        };
        *self = Self::default();
        Some(report)
    }
}
