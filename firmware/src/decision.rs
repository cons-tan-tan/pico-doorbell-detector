use crate::config;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chime {
    Entrance,
    Door,
}

pub enum Runtime {
    Threshold(Decision),
    Confirmed(crate::fusion::Decision),
}

impl Runtime {
    pub fn new(policy: Option<crate::fusion::Policy>) -> Self {
        match policy {
            Some(policy) => Self::Confirmed(crate::fusion::Decision::new(policy)),
            None => Self::Threshold(Decision::new()),
        }
    }

    pub fn uses_patterns(&self) -> bool {
        matches!(self, Self::Confirmed(_))
    }

    pub fn observe_pattern(&mut self, kind: Chime, now_ms: u64) -> Option<Chime> {
        match self {
            Self::Confirmed(state) => state.observe_pattern(kind, now_ms),
            Self::Threshold(_) => None,
        }
    }

    pub fn update(&mut self, scores: [i8; 3], now_ms: u64) -> Option<Chime> {
        match self {
            Self::Confirmed(state) => state.update(scores, now_ms),
            Self::Threshold(state) => state.update(scores, now_ms),
        }
    }

    pub fn reset_history(&mut self) {
        match self {
            Self::Confirmed(state) => state.reset_history(),
            Self::Threshold(state) => state.reset_history(),
        }
    }
}

pub struct Decision {
    candidate: Option<Chime>,
    count: usize,
    last_event_ms: Option<u64>,
}

impl Default for Decision {
    fn default() -> Self {
        Self::new()
    }
}

impl Decision {
    pub const fn new() -> Self {
        Self {
            candidate: None,
            count: 0,
            last_event_ms: None,
        }
    }

    pub fn reset_history(&mut self) {
        self.candidate = None;
        self.count = 0;
    }

    pub fn update(&mut self, scores: [i8; 3], now_ms: u64) -> Option<Chime> {
        let mut best = 0;
        for i in 1..3 {
            if scores[i] > scores[best] {
                best = i;
            }
        }
        let probability =
            (scores[best] as i32 - config::OUTPUT_ZERO_POINT) as f32 * config::OUTPUT_SCALE;
        if best == 0 || probability < config::THRESHOLD {
            self.reset_history();
            return None;
        }
        let chime = if best == 1 {
            Chime::Entrance
        } else {
            Chime::Door
        };
        if self.candidate == Some(chime) {
            self.count = self.count.saturating_add(1);
        } else {
            self.candidate = Some(chime);
            self.count = 1;
        }
        if self.count < config::CONSECUTIVE
            || self
                .last_event_ms
                .is_some_and(|last| now_ms.saturating_sub(last) < config::COOLDOWN_MS)
        {
            return None;
        }
        self.last_event_ms = Some(now_ms);
        Some(chime)
    }
}
