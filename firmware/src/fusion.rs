use crate::{config, decision::Chime};

#[derive(Clone, Copy)]
pub struct Policy {
    pub low: f32,
    pub high: f32,
    pub confirmation_frames: usize,
    pub minimum_support: usize,
    pub rule_hold_ms: u64,
    pub cooldown_ms: u64,
    pub use_rules: bool,
}

impl Policy {
    pub const fn with_minimum_support(mut self, count: usize) -> Self {
        self.minimum_support = count;
        self
    }

    pub const fn with_rule_hold_ms(mut self, milliseconds: u64) -> Self {
        self.rule_hold_ms = milliseconds;
        self
    }

    pub const fn new(low: f32, high: f32, confirmation_frames: usize, use_rules: bool) -> Self {
        Self {
            low,
            high,
            confirmation_frames,
            use_rules,
            minimum_support: 2,
            rule_hold_ms: 768,
            cooldown_ms: config::COOLDOWN_MS,
        }
    }
}

struct Pending {
    frames: usize,
    support: [usize; 2],
    strength: [f32; 2],
}

#[derive(Debug, PartialEq, Eq)]
pub struct Status {
    pub pattern: Option<Chime>,
    pub frames: usize,
    pub support: [usize; 2],
    pub cooling_down: bool,
}

pub struct Decision {
    policy: Policy,
    last_rule: Option<(Chime, u64)>,
    last_prediction: Option<([i8; 3], u64)>,
    last_event_ms: Option<u64>,
    pending: Option<Pending>,
}

impl Decision {
    pub fn new(policy: Policy) -> Self {
        assert!(policy.low > 0.0 && policy.low <= policy.high && policy.high <= 1.0);
        assert!(
            policy.minimum_support >= 2 && policy.minimum_support <= policy.confirmation_frames
        );
        assert!(policy.confirmation_frames <= 8);
        Self {
            policy,
            last_rule: None,
            last_prediction: None,
            last_event_ms: None,
            pending: None,
        }
    }

    pub fn observe_pattern(&mut self, chime: Chime, now_ms: u64) -> Option<Chime> {
        self.last_rule = Some((chime, now_ms));
        if let Some((scores, time)) = self.last_prediction
            && now_ms.saturating_sub(time) <= (config::INFERENCE_HOP * 1000 / 8000) as u64
        {
            return self.confirm_pattern(scores, now_ms);
        }
        None
    }

    pub fn reset_history(&mut self) {
        self.last_rule = None;
        self.last_prediction = None;
        self.pending = None;
    }

    fn cooling_down(&self, now_ms: u64) -> bool {
        self.last_event_ms
            .is_some_and(|last| now_ms.saturating_sub(last) < self.policy.cooldown_ms)
    }

    fn active_pattern(&self, now_ms: u64) -> Option<Chime> {
        self.last_rule
            .filter(|(_, time)| {
                self.policy.use_rules && now_ms.saturating_sub(*time) <= self.policy.rule_hold_ms
            })
            .map(|(kind, _)| kind)
    }

    pub fn status(&self, now_ms: u64) -> Status {
        Status {
            pattern: self.active_pattern(now_ms),
            frames: self.pending.as_ref().map_or(0, |pending| pending.frames),
            support: self
                .pending
                .as_ref()
                .map_or([0; 2], |pending| pending.support),
            cooling_down: self.cooling_down(now_ms),
        }
    }

    fn confirm_pattern(&mut self, scores: [i8; 3], now_ms: u64) -> Option<Chime> {
        if self.cooling_down(now_ms) {
            return None;
        }
        let kind = self.active_pattern(now_ms)?;
        let index = match kind {
            Chime::Entrance => 1,
            Chime::Door => 2,
        };
        let score =
            (scores[index] as i32 - config::OUTPUT_ZERO_POINT) as f32 * config::OUTPUT_SCALE;
        if score < self.policy.low {
            return None;
        }
        self.last_event_ms = Some(now_ms);
        self.pending = None;
        Some(kind)
    }

    pub fn update(&mut self, scores: [i8; 3], now_ms: u64) -> Option<Chime> {
        self.last_prediction = Some((scores, now_ms));
        if self.cooling_down(now_ms) {
            self.pending = None;
            return None;
        }
        if let Some(chime) = self.confirm_pattern(scores, now_ms) {
            return Some(chime);
        }
        let probabilities =
            scores.map(|x| (x as i32 - config::OUTPUT_ZERO_POINT) as f32 * config::OUTPUT_SCALE);
        let chimes = [probabilities[1], probabilities[2]];
        let supported = chimes[0].max(chimes[1]) >= self.policy.high;
        if self.pending.is_none() && !supported {
            return None;
        }
        let pending = self.pending.get_or_insert(Pending {
            frames: 0,
            support: [0; 2],
            strength: [0.0; 2],
        });
        pending.frames += 1;
        for (index, value) in chimes.into_iter().enumerate() {
            pending.support[index] += usize::from(value >= self.policy.high);
            pending.strength[index] += value;
        }
        if pending.frames < self.policy.confirmation_frames {
            return None;
        }
        let pending = self.pending.take().unwrap();
        let eligible = pending
            .support
            .map(|count| count >= self.policy.minimum_support);
        if !eligible.iter().any(|value| *value) {
            return None;
        }
        self.last_event_ms = Some(now_ms);
        Some(
            if eligible[0] && (!eligible[1] || pending.strength[0] >= pending.strength[1]) {
                Chime::Entrance
            } else {
                Chime::Door
            },
        )
    }
}
