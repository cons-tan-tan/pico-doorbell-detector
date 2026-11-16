use portable_atomic::{AtomicU32, Ordering};

static BLOCKS_LEFT: AtomicU32 = AtomicU32::new(0);
const CAPTURE_BLOCKS: u32 = 1000;
pub const LINE_SAMPLES: usize = 64;

pub fn command(data: &[u8]) {
    for byte in data {
        match byte {
            b'r' => BLOCKS_LEFT.store(CAPTURE_BLOCKS, Ordering::Release),
            b's' => BLOCKS_LEFT.store(0, Ordering::Release),
            _ => (),
        }
    }
}

pub fn take_block() -> bool {
    BLOCKS_LEFT
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |left| {
            left.checked_sub(1)
        })
        .is_ok()
}

pub fn encode(samples: &[u16; LINE_SAMPLES]) -> [u8; LINE_SAMPLES * 3] {
    let mut bytes = [0; LINE_SAMPLES * 3];
    for (sample, digits) in samples.iter().zip(bytes.as_chunks_mut::<3>().0) {
        for (digit, shift) in digits.iter_mut().zip([8, 4, 0]) {
            *digit = b"0123456789abcdef"[((sample >> shift) & 15) as usize];
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_hex_preserves_every_adc_value_and_ignores_status_bits() {
        for start in (0..4096).step_by(LINE_SAMPLES) {
            let samples = core::array::from_fn(|i| (start + i) as u16 | 0xf000);
            let encoded = encode(&samples);
            for (i, digits) in encoded.as_chunks::<3>().0.iter().enumerate() {
                let text = core::str::from_utf8(digits).unwrap();
                assert_eq!(u16::from_str_radix(text, 16).unwrap(), (start + i) as u16);
            }
        }
    }

    #[test]
    fn capture_is_opt_in_bounded_and_can_be_stopped() {
        command(b"s");
        assert!(!take_block());
        command(b"r\n");
        for _ in 0..CAPTURE_BLOCKS {
            assert!(take_block());
        }
        assert!(!take_block());
        command(b"r");
        assert!(take_block());
        command(b"s\n");
        assert!(!take_block());
    }
}
