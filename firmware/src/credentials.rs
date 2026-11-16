use blake2::{
    Blake2b, Digest,
    digest::consts::{U24, U32},
};
use crypto_box::{PublicKey, SalsaBox, SecretKey, aead::AeadInPlace};
use zeroize::Zeroizing;

pub const PLAINTEXT_LEN: usize = 100;
pub const SEALED_LEN: usize = PLAINTEXT_LEN + 48;
pub const RECORD_LEN: usize = 256;
pub const FLASH_SIZE: usize = 4 * 1024 * 1024;
pub const SECTOR_SIZE: u32 = 4096;
pub const KEY_OFFSET: u32 = FLASH_SIZE as u32 - SECTOR_SIZE;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    InvalidCredentials,
    InvalidCiphertext,
    InvalidRecord,
}

pub struct Credentials(Zeroizing<[u8; PLAINTEXT_LEN]>);

impl Credentials {
    pub fn new(ssid: &str, password: &str) -> Result<Self, Error> {
        if ssid.is_empty() || ssid.len() > 32 || password.len() > 64 {
            return Err(Error::InvalidCredentials);
        }
        let mut bytes = Zeroizing::new([0; PLAINTEXT_LEN]);
        bytes[0] = 1;
        bytes[1] = ssid.len() as u8;
        bytes[2] = password.len() as u8;
        bytes[4..4 + ssid.len()].copy_from_slice(ssid.as_bytes());
        bytes[36..36 + password.len()].copy_from_slice(password.as_bytes());
        Self::decode(bytes)
    }

    fn decode(bytes: Zeroizing<[u8; PLAINTEXT_LEN]>) -> Result<Self, Error> {
        let ssid_len = bytes[1] as usize;
        let password_len = bytes[2] as usize;
        if bytes[0] != 1 || bytes[3] != 0 || !(1..=32).contains(&ssid_len) || password_len > 64 {
            return Err(Error::InvalidCredentials);
        }
        let ssid = &bytes[4..4 + ssid_len];
        let password = &bytes[36..36 + password_len];
        if core::str::from_utf8(ssid).is_err()
            || core::str::from_utf8(password).is_err()
            || !(8..=63).contains(&password_len)
                && !(password_len == 64 && password.iter().all(u8::is_ascii_hexdigit))
            || bytes[4 + ssid_len..36].iter().any(|b| *b != 0)
            || bytes[36 + password_len..].iter().any(|b| *b != 0)
        {
            return Err(Error::InvalidCredentials);
        }
        Ok(Self(bytes))
    }

    pub fn ssid(&self) -> &str {
        core::str::from_utf8(&self.0[4..4 + self.0[1] as usize]).unwrap()
    }

    pub fn password(&self) -> &[u8] {
        &self.0[36..36 + self.0[2] as usize]
    }

    pub fn as_bytes(&self) -> &[u8; PLAINTEXT_LEN] {
        &self.0
    }
}

pub fn open(key: &SecretKey, sealed: &[u8; SEALED_LEN]) -> Result<Credentials, Error> {
    let sender = PublicKey::from(<[u8; 32]>::try_from(&sealed[..32]).unwrap());
    let mut nonce = Blake2b::<U24>::new();
    nonce.update(sender.as_bytes());
    nonce.update(key.public_key().as_bytes());
    let nonce = nonce.finalize();
    let mut plaintext = Zeroizing::new([0; PLAINTEXT_LEN]);
    plaintext.copy_from_slice(&sealed[48..]);
    // Libsodium sealed boxes put the authentication tag before the ciphertext.
    SalsaBox::new(&sender, key)
        .decrypt_in_place_detached(&nonce, &[], &mut *plaintext, sealed[32..48].into())
        .map_err(|_| Error::InvalidCiphertext)?;
    Credentials::decode(plaintext)
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Blake2b::<U32>::digest(bytes).into()
}

pub fn key_record(key: &SecretKey) -> Zeroizing<[u8; RECORD_LEN]> {
    let mut record = Zeroizing::new([0xff; RECORD_LEN]);
    record[..8].copy_from_slice(b"DBKEY001");
    let bytes = Zeroizing::new(key.to_bytes());
    record[8..40].copy_from_slice(&*bytes);
    let checksum = digest(&record[..40]);
    record[40..72].copy_from_slice(&checksum);
    record
}

pub fn read_key(record: &[u8; RECORD_LEN]) -> Result<SecretKey, Error> {
    if &record[..8] != b"DBKEY001" || digest(&record[..40]) != record[40..72] {
        return Err(Error::InvalidRecord);
    }
    Ok(SecretKey::from(
        <[u8; 32]>::try_from(&record[8..40]).unwrap(),
    ))
}

pub fn decode_hex<const N: usize>(hex: &[u8]) -> Option<[u8; N]> {
    if hex.len() != N * 2 {
        return None;
    }
    let mut output = [0; N];
    for (byte, pair) in output.iter_mut().zip(hex.as_chunks::<2>().0) {
        let digit = |b: u8| match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b - b'a' + 10),
            _ => None,
        };
        *byte = digit(pair[0])? * 16 + digit(pair[1])?;
    }
    Some(output)
}

pub fn write_hex(bytes: &[u8], output: &mut impl core::fmt::Write) -> core::fmt::Result {
    for byte in bytes {
        write!(output, "{byte:02x}")?;
    }
    Ok(())
}

pub enum Command {
    Info,
    Capture(u8),
    Invalid,
}

#[derive(Default)]
pub struct Commands {
    line: [u8; 32],
    length: usize,
    overflow: bool,
}

impl Commands {
    pub fn push(&mut self, byte: u8) -> Option<Command> {
        if self.length == 0 && !self.overflow && matches!(byte, b'r' | b's') {
            return Some(Command::Capture(byte));
        }
        if !matches!(byte, b'\r' | b'\n') {
            if self.length < self.line.len() {
                self.line[self.length] = byte;
                self.length += 1;
            } else {
                self.overflow = true;
            }
            return None;
        }
        let command = if self.overflow {
            Command::Invalid
        } else {
            match &self.line[..self.length] {
                b"WIFI INFO" => Command::Info,
                b"" => {
                    return None;
                }
                _ => Command::Invalid,
            }
        };
        self.length = 0;
        self.overflow = false;
        Some(command)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> [u8; SEALED_LEN] {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/wifi.json")).unwrap();
        decode_hex(fixture["ciphertext"].as_str().unwrap().as_bytes()).unwrap()
    }

    #[test]
    fn opens_libsodium_sealed_box() {
        let key = SecretKey::from([7; 32]);
        let credentials = open(&key, &fixture()).unwrap();
        assert_eq!(credentials.ssid(), "unit-test-network");
        assert_eq!(credentials.password(), b"unit-test-password");
    }

    #[test]
    fn rejects_wrong_recipient_and_corrupted_ciphertext() {
        let sealed = fixture();
        assert!(open(&SecretKey::from([8; 32]), &sealed).is_err());
        let key = SecretKey::from([7; 32]);
        for index in 0..SEALED_LEN {
            let mut corrupted = sealed;
            corrupted[index] ^= 1;
            assert!(
                open(&key, &corrupted).is_err(),
                "accepted corruption at {index}"
            );
        }
    }

    #[test]
    fn incomplete_or_corrupted_key_records_are_rejected() {
        let key = SecretKey::from([7; 32]);
        let record = key_record(&key);
        assert_eq!(read_key(&record).unwrap().public_key(), key.public_key());
        for length in 0..72 {
            let mut torn = [0xff; RECORD_LEN];
            torn[..length].copy_from_slice(&record[..length]);
            assert!(read_key(&torn).is_err());
            let mut corrupted = *record;
            corrupted[length] ^= 1;
            assert!(read_key(&corrupted).is_err());
        }
    }

    #[test]
    fn validates_byte_lengths_and_wpa2_passwords() {
        assert!(Credentials::new("", "12345678").is_err());
        assert!(Credentials::new("wifi", "1234567").is_err());
        assert!(Credentials::new(&"あ".repeat(11), "12345678").is_err());
        assert!(Credentials::new("wifi", &"g".repeat(64)).is_err());
        assert!(Credentials::new("wifi", &"a".repeat(64)).is_ok());
        let valid = Credentials::new("玄関", "password-for-test").unwrap();
        assert_eq!(valid.ssid(), "玄関");
        let mut malformed = Zeroizing::new(*valid.as_bytes());
        malformed[1] = 255;
        assert!(Credentials::decode(malformed).is_err());
    }

    #[test]
    fn frames_usb_commands_and_retains_pcm_controls() {
        let mut parser = Commands::default();
        assert!(matches!(parser.push(b'r'), Some(Command::Capture(b'r'))));
        assert!(parser.push(b'\n').is_none());
        for byte in b"WIFI INFO" {
            assert!(parser.push(*byte).is_none());
        }
        assert!(matches!(parser.push(b'\r'), Some(Command::Info)));
        assert!(parser.push(b'\n').is_none());
        for _ in 0..1000 {
            assert!(parser.push(b'X').is_none());
        }
        assert!(matches!(parser.push(b'\n'), Some(Command::Invalid)));
        assert!(matches!(parser.push(b's'), Some(Command::Capture(b's'))));
        for byte in b"WIFI INFO" {
            assert!(parser.push(*byte).is_none());
        }
        assert!(matches!(parser.push(b'\n'), Some(Command::Info)));
    }
}
