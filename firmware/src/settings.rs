use crypto_box::SecretKey;
use doorbell_firmware::credentials::{
    self, Credentials, FLASH_SIZE, KEY_OFFSET, RECORD_LEN, SECTOR_SIZE,
};
use embassy_rp::{flash::Flash, mode::Blocking, trng::Trng};
use zeroize::Zeroizing;

pub struct Settings {
    pub public_key: [u8; 32],
    pub credentials: Option<Credentials>,
    pub status: &'static str,
}

pub fn load(
    mut flash: Flash<'static, Blocking, FLASH_SIZE>,
    mut trng: Trng<'static, Blocking>,
) -> Settings {
    let mut record = Zeroizing::new([0; RECORD_LEN]);
    flash
        .blocking_read(KEY_OFFSET, &mut *record)
        .expect("Device key read failed");
    let key = match credentials::read_key(&record) {
        Ok(key) => key,
        Err(_) => {
            let mut random = Zeroizing::new([0; 32]);
            trng.blocking_fill_bytes(&mut *random);
            let key = SecretKey::from(*random);
            let encoded = credentials::key_record(&key);
            flash
                .blocking_erase(KEY_OFFSET, KEY_OFFSET + SECTOR_SIZE)
                .expect("Device key erase failed");
            flash
                .blocking_write(KEY_OFFSET, &*encoded)
                .expect("Device key write failed");
            flash
                .blocking_read(KEY_OFFSET, &mut *record)
                .expect("Device key read failed");
            assert!(*record == *encoded, "Device key verification failed");
            key
        }
    };
    let public_key = *key.public_key().as_bytes();
    let sealed = include_bytes!(concat!(env!("OUT_DIR"), "/wifi.sealed"));
    let (credentials, status) = if sealed.is_empty() {
        (
            None,
            "Wi-Fi configuration absent; encrypt settings with this device's public key",
        )
    } else {
        match credentials::open(
            &key,
            sealed
                .as_slice()
                .try_into()
                .expect("Invalid Wi-Fi ciphertext size"),
        ) {
            Ok(credentials) => (Some(credentials), "Wi-Fi configuration decrypted"),
            Err(_) => (
                None,
                "Wi-Fi configuration rejected; wrong device key or damaged settings",
            ),
        }
    };
    Settings {
        public_key,
        credentials,
        status,
    }
}
