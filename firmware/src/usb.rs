use core::cell::RefCell;
use core::fmt::Write;
use doorbell_firmware::credentials::{Command, Commands as CommandParser};
use embassy_futures::join::join;
use embassy_rp::{Peri, peripherals::USB, usb::Driver};
use embassy_sync::blocking_mutex::{Mutex, raw::CriticalSectionRawMutex};
use embassy_usb::class::cdc_acm::device::{CdcAcmClass, Config as CdcConfig, State};
use embassy_usb_logger::{ReceiverHandler, UsbLogger};
use static_cell::StaticCell;

pub struct Commands {
    parser: Mutex<CriticalSectionRawMutex, RefCell<CommandParser>>,
    public_key: [u8; 32],
}

impl ReceiverHandler for Commands {
    async fn handle_data(&self, data: &[u8]) {
        for byte in data {
            let command = self.parser.lock(|parser| parser.borrow_mut().push(*byte));
            match command {
                Some(Command::Capture(byte)) => doorbell_firmware::diagnostic::command(&[byte]),
                Some(Command::Bootsel) => embassy_rp::rom_data::reset_to_usb_boot(0, 0),
                Some(Command::Info) => {
                    let mut public_key = heapless::String::<64>::new();
                    doorbell_firmware::credentials::write_hex(&self.public_key, &mut public_key)
                        .unwrap();
                    log::info!("WIFI PUBLIC KEY {}", public_key);
                }
                Some(Command::Invalid) => log::info!("Unknown USB command"),
                None => (),
            }
        }
    }

    fn new() -> Self {
        Self {
            parser: Mutex::new(RefCell::new(CommandParser::default())),
            public_key: [0; 32],
        }
    }
}

pub fn init_logger(public_key: [u8; 32]) -> &'static UsbLogger<8192, Commands> {
    static LOGGER: StaticCell<UsbLogger<8192, Commands>> = StaticCell::new();
    let logger = LOGGER.init(UsbLogger::with_custom_style(|record, writer| {
        // Keep PCM records intact when the other core logs ADC or network activity.
        critical_section::with(|_| {
            if record.level() <= log::Level::Warn {
                let _ = write!(writer, "[{}] ", record.level());
            }
            let _ = write!(writer, "{}\r\n", record.args());
        });
    }));
    let mut commands = Commands::new();
    commands.public_key = public_key;
    logger.with_handler(commands);
    log::set_logger(logger).unwrap();
    log::set_max_level(log::LevelFilter::Info);
    logger
}

#[embassy_executor::task]
pub async fn run(
    usb: Peri<'static, USB>,
    serial: &'static str,
    logger: &'static UsbLogger<8192, Commands>,
) -> ! {
    let driver = Driver::new(usb, crate::Irqs);
    let mut config = embassy_usb::Config::new(0x2e8a, 0x000a);
    config.manufacturer = Some("Raspberry Pi");
    config.product = Some("Pico 2 W");
    config.serial_number = Some(serial);
    config.max_power = 100;
    config.max_packet_size_0 = 64;
    let mut descriptor = [0; 256];
    let mut bos = [0; 64];
    let mut control = [0; 64];
    let mut state = State::new();
    let mut builder = embassy_usb::Builder::new(
        driver,
        config,
        &mut descriptor,
        &mut bos,
        &mut [],
        &mut control,
    );
    let class = CdcAcmClass::new(&mut builder, &mut state, CdcConfig::new());
    let mut device = builder.build();
    join(device.run(), logger.create_future_from_class(class)).await;
    unreachable!()
}
