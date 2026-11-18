use cyw43::{JoinOptions, aligned_bytes};
use cyw43_pio::{PioSpi, RM2_CLOCK_DIVIDER};
use doorbell_firmware::{credentials::Credentials, decision::Chime, notification};
use embassy_executor::Spawner;
use embassy_futures::select::{Either, select};
use embassy_net::{
    Stack, StackStorage, StaticPool,
    route::Route,
    tcp::TcpSocket,
    wire::{IpAddr, Ipv4Addr, Ipv4Cidr},
};
use embassy_rp::{
    Peri,
    clocks::RoscRng,
    dma,
    gpio::{Level, Output},
    peripherals::{DMA_CH0, DMA_CH1, PIN_23, PIN_24, PIN_25, PIN_29, PIO0},
    pio::Pio,
};
use embassy_time::{Duration, Timer, with_timeout};
use embedded_io_async::Write;
use heapless::String;
use static_cell::{ConstStaticCell, StaticCell};

const ADDRESS: Ipv4Addr = Ipv4Addr::new(192, 168, 1, 101);
const GATEWAY: Ipv4Addr = Ipv4Addr::new(192, 168, 0, 1);
const DNS: Ipv4Addr = Ipv4Addr::new(192, 168, 0, 1);

pub struct Pins {
    pub power: Peri<'static, PIN_23>,
    pub cs: Peri<'static, PIN_25>,
    pub data: Peri<'static, PIN_24>,
    pub clock: Peri<'static, PIN_29>,
}

#[embassy_executor::task]
async fn radio(
    runner: cyw43::Runner<
        'static,
        cyw43::SpiBus<Output<'static>, PioSpi<'static, PIO0, 0>>,
        cyw43::Cyw43439,
    >,
) -> ! {
    runner.run().await
}

#[embassy_executor::task]
async fn stack_runner(mut runner: embassy_net::Runner<'static>) -> ! {
    runner.run().await
}

#[embassy_executor::task]
pub async fn run(
    spawner: Spawner,
    pio: Peri<'static, PIO0>,
    dma_channels: (Peri<'static, DMA_CH0>, Peri<'static, DMA_CH1>),
    pins: Pins,
    credentials: Credentials,
) -> ! {
    let Pins {
        power,
        cs,
        data,
        clock,
    } = pins;
    let fw = aligned_bytes!(concat!(env!("CYW43_FIRMWARE_DIR"), "/43439A0.bin"));
    let clm = aligned_bytes!(concat!(env!("CYW43_FIRMWARE_DIR"), "/43439A0_clm.bin"));
    let nvram = aligned_bytes!(concat!(env!("CYW43_FIRMWARE_DIR"), "/nvram_rp2040.bin"));
    let power = Output::new(power, Level::Low);
    let cs = Output::new(cs, Level::High);
    let mut pio = Pio::new(pio, crate::Irqs);
    let spi = PioSpi::new(
        &mut pio.common,
        pio.sm0,
        RM2_CLOCK_DIVIDER,
        pio.irq0,
        cs,
        data,
        clock,
        dma::Channel::new(dma_channels.0, crate::Irqs),
        dma::Channel::new(dma_channels.1, crate::Irqs),
    );
    static STATE: ConstStaticCell<cyw43::State> = ConstStaticCell::new(cyw43::State::new());
    let (device, mut control, runner) = cyw43::new(STATE.take(), power, spi, fw, nvram).await;
    spawner.spawn(radio(runner).unwrap());
    control.init(clm, cyw43::Config::default()).await;
    control
        .set_power_management(cyw43::PowerManagementMode::None)
        .await;
    static STORAGE: StaticCell<StackStorage> = StaticCell::new();
    static POOL: StaticPool<{ cyw43::RX_BUF_SIZE }, 8, { cyw43::RX_BUF_ALIGN }> = StaticPool::new();
    let (stack, runner) = Stack::new(STORAGE.init(StackStorage::new()), &POOL, RoscRng.next_u64());
    static DEVICE: StaticCell<cyw43::NetDriver<'static>> = StaticCell::new();
    let iface = stack.add_iface_borrowed(DEVICE.init(device)).unwrap();
    iface
        .add_ip_addr(Ipv4Cidr::new(ADDRESS, 20).into())
        .unwrap();
    stack
        .routes()
        .add(Route::new_ipv4_gateway(GATEWAY, iface.handle()))
        .unwrap();
    stack.set_dns_servers(&[IpAddr::V4(DNS)]);
    spawner.spawn(stack_runner(runner).unwrap());
    loop {
        if !iface.is_link_up() {
            match with_timeout(
                Duration::from_secs(20),
                control.join(credentials.ssid(), JoinOptions::new(credentials.password())),
            )
            .await
            {
                Ok(Ok(())) => log::info!("Wi-Fi connected"),
                result => {
                    log::warn!("Wi-Fi join failed: {:?}", result);
                    Timer::after_secs(5).await;
                    continue;
                }
            }
        }
        match select(crate::EVENTS.receive(), Timer::after_secs(2)).await {
            Either::Second(_) => continue,
            Either::First(event) => {
                control.gpio_set(0, true).await;
                for attempt in 0..3 {
                    match with_timeout(Duration::from_secs(15), send(stack, event)).await {
                        Ok(Ok(())) => {
                            log::info!("Notification delivered");
                            break;
                        }
                        result => {
                            log::warn!("Notification attempt {} failed: {:?}", attempt + 1, result)
                        }
                    }
                    if !iface.is_link_up() {
                        break;
                    }
                    Timer::after_secs(1).await;
                }
                control.gpio_set(0, false).await;
            }
        }
    }
}

async fn send(stack: Stack<'_>, chime: Chime) -> Result<(), &'static str> {
    let addresses = stack
        .dns_query(notification::HOST, embassy_net::dns::DnsQueryType::A)
        .await
        .map_err(|_| "DNS lookup failed")?;
    let address = *addresses.first().ok_or("DNS returned no address")?;
    let mut rx = [0; 2048];
    let mut tx = [0; 2048];
    let mut socket =
        TcpSocket::new(stack, &mut rx, &mut tx).map_err(|_| "No TCP socket available")?;
    socket.set_timeout(Some(Duration::from_secs(10)));
    socket
        .connect((address, 80))
        .await
        .map_err(|_| "TCP connect failed")?;
    let mut request = String::<512>::new();
    notification::request(&mut request, env!("DOORBELL_TOPIC"), chime)
        .map_err(|_| "HTTP request too long")?;
    socket
        .write_all(request.as_bytes())
        .await
        .map_err(|_| "TCP write failed")?;
    // Timestamp the local send before flush waits for the peer's TCP ACK.
    log::info!("Notification request sent");
    socket.flush().await.map_err(|_| "TCP flush failed")?;
    let mut line = [0; 256];
    let mut length = 0;
    loop {
        let n = socket
            .read(&mut line[length..])
            .await
            .map_err(|_| "TCP read failed")?;
        if n == 0 {
            return Err("HTTP response closed before status");
        }
        length += n;
        if let Some(end) = line[..length].windows(2).position(|b| b == b"\r\n") {
            let status = notification::status(&line[..end + 2]).ok_or("Invalid HTTP status")?;
            socket.close();
            if (200..300).contains(&status) {
                return Ok(());
            }
            log::warn!("ntfy returned HTTP {}", status);
            return Err("ntfy rejected notification");
        }
        if length == line.len() {
            return Err("HTTP status line too long");
        }
    }
}
