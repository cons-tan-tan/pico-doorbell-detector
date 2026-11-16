#![no_std]
#![no_main]

mod capture;
mod network;
mod settings;
mod usb;

use doorbell_firmware::{
    config,
    decision::{Chime, Runtime},
    diagnostic,
    frontend::{AudioWindow, BINS, FEATURES, FRAME, FrameFrontend, HOP, SAMPLE_RATE, quantize},
    levels::AudioLevels,
    model::Model,
    rules::Detector,
};
use embassy_rp::{
    bind_interrupts, dma,
    executor::Executor,
    multicore::{Stack, spawn_core1},
    peripherals::{DMA_CH0, DMA_CH1, PIO0, USB},
};
use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, channel::Channel};
use embassy_time::Instant;
use static_cell::{ConstStaticCell, StaticCell};

bind_interrupts!(struct Irqs {
    DMA_IRQ_0 => dma::InterruptHandler<DMA_CH0>, dma::InterruptHandler<DMA_CH1>;
    DMA_IRQ_1 => capture::InterruptHandler;
    PIO0_IRQ_0 => embassy_rp::pio::InterruptHandler<PIO0>;
    USBCTRL_IRQ => embassy_rp::usb::InterruptHandler<USB>;
});

static EVENTS: Channel<CriticalSectionRawMutex, Chime, 4> = Channel::new();
static mut CORE1_STACK: Stack<24576> = Stack::new();
static EXECUTOR0: StaticCell<Executor> = StaticCell::new();
static EXECUTOR1: StaticCell<Executor> = StaticCell::new();

#[cortex_m_rt::entry]
fn main() -> ! {
    let p = embassy_rp::init(Default::default());
    let settings = settings::load(
        embassy_rp::flash::Flash::new_blocking(p.FLASH),
        embassy_rp::trng::Trng::new_blocking(p.TRNG, Default::default()),
    );
    let logger = usb::init_logger(settings.public_key);
    log::info!("{}", settings.status);
    let id = embassy_rp::otp::get_chipid().unwrap().to_be_bytes();
    static SERIAL: StaticCell<[u8; 16]> = StaticCell::new();
    let serial = SERIAL.init([0; 16]);
    for (i, byte) in id.iter().enumerate() {
        serial[i * 2] = b"0123456789abcdef"[(byte >> 4) as usize];
        serial[i * 2 + 1] = b"0123456789abcdef"[(byte & 15) as usize];
    }
    let serial = core::str::from_utf8(serial).unwrap();
    spawn_core1(
        p.CORE1,
        unsafe { &mut *core::ptr::addr_of_mut!(CORE1_STACK) },
        || {
            EXECUTOR1
                .init(Executor::new())
                .run(|spawner| spawner.spawn(inference().unwrap()));
        },
    );
    capture::start(p.ADC, p.PIN_26, p.DMA_CH2, p.DMA_CH3, Irqs);
    EXECUTOR0.init(Executor::new()).run(|spawner| {
        spawner.spawn(usb::run(p.USB, serial, logger).unwrap());
        spawner.spawn(capture_statistics().unwrap());
        if let Some(credentials) = settings.credentials {
            spawner.spawn(
                network::run(
                    spawner,
                    p.PIO0,
                    (p.DMA_CH0, p.DMA_CH1),
                    network::Pins {
                        power: p.PIN_23,
                        cs: p.PIN_25,
                        data: p.PIN_24,
                        clock: p.PIN_29,
                    },
                    credentials,
                )
                .unwrap(),
            );
        }
    })
}

#[embassy_executor::task]
async fn capture_statistics() -> ! {
    let mut previous = capture::statistics();
    let mut started = Instant::now();
    loop {
        embassy_time::Timer::after_secs(10).await;
        let current = capture::statistics();
        let rate = current.samples.wrapping_sub(previous.samples) as u64 * 1000
            / started.elapsed().as_millis();
        log::info!(
            "ADC rate={} samples/s; lost blocks={}; queue full={}; clock={} Hz",
            rate,
            current.lost_blocks.wrapping_sub(previous.lost_blocks),
            current
                .queue_full_blocks
                .wrapping_sub(previous.queue_full_blocks),
            embassy_rp::clocks::clk_sys_freq()
        );
        previous = current;
        started = Instant::now();
    }
}

#[embassy_executor::task]
async fn inference() -> ! {
    log::info!("Model sha256={}", config::MODEL_SHA256);
    let mut model = Model::take().expect("TFLM initialization failed");
    let reference =
        include_bytes!(concat!(env!("DOORBELL_MODEL_DIR"), "/reference-input.i8")).map(|b| b as i8);
    let expected = include_bytes!(concat!(env!("DOORBELL_MODEL_DIR"), "/reference-output.i8"))
        .map(|b| b as i8);
    let reference_started = Instant::now();
    let actual = model.predict(&reference).unwrap();
    let reference_us = reference_started.elapsed().as_micros();
    assert_eq!(actual, expected, "TFLM reference mismatch");
    log::info!(
        "TFLM reference passed; arena={} bytes; inference={} us; clock={} Hz",
        model.arena_used(),
        reference_us,
        embassy_rp::clocks::clk_sys_freq()
    );
    let mut features = [0; FEATURES];
    let frontend_started = Instant::now();
    verify_frontend(&mut features);
    let frontend_us = frontend_started.elapsed().as_micros();
    let computed = model.predict(&features).unwrap();
    assert!(
        computed
            .into_iter()
            .zip(expected)
            .all(|(a, b)| (a as i16 - b as i16).abs() <= 1)
    );
    log::info!("Frontend reference passed; full window={} us", frontend_us);
    static WINDOW: ConstStaticCell<AudioWindow> = ConstStaticCell::new(AudioWindow::new());
    let window = WINDOW.take();
    let mut policy = Runtime::new(config::FUSION_POLICY);
    if let Some(settings) = config::FUSION_POLICY {
        log::info!(
            "Decision pattern_min={:.2}; ml_min={:.2}; confirmation_frames={}; minimum_support={}; rule_hold_ms={}; cooldown_ms={}",
            settings.low,
            settings.high,
            settings.confirmation_frames,
            settings.minimum_support,
            settings.rule_hold_ms,
            settings.cooldown_ms
        );
    } else {
        log::info!(
            "Decision threshold={:.2}; consecutive={}",
            config::THRESHOLD,
            config::CONSECUTIVE
        );
    }
    let mut patterns = Detector::tolerant(SAMPLE_RATE as f32);
    let mut frame = [0.0; FRAME];
    let mut frame_index = 0;
    let mut sample_clock = 0u64;
    let mut expected_sequence: Option<u32> = None;
    let mut frontend_us = 0;
    let mut processed_samples = 0;
    let mut levels = AudioLevels::default();
    loop {
        let block = capture::AUDIO.receive().await;
        if expected_sequence.is_some_and(|seq| seq != block.sequence) {
            sample_clock += block.sequence.wrapping_sub(expected_sequence.unwrap()) as u64
                * block.samples.len() as u64;
            log::warn!("Audio gap; discarding detection history");
            window.reset();
            policy.reset_history();
            patterns.reset();
            frame_index = 0;
            frontend_us = 0;
            processed_samples = 0;
            levels = AudioLevels::default();
        }
        expected_sequence = Some(block.sequence.wrapping_add(1));
        let mut frontend_started = Instant::now();
        if diagnostic::take_block() {
            for (part, samples) in block
                .samples
                .as_chunks::<{ diagnostic::LINE_SAMPLES }>()
                .0
                .iter()
                .enumerate()
            {
                let encoded = diagnostic::encode(samples);
                log::info!(
                    "PCM {},{}:{}",
                    block.sequence,
                    part,
                    core::str::from_utf8(&encoded).unwrap()
                );
            }
        }
        for sample in block.samples {
            sample_clock += 1;
            processed_samples += 1;
            levels.push(sample);
            if policy.uses_patterns() {
                frame[frame_index] = ((sample & 0x0fff) as f32 - 2048.0) / 2048.0;
                frame_index += 1;
                if frame_index == FRAME {
                    frame_index = 0;
                    if let Some(kind) = patterns.feed(&frame) {
                        let now_ms = sample_clock * 1000 / SAMPLE_RATE as u64;
                        log::info!("Pattern {:?}; t={} ms", kind, now_ms);
                        if let Some(event) = policy.observe_pattern(kind, now_ms) {
                            notify(event);
                        }
                    }
                }
            }
            if !window.push(sample) {
                continue;
            }
            window.features(&mut features);
            let features_finished = Instant::now();
            frontend_us += (features_finished - frontend_started).as_micros();
            let result = model.predict(&features).expect("TFLM inference failed");
            let finished = Instant::now();
            let inference_us = (finished - features_finished).as_micros();
            let elapsed_us = frontend_us + inference_us;
            let budget_us = processed_samples * 1_000_000 / SAMPLE_RATE as u64;
            log::info!(
                "Processing frontend={} us; inference={} us; total={} us; budget={} us; samples={}",
                frontend_us,
                inference_us,
                elapsed_us,
                budget_us,
                processed_samples
            );
            if elapsed_us > budget_us {
                log::warn!("Inference cadence exceeded");
            }
            let level = levels.take().unwrap();
            log::info!(
                "Audio samples={}; min={}; max={}; mean={:.1}; ac_rms={:.2}; rails={}; rail_pct={:.1}%",
                level.samples,
                level.min,
                level.max,
                level.mean,
                level.ac_rms,
                level.rails,
                level.rails as f32 * 100.0 / level.samples as f32
            );
            let scores = result.map(|score| {
                (score as i32 - config::OUTPUT_ZERO_POINT) as f32 * config::OUTPUT_SCALE * 100.0
            });
            log::info!(
                "Scores background={:.1}%; entrance={:.1}%; door={:.1}%",
                scores[0],
                scores[1],
                scores[2]
            );
            let now_ms = sample_clock * 1000 / SAMPLE_RATE as u64;
            if let Some(event) = policy.update(result, now_ms) {
                notify(event);
            }
            if let Runtime::Confirmed(state) = &policy {
                let status = state.status(now_ms);
                if scores[1].max(scores[2]) >= 50.0 || status.frames > 0 {
                    log::info!(
                        "Gate t={} ms; pattern={:?}; ml_support={:?}; frames={}; cooldown={}",
                        now_ms,
                        status.pattern,
                        status.support,
                        status.frames,
                        status.cooling_down
                    );
                }
            }
            frontend_us = 0;
            processed_samples = 0;
            frontend_started = Instant::now();
        }
        frontend_us += frontend_started.elapsed().as_micros();
    }
}

fn notify(event: Chime) {
    log::info!("Detected {:?}", event);
    if EVENTS.try_send(event).is_err() {
        log::warn!("Notification queue full");
    }
}

fn verify_frontend(output: &mut [i8; FEATURES]) {
    let mut frontend = FrameFrontend::new();
    let pcm = include_bytes!(concat!(env!("DOORBELL_MODEL_DIR"), "/reference-pcm.f32"))
        .as_chunks::<4>()
        .0;
    let expected = include_bytes!(concat!(
        env!("DOORBELL_MODEL_DIR"),
        "/reference-features.f32"
    ))
    .as_chunks::<4>()
    .0;
    for (index, target) in output.as_chunks_mut::<BINS>().0.iter_mut().enumerate() {
        let frame: [f32; FRAME] =
            core::array::from_fn(|i| f32::from_le_bytes(pcm[index * HOP + i]));
        let mut bins = [0.0; BINS];
        frontend.process(&frame, &mut bins);
        for (bin, (dest, value)) in target.iter_mut().zip(bins).enumerate() {
            assert!(
                (value - f32::from_le_bytes(expected[index * BINS + bin])).abs() < 2e-5,
                "frontend reference mismatch"
            );
            *dest = quantize(value);
        }
    }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    log::error!("{}", info);
    cortex_m::peripheral::SCB::sys_reset()
}
