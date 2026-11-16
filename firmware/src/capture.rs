use core::{
    cell::UnsafeCell,
    ptr,
    sync::atomic::{Ordering, compiler_fence},
};
use embassy_rp::{
    Peri,
    adc::{Adc, Channel, Config},
    dma,
    gpio::Pull,
    interrupt::{self, InterruptExt},
    pac,
    peripherals::{ADC, DMA_CH2, DMA_CH3, PIN_26},
};
use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, channel::Channel as Queue};
use portable_atomic::AtomicU32;

pub const CHUNK: usize = 512;
pub struct AudioBlock {
    pub sequence: u32,
    pub samples: [u16; CHUNK],
}
pub static AUDIO: Queue<CriticalSectionRawMutex, AudioBlock, 8> = Queue::new();

#[repr(align(1024))]
struct Buffers(UnsafeCell<[[u16; CHUNK]; 2]>);
// DMA writes through raw pointers; only DMA_IRQ_1 reads a completed buffer, using volatile loads.
unsafe impl Sync for Buffers {}
static BUFFERS: Buffers = Buffers(UnsafeCell::new([[0; CHUNK]; 2]));
struct Sequence(UnsafeCell<u32>);
// Only DMA_IRQ_1 accesses this counter after initialization.
unsafe impl Sync for Sequence {}
static SEQUENCE: Sequence = Sequence(UnsafeCell::new(0));
const CHANNELS: u32 = (1 << 2) | (1 << 3);
static SAMPLES: AtomicU32 = AtomicU32::new(0);
static LOST_BLOCKS: AtomicU32 = AtomicU32::new(0);
static QUEUE_FULL_BLOCKS: AtomicU32 = AtomicU32::new(0);

pub struct Statistics {
    pub samples: u32,
    pub lost_blocks: u32,
    pub queue_full_blocks: u32,
}

pub fn statistics() -> Statistics {
    Statistics {
        samples: SAMPLES.load(Ordering::Relaxed),
        lost_blocks: LOST_BLOCKS.load(Ordering::Relaxed),
        queue_full_blocks: QUEUE_FULL_BLOCKS.load(Ordering::Relaxed),
    }
}

pub fn start(
    adc: Peri<'static, ADC>,
    pin: Peri<'static, PIN_26>,
    first: Peri<'static, DMA_CH2>,
    second: Peri<'static, DMA_CH3>,
    _irq: impl interrupt::typelevel::Binding<interrupt::typelevel::DMA_IRQ_1, InterruptHandler>,
) {
    let adc = Adc::new_blocking(adc, Config::default());
    let pin = Channel::new_pin(pin, Pull::None);
    let first = dma::Channel::new_no_interrupt(first);
    let second = dma::Channel::new_no_interrupt(second);
    // The circular DMA stream owns these peripherals until the next hardware reset.
    core::mem::forget((adc, pin, first, second));
    interrupt::DMA_IRQ_1.set_priority(interrupt::Priority::P1);
    unsafe {
        restart();
        pac::DMA.inte(1).write_value(CHANNELS);
        interrupt::DMA_IRQ_1.enable();
    }
}

fn address(index: usize) -> u32 {
    BUFFERS.0.get().cast::<u16>().wrapping_add(index * CHUNK) as u32
}

unsafe fn restart() {
    pac::ADC.cs().modify(|w| w.set_start_many(false));
    while !pac::ADC.cs().read().ready() {}
    pac::DMA
        .chan_abort()
        .write_value(pac::dma::regs::ChanAbort(CHANNELS));
    while pac::DMA.chan_abort().read().0 & CHANNELS != 0 {}
    pac::DMA.ints(1).write_value(CHANNELS);
    while !pac::ADC.fcs().read().empty() {
        pac::ADC.fifo().read();
    }
    pac::ADC.cs().modify(|w| {
        w.set_ainsel(0);
        w.set_rrobin(0);
        w.set_err_sticky(true);
    });
    pac::ADC.div().write(|w| {
        w.set_int(5999);
        w.set_frac(0);
    });
    pac::ADC.fcs().write(|w| {
        w.set_en(true);
        w.set_dreq_en(true);
        w.set_thresh(1);
        w.set_err(true);
        w.set_over(true);
        w.set_under(true);
    });
    for index in 0..2 {
        let channel = pac::DMA.ch(index + 2);
        channel
            .read_addr()
            .write_value(pac::ADC.fifo().as_ptr() as u32);
        channel.write_addr().write_value(address(index));
        channel.trans_count().write(|w| {
            w.set_count(CHUNK as u32);
            w.set_mode(pac::dma::vals::TransCountMode::Normal);
        });
        channel.al1_ctrl().write(|w| {
            w.set_data_size(pac::dma::vals::DataSize::SizeHalfword);
            w.set_incr_write(true);
            w.set_treq_sel(pac::dma::vals::TreqSel::Adc);
            w.set_ring_sel(true);
            w.set_ring_size(10);
            w.set_chain_to((3 - index) as u8);
            w.set_en(true);
        });
    }
    compiler_fence(Ordering::SeqCst);
    pac::DMA
        .multi_chan_trigger()
        .write_value(pac::dma::regs::MultiChanTrigger(1 << 2));
    pac::ADC.cs().modify(|w| w.set_start_many(true));
}

pub struct InterruptHandler;
impl interrupt::typelevel::Handler<interrupt::typelevel::DMA_IRQ_1> for InterruptHandler {
    unsafe fn on_interrupt() {
        let pending = pac::DMA.ints(1).read() & CHANNELS;
        if pending == 0 {
            return;
        }
        pac::DMA.ints(1).write_value(pending);
        SAMPLES.fetch_add(pending.count_ones() * CHUNK as u32, Ordering::Relaxed);
        if (2..4).any(|channel| pac::DMA.ch(channel).ctrl_trig().read().ahb_error()) {
            log::error!("ADC DMA bus error");
            cortex_m::peripheral::SCB::sys_reset();
        }
        let sequence = unsafe { &mut *SEQUENCE.0.get() };
        let index = if pending & (1 << 2) != 0 { 0 } else { 1 };
        let other = pac::DMA.ch(3 - index);
        if pending == CHANNELS || !other.ctrl_trig().read().busy() || pac::ADC.fcs().read().over() {
            *sequence = sequence.wrapping_add(2);
            LOST_BLOCKS.fetch_add(2, Ordering::Relaxed);
            unsafe {
                restart();
            }
            return;
        }
        let mut block = AudioBlock {
            sequence: *sequence,
            samples: [0; CHUNK],
        };
        *sequence = sequence.wrapping_add(1);
        compiler_fence(Ordering::SeqCst);
        let source = address(index) as *const u16;
        for (offset, sample) in block.samples.iter_mut().enumerate() {
            *sample = unsafe { ptr::read_volatile(source.add(offset)) };
        }
        let valid = block.samples.iter().all(|s| s & 0x8000 == 0);
        compiler_fence(Ordering::SeqCst);
        if pac::DMA.ch(index + 2).ctrl_trig().read().busy()
            || pac::DMA.ints(1).read() & CHANNELS != 0
        {
            *sequence = sequence.wrapping_add(2);
            LOST_BLOCKS.fetch_add(2, Ordering::Relaxed);
            unsafe {
                restart();
            }
            return;
        }
        if !valid {
            LOST_BLOCKS.fetch_add(1, Ordering::Relaxed);
        } else if AUDIO.try_send(block).is_err() {
            QUEUE_FULL_BLOCKS.fetch_add(1, Ordering::Relaxed);
            LOST_BLOCKS.fetch_add(1, Ordering::Relaxed);
        }
    }
}
