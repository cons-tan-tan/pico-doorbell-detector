MEMORY {
  FLASH : ORIGIN = 0x10000000, LENGTH = 4092K
  DEVICE_KEY : ORIGIN = 0x103ff000, LENGTH = 4K
  RAM : ORIGIN = 0x20000000, LENGTH = 520K
}

SECTIONS {
  /* The RP2350 boot ROM searches for the image definition in the first 4 KiB. */
  .start_block : ALIGN(4) {
    __start_block_addr = .;
    KEEP(*(.start_block));
    KEEP(*(.boot_info));
  } > FLASH
} INSERT AFTER .vector_table;

_stext = ALIGN(ADDR(.start_block) + SIZEOF(.start_block), ALIGNOF(.text));

SECTIONS {
  .end_block : ALIGN(4) {
    __end_block_addr = .;
    KEEP(*(.end_block));
  } > FLASH
} INSERT AFTER .uninit;

PROVIDE(start_to_end = __end_block_addr - __start_block_addr);
PROVIDE(end_to_start = __start_block_addr - __end_block_addr);
ASSERT(_stack_start - __sheap >= 16K, "Less than 16 KiB remains for the core 0 stack");
