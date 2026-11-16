unsafe fn copy_bytes(destination: *mut u8, source: *const u8, length: usize) {
    for offset in 0..length {
        // Volatile byte accesses prevent lowering this loop back to memcpy.
        unsafe {
            destination
                .add(offset)
                .write_volatile(source.add(offset).read_volatile());
        }
    }
}

#[cfg(target_os = "none")]
unsafe extern "C" {
    fn __real_memcpy(destination: *mut u8, source: *const u8, length: usize) -> *mut u8;
}

#[cfg(target_os = "none")]
#[unsafe(no_mangle)]
unsafe extern "C" fn __wrap_memcpy(
    destination: *mut u8,
    source: *const u8,
    length: usize,
) -> *mut u8 {
    // Newlib's odd-length tails use unaligned halfwords, which fault in USB Device memory.
    let dpram = 0x5010_0000..0x5010_1000;
    if dpram.contains(&(destination as usize)) || dpram.contains(&(source as usize)) {
        unsafe { copy_bytes(destination, source, length) };
        destination
    } else {
        unsafe { __real_memcpy(destination, source, length) }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn byte_copy_preserves_bounds_for_all_alignments_and_packet_lengths() {
        let source: [u8; 80] = core::array::from_fn(|i| i as u8);
        for source_offset in 0..4 {
            for destination_offset in 0..4 {
                for length in 0..=68 {
                    let mut destination = [0xa5; 80];
                    unsafe {
                        super::copy_bytes(
                            destination.as_mut_ptr().add(destination_offset),
                            source.as_ptr().add(source_offset),
                            length,
                        );
                    }
                    assert!(destination[..destination_offset].iter().all(|b| *b == 0xa5));
                    assert_eq!(
                        &destination[destination_offset..destination_offset + length],
                        &source[source_offset..source_offset + length],
                    );
                    assert!(
                        destination[destination_offset + length..]
                            .iter()
                            .all(|b| *b == 0xa5)
                    );
                }
            }
        }
    }
}
