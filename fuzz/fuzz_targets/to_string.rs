#![no_main]
use libfuzzer_sys::fuzz_target;

use roc_dec::RocDec;
use std::convert::TryInto;
use std::mem::size_of;

fuzz_target!(|data: &[u8]| {
    // TODO use the `arbitrary` crate instead of doing this
    let num = if data.len() >= size_of::<i128>() {
        i128::from_le_bytes(data[0..size_of::<i128>()].try_into().unwrap())
    } else if data.len() >= size_of::<i64>() {
        i64::from_le_bytes(data[0..size_of::<i64>()].try_into().unwrap()) as i128
    } else if data.len() >= size_of::<i32>() {
        i32::from_le_bytes(data[0..size_of::<i32>()].try_into().unwrap()) as i128
    } else if data.len() >= size_of::<i16>() {
        i16::from_le_bytes(data[0..size_of::<i16>()].try_into().unwrap()) as i128
    } else if data.is_empty() {
        0
    } else {
        data[0] as i128
    };

    dbg!(num);

    let dec_to_str = roc_dec::fuzz_new(num).to_string();

    // There should be a dot with something before it and something after.
    let mut dec_pieces = dec_to_str.split('.');
    let before_dot = dec_pieces.next().unwrap();
    let after_dot = dec_pieces.next().unwrap();

    // There should only be one dot!
    assert!(dec_pieces.next().is_none());

    // We shouldn't have more than the maximum number of decimal places.
    assert!(after_dot.len() <= RocDec::DECIMAL_PLACES as usize);

    // We shouldn't have more than the maximum number of before-dot digits.
    assert!(before_dot.len() <= 39 - RocDec::DECIMAL_PLACES as usize);

    // if num == 0 {
    //     // 0 is a special case; it should print 0.0
    //     assert_eq!("0.0", dec_to_str);
    // } else {
    //     // Calling to_string() on the number should give the same answer
    //     // as the decimal's to_string, except without the dot.
    //     assert_eq!(num.to_string(), format!("{}{}", before_dot, after_dot));
    // }
});
