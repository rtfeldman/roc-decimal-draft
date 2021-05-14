#![feature(test)]

extern crate test;

use roc_dec::RocDec;
use std::convert::TryInto;
use test::{black_box, Bencher};

#[bench]
fn dec_sub1(bench: &mut Bencher) {
    let dec1: RocDec = "1.2".try_into().unwrap();
    let dec2: RocDec = "3.4".try_into().unwrap();

    bench.iter(|| {
        black_box(dec1 - dec2);
    });
}

#[bench]
fn dec_sub7(bench: &mut Bencher) {
    let dec1: RocDec = "1.2".try_into().unwrap();
    let dec2: RocDec = "3.4".try_into().unwrap();

    bench.iter(|| {
        black_box({
            let a = black_box(dec1 - dec2);
            let a = black_box(a - dec1);
            let a = black_box(a - dec2);
            let a = black_box(a - dec1);
            let a = black_box(a - dec2);
            let a = black_box(a - dec1);
            let a = black_box(a - dec2);
            let a = black_box(a - dec1);
            let a = black_box(a - dec2);
            let a = black_box(a - dec1);
            let a = black_box(a - dec2);
            let a = black_box(a - dec1);
            let a = black_box(a - dec2);
            let a = black_box(a - dec1);
            let a = black_box(a - dec2);
            let a = black_box(a - dec1);
            let a = black_box(a - dec2);

            black_box(a)
        })
    });
}

#[bench]
fn f64_sub1(bench: &mut Bencher) {
    let f1: f64 = 1.2;
    let f2: f64 = 3.4;

    bench.iter(|| {
        black_box(sub_or_panic(f1, f2));
    });
}

#[bench]
fn f64_sub7(bench: &mut Bencher) {
    let f1: f64 = 1.2;
    let f2: f64 = 3.4;

    bench.iter(|| {
        black_box({
            let a = black_box(sub_or_panic(f1, f2));
            let a = black_box(sub_or_panic(a, f1));
            let a = black_box(sub_or_panic(a, f2));
            let a = black_box(sub_or_panic(a, f1));
            let a = black_box(sub_or_panic(a, f2));
            let a = black_box(sub_or_panic(a, f1));
            let a = black_box(sub_or_panic(a, f2));
            let a = black_box(sub_or_panic(a, f1));
            let a = black_box(sub_or_panic(a, f2));
            let a = black_box(sub_or_panic(a, f1));
            let a = black_box(sub_or_panic(a, f2));
            let a = black_box(sub_or_panic(a, f1));
            let a = black_box(sub_or_panic(a, f2));
            let a = black_box(sub_or_panic(a, f1));
            let a = black_box(sub_or_panic(a, f2));
            let a = black_box(sub_or_panic(a, f1));
            let a = black_box(sub_or_panic(a, f2));

            black_box(a)
        })
    });
}

fn sub_or_panic(a: f64, b: f64) -> f64 {
    let answer = a - b;

    if answer.is_finite() {
        answer
    } else {
        todo!("throw an exception");
    }
}
