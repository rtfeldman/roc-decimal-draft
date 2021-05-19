#![feature(test)]

extern crate test;

use decimal::d128;
use roc_dec::RocDec;
use std::convert::TryInto;
use test::{black_box, Bencher};

#[bench]
fn dec_div1(bench: &mut Bencher) {
    let dec1: RocDec = "1.2".try_into().unwrap();
    let dec2: RocDec = "3.4".try_into().unwrap();

    bench.iter(|| {
        black_box(dec1 / dec2);
    });
}

#[bench]
fn dec_div7(bench: &mut Bencher) {
    let dec1: RocDec = "1.2".try_into().unwrap();
    let dec2: RocDec = "3.4".try_into().unwrap();

    bench.iter(|| {
        black_box({
            let a = black_box(dec1 / dec2);
            let a = black_box(a / dec1);
            let a = black_box(a / dec2);
            let a = black_box(a / dec1);
            let a = black_box(a / dec2);
            let a = black_box(a / dec1);
            let a = black_box(a / dec2);
            let a = black_box(a / dec1);
            let a = black_box(a / dec2);
            let a = black_box(a / dec1);
            let a = black_box(a / dec2);
            let a = black_box(a / dec1);
            let a = black_box(a / dec2);
            let a = black_box(a / dec1);
            let a = black_box(a / dec2);
            let a = black_box(a / dec1);
            let a = black_box(a / dec2);

            black_box(a)
        })
    });
}

#[bench]
fn d128_div1(bench: &mut Bencher) {
    let d1: d128 = d128!(1.2);
    let d2: d128 = d128!(3.4);

    bench.iter(|| {
        black_box(div_d128_or_panic(d1, d2));
    });
}

#[bench]
fn i128_div1(bench: &mut Bencher) {
    let i1: i128 = 12;
    let i2: i128 = 34;

    bench.iter(|| {
        black_box(div_i128_or_panic(i1, i2));
    });
}

#[bench]
fn f64_div1(bench: &mut Bencher) {
    let f1: f64 = 1.2;
    let f2: f64 = 3.4;

    bench.iter(|| {
        black_box(div_or_panic(f1, f2));
    });
}

#[bench]
fn f64_div7(bench: &mut Bencher) {
    let f1: f64 = 1.2;
    let f2: f64 = 3.4;

    bench.iter(|| {
        black_box({
            let a = black_box(div_or_panic(f1, f2));
            let a = black_box(div_or_panic(a, f1));
            let a = black_box(div_or_panic(a, f2));
            let a = black_box(div_or_panic(a, f1));
            let a = black_box(div_or_panic(a, f2));
            let a = black_box(div_or_panic(a, f1));
            let a = black_box(div_or_panic(a, f2));
            let a = black_box(div_or_panic(a, f1));
            let a = black_box(div_or_panic(a, f2));
            let a = black_box(div_or_panic(a, f1));
            let a = black_box(div_or_panic(a, f2));
            let a = black_box(div_or_panic(a, f1));
            let a = black_box(div_or_panic(a, f2));
            let a = black_box(div_or_panic(a, f1));
            let a = black_box(div_or_panic(a, f2));
            let a = black_box(div_or_panic(a, f1));
            let a = black_box(div_or_panic(a, f2));

            black_box(a)
        })
    });
}

#[bench]
fn i128_div7(bench: &mut Bencher) {
    let i1: i128 = 12;
    let i2: i128 = 34;

    bench.iter(|| {
        black_box({
            let a = black_box(div_i128_or_panic(i1, i2));
            let a = black_box(div_i128_or_panic(a, i1));
            let a = black_box(div_i128_or_panic(a, i2));
            let a = black_box(div_i128_or_panic(a, i1));
            let a = black_box(div_i128_or_panic(a, i2));
            let a = black_box(div_i128_or_panic(a, i1));
            let a = black_box(div_i128_or_panic(a, i2));
            let a = black_box(div_i128_or_panic(a, i1));
            let a = black_box(div_i128_or_panic(a, i2));
            let a = black_box(div_i128_or_panic(a, i1));
            let a = black_box(div_i128_or_panic(a, i2));
            let a = black_box(div_i128_or_panic(a, i1));
            let a = black_box(div_i128_or_panic(a, i2));
            let a = black_box(div_i128_or_panic(a, i1));
            let a = black_box(div_i128_or_panic(a, i2));
            let a = black_box(div_i128_or_panic(a, i1));
            let a = black_box(div_i128_or_panic(a, i2));

            black_box(a)
        })
    });
}

#[bench]
fn d128_div7(bench: &mut Bencher) {
    let d1: d128 = d128!(1.2);
    let d2: d128 = d128!(3.4);

    bench.iter(|| {
        black_box({
            let a = black_box(div_d128_or_panic(d1, d2));
            let a = black_box(div_d128_or_panic(a, d1));
            let a = black_box(div_d128_or_panic(a, d2));
            let a = black_box(div_d128_or_panic(a, d1));
            let a = black_box(div_d128_or_panic(a, d2));
            let a = black_box(div_d128_or_panic(a, d1));
            let a = black_box(div_d128_or_panic(a, d2));
            let a = black_box(div_d128_or_panic(a, d1));
            let a = black_box(div_d128_or_panic(a, d2));
            let a = black_box(div_d128_or_panic(a, d1));
            let a = black_box(div_d128_or_panic(a, d2));
            let a = black_box(div_d128_or_panic(a, d1));
            let a = black_box(div_d128_or_panic(a, d2));
            let a = black_box(div_d128_or_panic(a, d1));
            let a = black_box(div_d128_or_panic(a, d2));
            let a = black_box(div_d128_or_panic(a, d1));
            let a = black_box(div_d128_or_panic(a, d2));

            black_box(a)
        })
    });
}

fn div_i128_or_panic(a: i128, b: i128) -> i128 {
    let (answer, overflowed) = a.overflowing_div(b);

    if !overflowed {
        answer
    } else {
        todo!("throw an exception");
    }
}

fn div_d128_or_panic(a: d128, b: d128) -> d128 {
    let answer = a / b;

    if answer.is_finite() {
        answer
    } else {
        todo!("throw an exception");
    }
}

fn div_or_panic(a: f64, b: f64) -> f64 {
    let answer = a / b;

    if answer.is_finite() {
        answer
    } else {
        todo!("throw an exception");
    }
}
