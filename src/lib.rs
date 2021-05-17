#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct RocDec(i128);

pub fn fuzz_new(num: i128) -> RocDec {
    RocDec(num)
}

// The result of calling to_string() on RocDec(i128::MIN).
// This is both a special case and also the longest to_string().
static I128_MIN_STR: &str = "-1701411834604692317.31687303715884105728";

impl Into<String> for RocDec {
    fn into(self) -> String {
        return self.to_string();
    }
}

impl<'a> std::convert::TryFrom<&'a str> for RocDec {
    type Error = ();

    fn try_from(value: &'a str) -> Result<Self, ()> {
        // Split the string into the parts before and after the "."
        let mut parts = value.split(".");

        let before_point = match parts.next() {
            Some(answer) => answer,
            None => {
                return Err(());
            }
        };

        let after_point = match parts.next() {
            Some(answer) if answer.len() <= Self::DECIMAL_PLACES as usize => answer,
            _ => {
                return Err(());
            }
        };

        // There should have only been one "." in the string!
        if parts.next().is_some() {
            return Err(());
        }

        // Calculate the low digits - the ones after the decimal point.
        let lo = match after_point.parse::<i128>() {
            Ok(answer) => {
                // Translate e.g. the 1 from 0.1 into 10000000000000000000
                // by "restoring" the elided trailing zeroes to the number!
                let trailing_zeroes = Self::DECIMAL_PLACES as usize - after_point.len();
                let lo = answer * 10i128.pow(trailing_zeroes as u32);

                if !before_point.starts_with("-") {
                    lo
                } else {
                    -lo
                }
            }
            Err(_) => {
                return Err(());
            }
        };

        // Calculate the high digits - the ones before the decimal point.
        match before_point.parse::<i128>() {
            Ok(answer) => match answer.checked_mul(10i128.pow(Self::DECIMAL_PLACES)) {
                Some(hi) => match hi.checked_add(lo) {
                    Some(num) => Ok(RocDec(num)),
                    None => Err(()),
                },
                None => Err(()),
            },
            Err(_) => Err(()),
        }
    }
}

impl std::ops::Neg for RocDec {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        match self.0.checked_neg() {
            Some(answer) => RocDec(answer),
            None => {
                todo!("throw exception");
            }
        }
    }
}

impl std::ops::Add for RocDec {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        let (answer, overflowed) = self.0.overflowing_add(other.0);

        if !overflowed {
            RocDec(answer)
        } else {
            todo!("throw an exception");
        }
    }
}

impl std::ops::Sub for RocDec {
    type Output = Self;

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        let (answer, overflowed) = self.0.overflowing_sub(other.0);

        if !overflowed {
            RocDec(answer)
        } else {
            todo!("throw an exception");
        }
    }
}

impl std::ops::Mul for RocDec {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let self_i128 = self.0;
        let other_i128 = other.0;

        // If they're both negative, or if neither is negative, the final answer
        // is positive or zero. If one is negative and the other isn't, the
        // final answer is negative (or zero, in which case final sign won't matter).
        //
        // It's important that we do this in terms of negatives, because doing
        // it in terms of positives can cause bugs when one is zero.
        let is_answer_negative = self_i128.is_negative() != other_i128.is_negative();

        // Break the two i128s into two { hi: u64, lo: u64 } tuples, discarding
        // the sign for now.
        //
        // We'll multiply all 4 combinations of these (hi1 x lo1, hi2 x lo2,
        // hi1 x lo2, hi2 x lo1) and add them as appropriate, then apply the
        // appropriate sign at the very end.
        //
        // We do checked_abs because if we had -i128::MAX before, this will overflow.
        let self_u128 = match self_i128.checked_abs() {
            Some(answer) => answer as u128,
            None => {
                // Currently, if you try to do multiplication on i64::MIN, panic
                // unless you're specifically multiplying by 0 or 1.
                //
                // Maybe we could support more cases in the future
                if other_i128 == 0 {
                    // Anything times 0 is 0
                    return RocDec(0);
                } else if other_i128 == Self::ONE_POINT_ZERO {
                    // Anything times 1 is itself
                    return self;
                } else {
                    todo!("TODO overflow!");
                }
            }
        };

        let other_u128 = match other_i128.checked_abs() {
            Some(answer) => answer as u128,
            None => {
                // Currently, if you try to do multiplication on i64::MIN, panic
                // unless you're specifically multiplying by 0 or 1.
                //
                // Maybe we could support more cases in the future
                if self_i128 == 0 {
                    // Anything times 0 is 0
                    return RocDec(0);
                } else if self_i128 == Self::ONE_POINT_ZERO {
                    // Anything times 1 is itself
                    return other;
                } else {
                    todo!("TODO overflow!");
                }
            }
        };

        let unsigned_answer = mul_and_decimalize(self_u128, other_u128) as i128;
        // TODO make sure this actually fits in i128!

        // This compiles to a cmov!
        if is_answer_negative {
            RocDec(-unsigned_answer)
        } else {
            RocDec(unsigned_answer)
        }
    }
}

/// A fixed-point decimal value with 20 decimal places of precision.
///
/// The lowest value it can store is -1701411834604692317.31687303715884105728
/// and the highest is 1701411834604692317.31687303715884105727
impl RocDec {
    pub const MIN: Self = Self(i128::MIN);
    pub const MAX: Self = Self(i128::MAX);

    pub const DECIMAL_PLACES: u32 = 20;

    const ONE_POINT_ZERO: i128 = 10i128.pow(Self::DECIMAL_PLACES);

    pub fn to_string(self) -> String {
        let self_i128 = self.0;
        let is_negative = self_i128.is_negative();

        match self_i128.checked_abs() {
            Some(answer) => {
                if answer == 0 {
                    return "0.0".to_string();
                }

                let mut self_u128 = answer as u128;

                // I128_MIN_STR has the longest possible length of one of these.
                // This ensures we will not need to reallocate.
                //
                // TODO do this with a RocStr, and don't allocate on the heap
                // until we've run out of stack bytes for a small str.
                let mut string = String::with_capacity(I128_MIN_STR.len());
                let mut display_zeroes = false;
                let mut decimal_places_used = 0;

                // The number has at least one nonzero digit before the
                // decimal point, so we don't need to pad the rest with 0s.
                while self_u128 != 0 {
                    // Get the last digit of the number and convert it to ASCII.
                    let rem = (self_u128 % 10) as u8;
                    let ascii_byte = rem + if rem > 9 { b'a' - 10 } else { b'0' };

                    // Only print zeroes if display_zeroes is set,
                    // but always print non-zeroes.
                    if display_zeroes || ascii_byte != b'0' {
                        string.push(ascii_byte as char);

                        // As soon as we hit our first nonzero number,
                        // we should start showing zeroes from then on.
                        // (All the others are potentially trailing zeroes.)
                        display_zeroes = true;
                    }

                    self_u128 = self_u128 / 10;

                    // Increment this even if we didn't push anything onto the string!
                    // Later on, this tells us how many zeroes to use for padding.
                    decimal_places_used += 1;

                    // If we've used all our decimal places, it's now time to
                    // add the dot.
                    if decimal_places_used == Self::DECIMAL_PLACES {
                        // If we're about to end the loop, push the decimal
                        // point folloewd by the 0 we need, and break early.
                        if self_u128 == 0 {
                            string.push_str(".0");
                            break;
                        }

                        // display_zeroes would only be false here if we had
                        // seen only (trailing, unnecessary) zeroes so far.
                        if !display_zeroes {
                            // Since we had all zeroes after the decimal point
                            // put a single 0 in there so the final string will
                            // end in ".0" instead of in "."
                            // (after reversing, of course).
                            string.push('0');
                        }

                        string.push('.');

                        // After the dot, display all zeroes.
                        // (Otherwise we would incorrectly display `100` as `1`)
                        display_zeroes = true;
                    }
                }

                if decimal_places_used < Self::DECIMAL_PLACES {
                    // We ended without having emitted a dot, so print out
                    // some zeroes as padding and then add a dot and a zero.
                    // (Remember, these will get reversed! This adds "0." to
                    // the beginning of the string, and then a bunch of zeroes
                    // after the dot.)
                    for _ in 0..(Self::DECIMAL_PLACES - decimal_places_used) {
                        string.push('0');
                    }

                    string.push_str(".0");
                }

                // If number is negative, append '-'
                if is_negative {
                    string.push('-');
                }

                // Reverse the string's bytes in place. We can do this byte-wise
                // because we know for sure they are all ASCII characters.
                //
                // Algorithm based on https://stackoverflow.com/a/65703808
                let mut remaining = unsafe { string.as_bytes_mut() };

                loop {
                    match remaining {
                        [] | [_] => break,
                        [first, rest @ .., last] => {
                            std::mem::swap(first, last);

                            remaining = rest;
                        }
                    }
                }

                string
            }
            None => {
                // if it was exactly i128::MIN, then taking its absolute value
                // is impossible to represent...but we also know the exact str:
                I128_MIN_STR.to_string()
            }
        }
    }
}

/// Multiply two 128-bit ints and divide the result by 10^DECIMAL_PLACES
///
/// Adapted from https://github.com/nlordell/ethnum-rs
/// Copyright (c) 2020 Nicholas Rodrigues Lordello
/// Licensed under the Apache License version 2.0
#[inline(always)]
fn mul_and_decimalize(a: u128, b: u128) -> u128 {
    // Multiply

    // The high and low bits of our u256 product
    let mut hi: u128;
    let mut lo: u128;

    const BITS_IN_DWORD_2: u32 = 64;
    const LOWER_MASK: u128 = u128::MAX >> BITS_IN_DWORD_2;

    lo = (a & LOWER_MASK) * (b & LOWER_MASK);
    let mut t = lo >> BITS_IN_DWORD_2;
    lo &= LOWER_MASK;
    t += (a >> BITS_IN_DWORD_2) * (b & LOWER_MASK);
    lo += (t & LOWER_MASK) << BITS_IN_DWORD_2;
    hi = t >> BITS_IN_DWORD_2;
    t = lo >> BITS_IN_DWORD_2;
    lo &= LOWER_MASK;
    t += (b >> BITS_IN_DWORD_2) * (a & LOWER_MASK);
    lo += (t & LOWER_MASK) << BITS_IN_DWORD_2;
    hi += t >> BITS_IN_DWORD_2;
    hi += (a >> BITS_IN_DWORD_2) * (b >> BITS_IN_DWORD_2);

    // Divide

    // Since we want to divide by 10^20, we can instead bit shift by 20 and then
    // divide by 5^20 instead. (This is an inlined u256 shift right.)
    hi = (hi >> RocDec::DECIMAL_PLACES) | (lo << (128 - RocDec::DECIMAL_PLACES));
    lo = lo >> RocDec::DECIMAL_PLACES;

    const DENOM: u128 = 5u128.pow(RocDec::DECIMAL_PLACES);

    #[derive(Copy, Clone)]
    struct U256 {
        hi: u128,
        lo: u128,
    }

    #[inline]
    fn wrapping_sub(r: U256, a: U256) -> U256 {
        let (lo, carry) = r.lo.overflowing_sub(a.lo);

        U256 {
            lo,
            hi: r.hi.wrapping_sub(carry as _).wrapping_sub(a.hi),
        }
    }

    const N_UDWORD_BITS: u32 = 128;
    const N_UTWORD_BITS: u32 = 256;

    if false {
        use ethnum::U256;

        return *(U256::from_words(hi, lo) / U256::from_words(0, DENOM)).low();
    }

    let mut q;
    let mut r;
    let mut sr: u32;

    // special cases, X is unknown, K != 0
    if hi == 0 {
        // we know d.hi == 0, so:

        // 0 X
        // ---
        // 0 X
        return lo / DENOM;
    }

    // K X
    // ---
    // 0 K
    sr = 1 + N_UDWORD_BITS + DENOM.leading_zeros() - (hi).leading_zeros();
    // 2 <= sr <= N_UTWORD_BITS - 1
    // q.all = n.all << (N_UTWORD_BITS - sr);
    // r.all = n.all >> sr;
    #[allow(clippy::comparison_chain)]
    if sr == N_UDWORD_BITS {
        q = U256 { hi: lo, lo: 0 };
        r = U256 { hi: 0, lo: hi };
    } else if sr < N_UDWORD_BITS {
        /* 2 <= sr <= N_UDWORD_BITS - 1 */
        q = U256 {
            hi: lo << (N_UDWORD_BITS - sr),
            lo: 0,
        };
        r = U256 {
            hi: hi >> sr,
            lo: (hi << (N_UDWORD_BITS - sr)) | (lo >> sr),
        };
    } else {
        /* N_UDWORD_BITS + 1 <= sr <= N_UTWORD_BITS - 1 */
        q = U256 {
            hi: (hi << (N_UTWORD_BITS - sr)) | (lo >> (sr - N_UDWORD_BITS)),
            lo: lo << (N_UTWORD_BITS - sr),
        };
        r = U256 {
            hi: 0,
            lo: hi >> (sr - N_UDWORD_BITS),
        };
    }

    // Not a special case
    // q and r are initialized with:
    // q.all = n.all << (N_UTWORD_BITS - sr);
    // r.all = n.all >> sr;
    // 1 <= sr <= N_UTWORD_BITS - 1
    let mut carry = 0u128;
    while sr > 0 {
        // r:q = ((r:q)  << 1) | carry
        r.hi = (r.hi << 1) | (r.lo >> (N_UDWORD_BITS - 1));
        r.lo = (r.lo << 1) | (q.hi >> (N_UDWORD_BITS - 1));
        q.hi = (q.hi << 1) | (q.lo >> (N_UDWORD_BITS - 1));
        q.lo = (q.lo << 1) | carry;
        // carry = 0;
        // if (r.all >= d.all)
        // {
        //     r.all -= d.all;
        //      carry = 1;
        // }
        // NOTE: Modified from `(d - r - 1) >> (N_UTWORD_BITS - 1)` to be an
        // **arithmetic** shift.
        let s = {
            let hi = wrapping_sub(
                wrapping_sub(U256 { hi: 0, lo: DENOM }, r),
                U256 { hi: 0, lo: 1 },
            )
            .hi;

            U256 {
                hi: 0,
                lo: ((hi as i128) >> 127) as u128,
            }
        };

        carry = s.lo & 1;
        r = wrapping_sub(
            r,
            U256 {
                hi: 0,
                lo: DENOM & s.lo,
            },
        );

        sr -= 1;
    }

    let final_hi = (q.hi << 1) | (q.lo >> (128 - 1));

    if final_hi == 0 {
        (q.lo << 1) | carry
    } else {
        // high bits of the u256 weren't empty!
        todo!("Overflow!");
    }
}

#[cfg(test)]
mod tests {
    use crate::RocDec;
    use std::convert::TryInto;
    use std::ops::{Add, Mul, Neg, Sub};

    fn assert_reflexive(string: &str) {
        let dec: RocDec = string.try_into().unwrap();

        // We should be able to convert it from a string back into this RocDec
        assert_eq!(Ok(dec), dec.to_string().as_str().try_into());
    }

    fn assert_negated(string: &str) {
        let dec: RocDec = string.try_into().unwrap();

        assert_eq!(-dec, RocDec(-(dec.0)));

        if dec.0 != 0 {
            // If it wasn't 0.0, the sign should have changed!
            assert_ne!(
                dec.neg().to_string().starts_with("-"),
                string.starts_with("-")
            );
        }

        // After removing minus signs, both strings should be equal.
        assert_eq!(string.replace("-", ""), (-dec).to_string().replace("-", ""));
    }

    #[test]
    fn neg_0() {
        assert_negated("0.0");
    }

    #[test]
    fn neg_1() {
        assert_negated("1.0");
    }

    #[test]
    fn neg_1pt1() {
        assert_negated("1.1");
    }

    #[test]
    fn neg_minus_1pt1() {
        assert_negated("-1.1");
    }

    #[test]
    fn zero_to_str() {
        assert_eq!("0.0", RocDec(0).to_string());
    }

    #[test]
    fn one_to_str() {
        assert_eq!("1.0", RocDec(100000000000000000000).to_string());
    }

    #[test]
    fn ten_to_str() {
        assert_eq!("10.0", RocDec(1000000000000000000000).to_string());
    }

    #[test]
    fn point_1_to_str() {
        assert_eq!("0.1", RocDec(10000000000000000000).to_string());
    }

    #[test]
    fn point_01_to_str() {
        assert_eq!("0.01", RocDec(1000000000000000000).to_string());
    }

    #[test]
    fn smallest_positive_to_str() {
        assert_eq!("0.00000000000000000001", RocDec(1).to_string());
    }

    #[test]
    fn i128_min_to_str() {
        assert_eq!(
            "-1701411834604692317.31687303715884105728",
            RocDec(i128::MIN).to_string()
        );
    }

    #[test]
    fn i128_almost_min_to_str() {
        // i128::MIN is special-cased because transforming it into a u128
        // would overflow, so make sure that i128::MAX + 1 works as expected!
        assert_eq!(
            "-1701411834604692317.31687303715884105727",
            RocDec(i128::MIN + 1).to_string()
        );
    }

    #[test]
    fn i128_max_to_str() {
        assert_eq!(
            "1701411834604692317.31687303715884105727",
            RocDec(i128::MAX).to_string()
        );
    }

    #[test]
    fn i128_almost_max_to_str() {
        assert_eq!(
            "1701411834604692317.31687303715884105726",
            RocDec(i128::MAX - 1).to_string()
        );
    }

    #[test]
    fn from_str_0() {
        assert_reflexive("0.0");
    }

    #[test]
    fn from_str_positive_int() {
        assert_reflexive("1.0");
        assert_reflexive("10.0");
        assert_reflexive("360.0");
    }

    #[test]
    fn from_str_negative_int() {
        assert_reflexive("-1.0");
        assert_reflexive("-10.0");
        assert_reflexive("-360.0");
    }

    #[test]
    fn from_str_0_point() {
        assert_reflexive("0.0000000000000000001");
    }

    #[test]
    fn from_str_before_and_after_dot() {
        assert_reflexive("360.0000000000000000001");
        assert_reflexive("360.00000000000000001");
        assert_reflexive("360.000000000000000012");
        assert_reflexive("3600.000000000000000012");
    }

    #[test]
    fn from_str_min() {
        assert_reflexive("-1701411834604692317.31687303715884105728"); // RocDec::MIN
        assert_reflexive("-1701411834604692317.31687303715884105727"); // RocDec::MIN + 1
    }

    #[test]
    fn from_str_max() {
        assert_reflexive("1701411834604692317.31687303715884105727"); // RocDec::MAX
        assert_reflexive("1701411834604692317.31687303715884105726"); // RocDec::MAX - 1
    }

    fn assert_add(dec1: &str, dec2: &str, expected: &str) {
        let dec1: RocDec = dec1.try_into().unwrap();
        let dec2: RocDec = dec2.try_into().unwrap();

        assert_eq!(expected, dec1.add(dec2).to_string());
    }

    fn assert_sub(dec1: &str, dec2: &str, expected: &str) {
        let dec1: RocDec = dec1.try_into().unwrap();
        let dec2: RocDec = dec2.try_into().unwrap();

        assert_eq!(expected, dec1.sub(dec2).to_string());
    }

    fn assert_mul(dec1: &str, dec2: &str, expected: &str) {
        let dec1: RocDec = dec1.try_into().unwrap();
        let dec2: RocDec = dec2.try_into().unwrap();

        assert_eq!(expected, dec1.mul(dec2).to_string());
    }

    #[test]
    fn add() {
        // integers
        assert_add("0.0", "0.0", "0.0");
        assert_add("360.0", "360.0", "720.0");
        assert_add("123.0", "456.0", "579.0");

        // non-integers
        assert_add("0.1", "0.2", "0.3");
        assert_add(
            "111.0000000000000000555",
            "222.0000000000000000444",
            "333.0000000000000000999",
        );
    }

    #[test]
    fn sub() {
        // integers
        assert_sub("0.0", "0.0", "0.0");
        assert_sub("360.0", "360.0", "0.0");
        assert_sub("123.0", "456.0", "-333.0");

        // non-integers
        assert_sub("0.3", "0.2", "0.1");
        assert_sub(
            "111.0000000000000000555",
            "222.0000000000000000444",
            "-110.9999999999999999889",
        );
    }

    #[test]
    fn mul() {
        // integers
        assert_mul("0.0", "0.0", "0.0");
        assert_mul("0.0003", "0.0002", "0.00000006");
        assert_mul("2.0", "3.0", "6.0");
        assert_mul("-2.0", "3.0", "-6.0");
        assert_mul("2.0", "-3.0", "-6.0");
        assert_mul("-2.0", "-3.0", "6.0");
        assert_mul("15.0", "74.0", "1110.0");
        assert_mul("-15.0", "74.0", "-1110.0");
        assert_mul("15.0", "-74.0", "-1110.0");
        assert_mul("-15.0", "-74.0", "1110.0");

        // non-integers
        assert_mul("1.1", "2.2", "2.42");
        assert_mul("-1.1", "-2.2", "2.42");
        assert_mul("1.1", "-2.2", "-2.42");
        assert_mul("2.0", "1.5", "3.0");
        assert_mul("2.3", "3.8", "8.74");
        assert_mul("1.01", "7.02", "7.0902");
        assert_mul("1.001", "7.002", "7.009002");
        assert_mul("1.0001", "7.0002", "7.00090002");
        assert_mul("1.00001", "7.00002", "7.0000900002");
        assert_mul("1.000001", "7.000002", "7.000009000002");
        assert_mul("1.0000001", "7.0000002", "7.00000090000002");
        assert_mul("1.00000001", "7.00000002", "7.0000000900000002");
        assert_mul("1.000000001", "7.000000002", "7.000000009000000002");
        assert_mul("-1.000000001", "7.000000002", "-7.000000009000000002");
        assert_mul("1.000000001", "-7.000000002", "-7.000000009000000002");
    }
}
