#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct RocDec(i128);

pub fn fuzz_new(num: i128) -> RocDec {
    RocDec(num)
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct U256 {
    hi: u128,
    lo: u128,
}

// The result of calling to_string() on RocDec::MIN.
// This is the longest to_string().
static MIN_STR: &str = "-170141183460469231731.687303715884105728";

// The result of calling to_string() on RocDec::MAX.
static MAX_STR: &str = "170141183460469231731.687303715884105727";

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

        // This compiles to a cmov!
        if is_answer_negative {
            RocDec(-unsigned_answer)
        } else {
            RocDec(unsigned_answer)
        }
    }
}

impl std::ops::Div for RocDec {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let self_i128 = self.0;
        let other_i128 = other.0;

        // Zero divided by anything is zero.
        if self_i128 == 0 {
            return RocDec(0);
        }

        // Anything divided by zero is an error.
        if self_i128 == 0 {
            todo!("division by zero");
        }

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
                if other_i128 == Self::ONE_POINT_ZERO {
                    // Anything divided by 1 is itself
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
                if self_i128 == Self::ONE_POINT_ZERO {
                    // Anything times 1 is itself
                    return other;
                } else {
                    todo!("TODO overflow!");
                }
            }
        };

        let unsigned_answer = {
            let numer_u256 = mul_u128(self_u128, 10u128.pow(Self::DECIMAL_PLACES));
            let answer = div_u256_by_u128(numer_u256, other_u128);
            let lo = answer.lo;

            if answer.hi == 0 && lo <= i128::MAX as u128 {
                lo as i128
            } else {
                todo!("TODO overflow!");
            }
        };

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

    pub const DECIMAL_PLACES: u32 = 18;

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

                // MIN_STR has the longest possible length of one of these.
                // This ensures we will not need to reallocate.
                //
                // TODO do this with a RocStr, and don't allocate on the heap
                // until we've run out of stack bytes for a small str.
                let mut string = String::with_capacity(MIN_STR.len());
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
                MIN_STR.to_string()
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
fn div_u256_by_u128(numer: U256, denom: u128) -> U256 {
    const N_UDWORD_BITS: u32 = 128;
    const N_UTWORD_BITS: u32 = 256;

    let mut q;
    let mut r;
    let mut sr: u32;

    // special case
    if numer.hi == 0 {
        // 0 X
        // ---
        // 0 X
        return U256 {
            hi: 0,
            lo: numer.lo / denom,
        };
    }

    // numer.hi != 0
    if denom == 0 {
        // K X
        // ---
        // 0 0
        return U256 {
            hi: 0,
            lo: numer.hi / denom,
        };
    } else {
        // K X
        // ---
        // 0 K
        // NOTE: Modified from `if (d.low() & (d.low() - 1)) == 0`.
        if denom.is_power_of_two() {
            /* if d is a power of 2 */
            if denom == 1 {
                return numer;
            }
            sr = denom.trailing_zeros();

            return U256 {
                hi: numer.hi >> sr,
                lo: (numer.hi << (N_UDWORD_BITS - sr)) | (numer.lo >> sr),
            };
        }

        // K X
        // ---
        // 0 K
        sr = 1 + N_UDWORD_BITS + denom.leading_zeros() - numer.hi.leading_zeros();
        // 2 <= sr <= N_UTWORD_BITS - 1
        // q.all = n.all << (N_UTWORD_BITS - sr);
        // r.all = n.all >> sr;
        #[allow(clippy::comparison_chain)]
        if sr == N_UDWORD_BITS {
            q = U256 {
                hi: numer.lo,
                lo: 0,
            };
            r = U256 {
                hi: 0,
                lo: numer.hi,
            };
        } else if sr < N_UDWORD_BITS {
            /* 2 <= sr <= N_UDWORD_BITS - 1 */
            q = U256 {
                hi: numer.lo << (N_UDWORD_BITS - sr),
                lo: 0,
            };
            r = U256 {
                hi: numer.hi >> sr,
                lo: (numer.hi << (N_UDWORD_BITS - sr)) | (numer.lo >> sr),
            };
        } else {
            /* N_UDWORD_BITS + 1 <= sr <= N_UTWORD_BITS - 1 */
            q = U256 {
                hi: (numer.hi << (N_UTWORD_BITS - sr)) | (numer.lo >> (sr - N_UDWORD_BITS)),
                lo: numer.lo << (N_UTWORD_BITS - sr),
            };
            r = U256 {
                hi: 0,
                lo: numer.hi >> (sr - N_UDWORD_BITS),
            };
        }
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
            let (lo, carry) = denom.overflowing_sub(r.lo);
            let hi = 0u128.wrapping_sub(carry as u128).wrapping_sub(r.hi);

            let (_lo, carry) = lo.overflowing_sub(1);
            let hi = hi.wrapping_sub(carry as u128);

            // TODO this U256 was originally created by:
            //
            // ((hi as i128) >> 127).as_u256()
            //
            // ...however, I can't figure out where that funciton is defined.
            // Maybe it's defined using a macro or something. Anyway, hopefully
            // this is what it would do in this scenario.
            U256 {
                hi: 0,
                lo: ((hi as i128) >> 127) as u128,
            }
        };
        carry = s.lo & 1;

        r = {
            let (lo, carry) = r.lo.overflowing_sub(denom & s.lo);
            let hi = r.hi.wrapping_sub(carry as _);

            U256 { hi, lo }
        };

        sr -= 1;
    }

    let hi = (q.hi << 1) | (q.lo >> (127));
    let lo = (q.lo << 1) | carry;

    U256 { hi, lo }
}

/// Multiply two 128-bit ints and divide the result by 10^DECIMAL_PLACES
#[inline(always)]
fn mul_and_decimalize(a: u128, b: u128) -> u128 {
    // Multiply
    let U256 {
        hi: lhs_hi,
        lo: lhs_lo,
    } = mul_u128(a, b);

    // Divide - or just add 1, multiply by floor(2^315/10^18), then right shift 315 times.
    // floor(2^315/10^18) is 66749594872528440074844428317798503581334516323645399060845050244444366430645

    // Add 1.
    // This can't overflow because the intial numbers are only 127bit due to removing the sign bit.
    let (lhs_lo, carry) = lhs_lo.overflowing_add(1);
    let lhs_hi = lhs_hi + if carry { 1 } else { 0 };

    // This needs to do multiplication in a way that expands,
    // since we throw away 315 bits we care only about the higher end, not lower.
    // So like need to do high low mult with 2 U256's and then bitshift.
    // I bet this has a lot of room for multiplication optimization.
    let rhs_hi = 0x9392ee8e921d5d073aff322e62439fcfu128;
    let rhs_lo = 0x32d7f344649470f90cac0c573bf9e1b5u128;
    let ea = mul_u128(lhs_lo, rhs_lo);
    let gf = mul_u128(lhs_hi, rhs_lo);
    let jh = mul_u128(lhs_lo, rhs_hi);
    let lk = mul_u128(lhs_hi, rhs_hi);

    let e = ea.hi;
    let _a = ea.lo;

    let g = gf.hi;
    let f = gf.lo;

    let j = jh.hi;
    let h = jh.lo;

    let l = lk.hi;
    let k = lk.lo;

    // b = e + f + h
    let (e_plus_f, overflowed) = e.overflowing_add(f);
    let b_carry1 = if overflowed { 1 } else { 0 };
    let (_b, overflowed) = e_plus_f.overflowing_add(h);
    let b_carry2 = if overflowed { 1 } else { 0 };

    // c = carry + g + j + k // it doesn't say +k but I think it should be?
    let (g_plus_j, overflowed) = g.overflowing_add(j);
    let c_carry1 = if overflowed { 1 } else { 0 };
    let (g_plus_j_plus_k, overflowed) = g_plus_j.overflowing_add(k); // it doesn't say +k but I think it should be?
    let c_carry2 = if overflowed { 1 } else { 0 };
    let (c_without_bcarry2, overflowed) = g_plus_j_plus_k.overflowing_add(b_carry1);
    let c_carry3 = if overflowed { 1 } else { 0 };
    let (c, overflowed) = c_without_bcarry2.overflowing_add(b_carry2);
    let c_carry4 = if overflowed { 1 } else { 0 };

    // d = carry + l
    let (d, overflowed1) = l.overflowing_add(c_carry1);
    let (d, overflowed2) = d.overflowing_add(c_carry2);
    let (d, overflowed3) = d.overflowing_add(c_carry3);
    let (d, overflowed4) = d.overflowing_add(c_carry4);

    if overflowed1 || overflowed2 || overflowed3 || overflowed4 {
        todo!("overflowed")
    }

    // Final 512bit value is d, c, b, a
    // need to left shift 321 times
    // 315 - 256 is 59. So left shift d, c 59 times.
    c >> 59 | (d << (128 - 59))
}

/// Adapted from https://github.com/nlordell/ethnum-rs
/// Copyright (c) 2020 Nicholas Rodrigues Lordello
/// Licensed under the Apache License version 2.0
#[inline(always)]
fn mul_u128(a: u128, b: u128) -> U256 {
    let mut hi;
    let mut lo;

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

    U256 { hi, lo }
}

#[cfg(test)]
mod tests {
    use crate::RocDec;
    use std::convert::TryInto;
    use std::ops::{Add, Div, Mul, Neg, Sub};

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
        assert_eq!("1.0", RocDec(1000000000000000000).to_string());
    }

    #[test]
    fn ten_to_str() {
        assert_eq!("10.0", RocDec(10000000000000000000).to_string());
    }

    #[test]
    fn point_1_to_str() {
        assert_eq!("0.1", RocDec(100000000000000000).to_string());
    }

    #[test]
    fn point_01_to_str() {
        assert_eq!("0.01", RocDec(10000000000000000).to_string());
    }

    #[test]
    fn smallest_positive_to_str() {
        assert_eq!("0.000000000000000001", RocDec(1).to_string());
    }

    #[test]
    fn i128_min_to_str() {
        assert_eq!(
            "-170141183460469231731.687303715884105728",
            RocDec(i128::MIN).to_string()
        );
    }

    #[test]
    fn i128_almost_min_to_str() {
        // i128::MIN is special-cased because transforming it into a u128
        // would overflow, so make sure that i128::MAX + 1 works as expected!
        assert_eq!(
            "-170141183460469231731.687303715884105727",
            RocDec(i128::MIN + 1).to_string()
        );
    }

    #[test]
    fn i128_max_to_str() {
        assert_eq!(
            "170141183460469231731.687303715884105727",
            RocDec(i128::MAX).to_string()
        );
    }

    #[test]
    fn i128_almost_max_to_str() {
        assert_eq!(
            "170141183460469231731.687303715884105726",
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
        assert_reflexive("0.000000000000000001");
    }

    #[test]
    fn from_str_before_and_after_dot() {
        assert_reflexive("360.000000000000000001");
        assert_reflexive("360.00000000000000001");
        assert_reflexive("360.000000000000000012");
        assert_reflexive("3600.000000000000000012");
    }

    #[test]
    fn from_str_min() {
        assert_reflexive("-170141183460469231731.687303715884105728"); // RocDec::MIN
        assert_reflexive("-170141183460469231731.687303715884105727"); // RocDec::MIN + 1
    }

    #[test]
    fn from_str_max() {
        assert_reflexive("170141183460469231731.687303715884105727"); // RocDec::MAX
        assert_reflexive("170141183460469231731.687303715884105726"); // RocDec::MAX - 1
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

    fn assert_div(dec1: &str, dec2: &str, expected: &str) {
        let dec1: RocDec = dec1.try_into().unwrap();
        let dec2: RocDec = dec2.try_into().unwrap();

        assert_eq!(expected, dec1.div(dec2).to_string());
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
            "111.000000000000000555",
            "222.000000000000000444",
            "333.000000000000000999",
        );
    }

    #[test]
    fn add_extremes() {
        let negative_max = &format!("-{}", super::MAX_STR);

        assert_add("0.0", super::MAX_STR, super::MAX_STR);
        assert_add("0.0", super::MIN_STR, super::MIN_STR);
        assert_add(super::MAX_STR, "0.0", super::MAX_STR);
        assert_add(super::MIN_STR, "0.0", super::MIN_STR);
        assert_add(super::MAX_STR, negative_max, "0.0");
        assert_add(negative_max, super::MAX_STR, "0.0");
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
            "111.000000000000000555",
            "222.000000000000000444",
            "-110.999999999999999889",
        );
    }

    #[test]
    fn sub_extremes() {
        let negative_max = &format!("-{}", super::MAX_STR);

        assert_sub("0.0", super::MAX_STR, negative_max);
        assert_sub("0.0", negative_max, super::MAX_STR);
        assert_sub(super::MAX_STR, super::MAX_STR, "0.0");
        assert_sub(super::MIN_STR, super::MIN_STR, "0.0");
    }

    #[test]
    fn mul_zero() {
        assert_mul("0.0", "0.0", "0.0");
    }

    #[test]
    fn mul_small() {
        assert_mul("0.0003", "0.0002", "0.00000006");
    }

    #[test]
    fn mul_positive_ints() {
        assert_mul("2.0", "3.0", "6.0");
        assert_mul("15.0", "74.0", "1110.0");
    }

    #[test]
    fn mul_negative_ints() {
        assert_mul("-2.0", "3.0", "-6.0");
        assert_mul("2.0", "-3.0", "-6.0");
        assert_mul("-2.0", "-3.0", "6.0");
        assert_mul("-15.0", "74.0", "-1110.0");
        assert_mul("15.0", "-74.0", "-1110.0");
        assert_mul("-15.0", "-74.0", "1110.0");
    }

    #[test]
    fn mul_positive_non_ints() {
        assert_mul("1.1", "2.2", "2.42");
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
    }

    #[test]
    fn mul_negative_non_ints() {
        assert_mul("-1.1", "-2.2", "2.42");
        assert_mul("1.1", "-2.2", "-2.42");
        assert_mul("-1.000000001", "7.000000002", "-7.000000009000000002");
        assert_mul("1.000000001", "-7.000000002", "-7.000000009000000002");
    }

    #[test]
    fn mul_extremes() {
        assert_mul("0.0", super::MIN_STR, "0.0");
        assert_mul("0.0", super::MAX_STR, "0.0");
        assert_mul(super::MIN_STR, "0.0", "0.0");
        assert_mul(super::MAX_STR, "0.0", "0.0");
        assert_mul(super::MIN_STR, "1.0", super::MIN_STR);
        assert_mul(super::MAX_STR, "1.0", super::MAX_STR);
        assert_mul("1.0", super::MIN_STR, super::MIN_STR);
        assert_mul("1.0", super::MAX_STR, super::MAX_STR);
    }

    #[test]
    fn div_zero() {
        assert_div("0.0", "1.0", "0.0");
        assert_div("0.0", super::MIN_STR, "0.0");
        assert_div("0.0", super::MAX_STR, "0.0");
    }

    #[test]
    fn div_positive_ints() {
        assert_div("3.0", "2.0", "1.5");
        assert_div("4.0", "2.0", "2.0");
        assert_div("1.0", "8.0", "0.125");
        assert_div("1.0", "3.0", "0.333333333333333333");
        assert_div("15.0", "74.0", "0.202702702702702702");
    }

    #[test]
    fn div_negative_ints() {
        assert_div("3.0", "-2.0", "-1.5");
        assert_div("-3.0", "2.0", "-1.5");
        assert_div("-3.0", "-2.0", "1.5");
        assert_div("4.0", "-2.0", "-2.0");
        assert_div("-4.0", "2.0", "-2.0");
        assert_div("-4.0", "-2.0", "2.0");
        assert_div("1.0", "-8.0", "-0.125");
        assert_div("-1.0", "8.0", "-0.125");
        assert_div("-1.0", "-8.0", "0.125");
        assert_div("1.0", "-3.0", "-0.333333333333333333");
        assert_div("-1.0", "3.0", "-0.333333333333333333");
        assert_div("-1.0", "-3.0", "0.333333333333333333");
        assert_div("15.0", "-74.0", "-0.202702702702702702");
        assert_div("-15.0", "74.0", "-0.202702702702702702");
        assert_div("-15.0", "-74.0", "0.202702702702702702");
    }

    #[test]
    fn div_positive_non_ints() {
        assert_div("0.9", "0.08", "11.25");
        assert_div("1.1", "2.2", "0.5");
        assert_div("2.0", "1.5", "1.333333333333333333");
        assert_div("2.3", "3.8", "0.605263157894736842");
        assert_div("1.01", "7.02", "0.143874643874643874");
    }

    #[test]
    fn div_negative_non_ints() {
        assert_div("0.9", "-0.08", "-11.25");
        assert_div("-0.9", "0.08", "-11.25");
        assert_div("-0.9", "-0.08", "11.25");
        assert_div("1.1", "-2.2", "-0.5");
        assert_div("-1.1", "2.2", "-0.5");
        assert_div("-1.1", "-2.2", "0.5");
        assert_div("2.0", "-1.5", "-1.333333333333333333");
        assert_div("-2.0", "1.5", "-1.333333333333333333");
        assert_div("-2.0", "-1.5", "1.333333333333333333");
        assert_div("2.3", "-3.8", "-0.605263157894736842");
        assert_div("-2.3", "3.8", "-0.605263157894736842");
        assert_div("-2.3", "-3.8", "0.605263157894736842");
        assert_div("1.01", "-7.02", "-0.143874643874643874");
        assert_div("-1.01", "7.02", "-0.143874643874643874");
        assert_div("-1.01", "-7.02", "0.143874643874643874");
    }

    #[test]
    fn div_extremes() {
        assert_div("0.0", super::MIN_STR, "0.0");
        assert_div("0.0", super::MAX_STR, "0.0");
        assert_div(super::MIN_STR, "1.0", super::MIN_STR);
        assert_div(super::MAX_STR, "1.0", super::MAX_STR);
    }
}
