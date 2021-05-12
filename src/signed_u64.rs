use std::fmt;

/// A 64-bit integer that is *not* stored in two's compliment.
/// Instead, the first bit indicates the sign.
///
/// This means that neither negation nor absolute value can overflow, because
/// SignedU64::MAX == -SignedU64::MIN, unlike i64 where -i64::MIN overflows.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct SignedU64(u64);

impl fmt::Debug for SignedU64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_negative() {
            write!(f, "-")?;
        }

        write!(f, "{}", self.as_u64())
    }
}

#[cfg(target_endian = "big")]
const SIGN: u64 = 0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000;

#[cfg(target_endian = "big")]
const INVERTED_SIGN: u64 =
    0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111;

#[cfg(target_endian = "little")]
const SIGN: u64 = 0b0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_1000_0000;

#[cfg(target_endian = "little")]
const INVERTED_SIGN: u64 =
    0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_0111_1111;

impl SignedU64 {
    pub const MIN: Self = Self(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF);
    pub const MAX: Self = Self(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF ^ SIGN);

    pub fn is_positive(self) -> bool {
        self.0 & SIGN == 0
    }

    pub fn is_negative(self) -> bool {
        self.0 & SIGN != 0
    }

    pub fn abs(self) -> Self {
        Self(self.0 & INVERTED_SIGN)
    }

    pub fn to_negative(self) -> Self {
        Self(self.0 | SIGN)
    }
}

impl Into<u64> for SignedU64 {
    fn into(self) -> u64 {
        self.abs().0
    }
}

impl Into<i64> for SignedU64 {
    fn into(self) -> i64 {
        let u64 = self.abs().0;

        if self.is_positive() {
            u64 as i64
        } else {
            -(u64 as i64)
        }
    }
}

impl std::ops::Neg for SignedU64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(self.0 ^ SIGN)
    }
}

impl std::ops::Add for SignedU64 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self(self.0 ^ SIGN)
    }
}

impl std::convert::TryFrom<i64> for SignedU64 {
    type Error = ();

    fn try_from(value: i64) -> Result<Self, ()> {
        if value != i64::MIN {
            if value.is_positive() {
                Ok(Self(value as u64))
            } else {
                Ok(Self(value as u64).abs())
            }
        } else {
            Err(())
        }
    }
}
