use std::sync::Arc;

use arrs_buffer::Buffer;

/// An uncompressed bitmap implementation, optimized for dense bitmaps.
#[derive(Clone)]
pub struct Bitmap {
    buf: Arc<Buffer>,
    num_bits: usize,
}

impl Bitmap {
    /// Create a bitmap from buffer
    ///
    /// # Panics
    ///
    /// Panics if given buffer can't hold the given number of bits.
    pub fn from_buf(buf: Arc<Buffer>, num_bits: usize) -> Self {
        let num_bytes = num_bits.checked_next_multiple_of(8).unwrap() / 8;
        assert!(num_bytes <= buf.len());

        Self { buf, num_bits }
    }

    /// Number of bits in this bitmap
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Slices the bitmap with given range.
    ///
    /// Beware! If start_bit != 0, this will allocate a new Buffer and move the whole sliced area to that buffer.
    ///
    /// # Panics
    ///
    /// Panics if given range is outside of the bitmap.
    pub fn slice(&self, start_bit: usize, num_bits: usize) -> Self {
        assert!(start_bit.checked_add(num_bits).unwrap() <= self.num_bits);

        if start_bit == 0 {
            return Self {
                buf: self.buf.clone(),
                num_bits,
            };
        }

        let num_bytes = num_bits.checked_next_multiple_of(8).unwrap() / 8;

        let mut buf = Buffer::new(num_bytes);

        let start_word = start_bit / 64;
        let shift = start_bit % 64;

        unsafe {
            crate::compute::re_align(
                (self.buf.as_ptr() as *const u64).add(start_word),
                buf.as_mut_ptr() as *mut u64,
                ((num_bits + 63) / 64) + 1,
                shift as u32,
            );
        };

        Self {
            buf: Arc::new(buf),
            num_bits,
        }
    }

    /// Returns the set of set bit ranges in the bitmap
    pub fn set_ranges(&self) -> Vec<(usize, usize)> {
        if self.num_bits == 0 {
            return Vec::new();
        }

        let len = self.buf.len().next_multiple_of(64);

        unsafe { crate::compute::set_ranges(self.buf.as_ptr(), len) }
    }

    pub fn from_bools(bools: &[bool]) -> Self {
        let num_bits = bools.len();
        let num_bytes = num_bits.checked_next_multiple_of(8).unwrap() / 8;

        let mut buf = Buffer::new(num_bytes);

        // Compiler vectorizes this
        unsafe {
            let mut bools_ptr = bools.as_ptr() as *const u8;
            let mut buf_ptr = buf.as_mut_ptr();
            for _ in 0..num_bits / 8 {
                let byte = *bools_ptr
                    | *bools_ptr.add(1) << 1
                    | *bools_ptr.add(2) << 2
                    | *bools_ptr.add(3) << 3
                    | *bools_ptr.add(4) << 4
                    | *bools_ptr.add(5) << 5
                    | *bools_ptr.add(6) << 6
                    | *bools_ptr.add(7) << 7;

                *buf_ptr = byte;

                bools_ptr = bools_ptr.add(8);
                buf_ptr = buf_ptr.add(1);
            }

            if num_bits % 8 > 0 {
                let mut byte = 0;
                for (shift, _) in (0..num_bits % 8).enumerate() {
                    byte |= *bools_ptr << shift;
                    bools_ptr = bools_ptr.add(1);
                }
                *buf_ptr = byte;
            }
        }

        Self {
            buf: Arc::new(buf),
            num_bits,
        }
    }

    #[inline(always)]
    /// Returns a shared pointer to the underlying buffer
    pub fn buf(&self) -> Arc<Buffer> {
        self.buf.clone()
    }

    #[inline(always)]
    pub fn get(&self, bit_index: usize) -> Option<bool> {
        if bit_index >= self.num_bits {
            return None;
        }

        Some(unsafe { self.get_unchecked(bit_index) })
    }

    /// # Safety
    ///
    /// `bit_index` should be less than `self.num_bits()`
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, bit_index: usize) -> bool {
        let byte_index = bit_index / 8;
        let mask = 1 << (bit_index % 8);

        unsafe { (*self.buf.as_ptr().add(byte_index) & mask) != 0 }
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;

    fn run_test(bools: &[bool]) {
        let bitmap = Bitmap::from_bools(bools);

        for i in 0..bools.len() {
            assert_eq!(bools[i], bitmap.get(i).unwrap());
        }

        let offset = bitmap.num_bits() / 2;
        let new_len = bitmap.num_bits().saturating_sub(offset);
        let shifted_bitmap = bitmap.slice(offset, new_len);

        for i in 0..new_len {
            assert_eq!(
                bitmap.get(i + offset).unwrap(),
                shifted_bitmap.get(i).unwrap(),
                "failed at idx {}, len is: {}",
                i,
                bools.len()
            );
        }
    }

    fn generate(len: usize) -> Vec<bool> {
        let mut buf = Vec::with_capacity(len);

        let mut rng = ChaCha8Rng::seed_from_u64(0);

        for _ in 0..len {
            buf.push(rng.gen_bool(0.5));
        }

        buf
    }

    #[test]
    fn test_all() {
        run_test(&[]);
        run_test(&generate(1));
        run_test(&generate(2));
        run_test(&generate(3));
        run_test(&generate(7));
        run_test(&generate(8));
        run_test(&generate(9));
        run_test(&generate(32));
        run_test(&generate(63));
        run_test(&generate(64));
        run_test(&generate(65));
        run_test(&generate(127));
        run_test(&generate(128));
        run_test(&generate(129));
        //run_test(&generate(1024));
        run_test(&generate(1023));
        run_test(&generate(123123));
    }
}
