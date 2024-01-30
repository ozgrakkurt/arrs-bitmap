// This would potentially be faster with avx2.
//  Leaving the naive implementation since compiler seems to be able to optimize it ok.
/// # Safety
///
/// Both `src`` and `dst` must have at least `len` size. `shift` should be less than 64
pub unsafe fn re_align(mut src: *const u64, mut dst: *mut u64, len: usize, shift: u32) {
    if len == 0 {
        return;
    }

    if shift == 0 {
        core::ptr::copy_nonoverlapping(src, dst, len);
        return;
    }

    let right_shift = 64 - shift;

    let mut left = *src;

    for _ in 0..len - 1 {
        src = src.add(1);
        let right = *src;

        *dst = left.to_le() >> shift | right.to_le() << right_shift;

        dst = dst.add(1);

        left = right;
    }

    *dst = left.to_le() >> shift;
}
