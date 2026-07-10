//! Faithful transpile of Masstree's `kpermuter<15>` (masstree-beta,
//! kpermuter.hh) — the packed key-slot permutation of a leaf node.
//!
//! One 64-bit word holds a whole permutation:
//!
//! ```text
//! bits [0, 4)              size: number of used logical positions, 0..=15
//! bits [4i+4, 4i+8)        physical slot (0..=14) at logical position i
//! ```
//!
//! The 15 slot nibbles are always a permutation of {0, ..., 14}. Logical
//! positions >= size() hold the free slots in allocation order; allocation
//! always takes back(), the slot at logical position 14. A writer computes
//! a new word and publishes it with one store, so a reader of the single
//! word sees a fully consistent ordering. The atomic storage lives in the
//! leaf node, not here — this is a pure value type.
//!
//! Transpile rules: every shift and mask matches the C++ bit for bit. C++
//! unsigned arithmetic wraps, so wrapping_add/wrapping_sub replace +/-
//! wherever the C++ result depends on wraparound (e.g. `(256 << 56) - 1`,
//! where the 256 has already overflowed to 0 and the -1 must yield
//! all-ones; a plain `-` would panic in debug builds). Shift *amounts*
//! never reach 64: the reference's `n == W` / `i == W` special cases,
//! which sidestep exactly that, are preserved verbatim.
//!
//! Provenance: this file was produced by transpiling kpermuter.hh twice with
//! two independent models and reconciling. Both arrived at bit-identical
//! packing arithmetic (strong triangulation for a pure-bit-twiddling type),
//! and both correctly followed the C++ source over a mistaken test
//! expectation in the porting brief (make_empty lays the free slots out in
//! reverse, so `self[i] == 14 - i`; allocation still hands them out 0..14
//! because it pops back()).

/// W: key slots in a leaf. The word has room for max_width slot nibbles
/// beside the size nibble, so W == MAX_WIDTH for 64-bit storage.
pub const WIDTH: i32 = 15;
const MAX_WIDTH: i32 = (core::mem::size_of::<u64>() * 2 - 1) as i32;

/* sized_kpermuter_info<2> (the 64-bit specialization). INITIAL_VALUE lists
 * the free slots in reverse physical order so that allocation, which pops
 * the topmost nibble, hands out 0, 1, ..., 14; FULL_VALUE is the fully
 * sorted permutation. */
const INITIAL_VALUE: u64 = 0x0123_4567_89AB_CDE0;
const FULL_VALUE: u64 = 0xEDCB_A987_6543_2100;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Permuter(u64);

impl Permuter {
    /// Value of an empty permuter: size 0, elements allocated in order
    /// 0, 1, ..., WIDTH-1.
    pub fn make_empty() -> u64 {
        /* Narrower widths share one initial_value per storage size; the
         * shift discards unused top nibbles. It is 0 for W == 15, but the
         * computation is kept as in the reference. */
        let p: u64 = INITIAL_VALUE >> ((MAX_WIDTH - WIDTH) << 2);
        p & !15u64
    }

    /// Value with size n and self[i] == i for 0 <= i < n. Slots n through
    /// WIDTH-1 are free and will be allocated in that order.
    pub fn make_sorted(n: i32) -> u64 {
        /* The reference special-cases n == W; kept verbatim even though
         * for u64 the unguarded `16 << 60` would wrap to the same value. */
        let mask = (if n == WIDTH { 0u64 } else { 16u64 << (n << 2) }).wrapping_sub(1);
        (Self::make_empty() << (n << 2)) | (FULL_VALUE & mask) | n as u64
    }

    pub fn from_value(x: u64) -> Self {
        Permuter(x)
    }

    pub fn value(&self) -> u64 {
        self.0
    }

    /// Size of a raw permuter word without unpacking it.
    pub fn size_of(x: u64) -> i32 {
        (x & 15) as i32
    }

    pub fn size(&self) -> i32 {
        (self.0 & 15) as i32
    }

    pub fn width() -> i32 {
        WIDTH
    }

    /// Physical slot at logical position i (operator[] in the reference).
    /// Requires 0 <= i < WIDTH.
    pub fn get(&self, i: i32) -> i32 {
        ((self.0 >> ((i << 2) + 4)) & 15) as i32
    }

    /// Slot at the last logical position: the next slot to be allocated.
    pub fn back(&self) -> i32 {
        self.get(WIDTH - 1)
    }

    /// The word with the size nibble and positions 0..=i-1 stripped off,
    /// so position i sits in the low nibble. Lets a reader scan a suffix
    /// with plain shifts.
    pub fn value_from(&self, i: i32) -> u64 {
        self.0 >> ((i + 1) << 2)
    }

    pub fn set_size(&mut self, n: i32) {
        self.0 = (self.0 & !15u64) | n as u64;
    }

    /// Allocate a new slot and insert it at logical position i; positions
    /// i and up shift toward the back. Returns the allocated slot.
    /// Requires 0 <= i < WIDTH and size() < WIDTH.
    ///
    /// With `q = p; let x = q.insert_from_back(i);`:
    ///   q.size() == p.size() + 1
    ///   q[j] == p[j] && q[j] != x    for 0 <= j < i
    ///   q[i] == x
    ///   q[j] == p[j-1] && q[j] != x  for i < j < q.size()
    pub fn insert_from_back(&mut self, i: i32) -> i32 {
        let value = self.back();
        // increment size, leave lower slots unchanged
        self.0 = (self.0.wrapping_add(1) & (16u64 << (i << 2)).wrapping_sub(1))
            // insert slot
            | ((value as u64) << ((i << 2) + 4))
            // shift up unchanged higher entries & empty slots
            | ((self.0 << 4) & !((256u64 << (i << 2)).wrapping_sub(1)));
        value
    }

    /// Insert the unallocated slot at position si at position di; the
    /// positions between shift toward the back, positions above si are
    /// untouched. Requires 0 <= di < WIDTH and size() <= si < WIDTH.
    pub fn insert_selected(&mut self, di: i32, si: i32) {
        let value = self.get(si);
        let mask = (256u64 << (si << 2)).wrapping_sub(1);
        // increment size, leave lower slots unchanged
        self.0 = (self.0.wrapping_add(1) & (16u64 << (di << 2)).wrapping_sub(1))
            // insert slot
            | ((value as u64) << ((di << 2) + 4))
            // shift up unchanged higher entries & empty slots
            | ((self.0 << 4) & mask & !((256u64 << (di << 2)).wrapping_sub(1)))
            // leave uppermost slots alone
            | (self.0 & !mask);
    }

    /// Remove the slot at position i; higher used positions shift down and
    /// the removed slot parks at the first free position.
    /// Requires 0 <= i < size().
    ///
    /// With `q = p; q.remove(i);`:
    ///   q.size() == p.size() - 1
    ///   q[j] == p[j]    for 0 <= j < i
    ///   q[j] == p[j+1]  for i <= j < q.size()
    ///   q[q.size()] == p[i]
    pub fn remove(&mut self, i: i32) {
        if (self.0 & 15) as i32 == i + 1 {
            // removing the last used position: nothing moves
            self.0 = self.0.wrapping_sub(1);
        } else {
            /* rotate the removed slot over positions i..size; positions
             * beyond size (the free slots) stay put */
            let rot_amount = (((self.0 & 15) as i32) - i - 1) << 2;
            let rot_mask = (16u64 << rot_amount).wrapping_sub(1) << ((i + 1) << 2);
            // decrement size, leave lower slots unchanged
            self.0 = (self.0.wrapping_sub(1) & !rot_mask)
                // shift higher entries down
                | (((self.0 & rot_mask) >> 4) & rot_mask)
                // shift value up
                | (((self.0 & rot_mask) << rot_amount) & rot_mask);
        }
    }

    /// Remove the slot at position i; ALL higher positions (free slots
    /// included) shift down and the removed slot parks at the very back,
    /// behind every other free slot. Requires 0 <= i < size().
    ///
    /// With `q = p; q.remove_to_back(i);`:
    ///   q.size() == p.size() - 1
    ///   q[j] == p[j]    for 0 <= j < i
    ///   q[j] == p[j+1]  for i <= j < WIDTH - 1
    ///   q.back() == p[i]
    pub fn remove_to_back(&mut self, i: i32) {
        let mask = !((16u64 << (i << 2)).wrapping_sub(1));
        // clear unused slots
        /* `16 << 60` wraps to 0 and the -1 to all-ones: with W == 15 every
         * nibble of the word is in use. Kept as written. */
        let x = self.0 & (16u64 << (WIDTH << 2)).wrapping_sub(1);
        // decrement size, leave lower slots unchanged
        self.0 = (x.wrapping_sub(1) & !mask)
            // shift higher entries down
            | ((x >> 4) & mask)
            // shift removed element up
            | ((x & mask) << ((WIDTH - i - 1) << 2));
    }

    /// Left-rotate positions i..WIDTH by (j - i): q[i] == p[j], and the
    /// slots that fall off the front wrap to the very back. Free slots
    /// rotate along with used ones. Requires 0 <= i <= j <= size().
    pub fn rotate(&mut self, i: i32, j: i32) {
        /* i == W special case kept verbatim (see make_sorted) */
        let mask = (if i == WIDTH { 0u64 } else { 16u64 << (i << 2) }).wrapping_sub(1);
        // clear unused slots
        let x = self.0 & (16u64 << (WIDTH << 2)).wrapping_sub(1);
        self.0 = (x & mask)
            | ((x >> ((j - i) << 2)) & !mask)
            | ((x & !mask) << ((WIDTH - j) << 2));
    }

    /// Exchange the slots at positions i and j.
    pub fn exchange(&mut self, i: i32, j: i32) {
        /* branchless nibble swap: 240 masks bits [4i+4, 4i+8) after the
         * (i<<2) shift, i.e. exactly the slot nibble of position i */
        let diff = ((self.0 >> (i << 2)) ^ (self.0 >> (j << 2))) & 240;
        self.0 ^= (diff << (i << 2)) | (diff << (j << 2));
    }

    /// Exchange the POSITIONS of slot values x and y, wherever they sit.
    pub fn exchange_values(&mut self, x: i32, y: i32) {
        let mut diff: u64 = 0;
        let mut p = self.0;
        /* Walk the nibbles top-down, accumulating x^y at every position
         * that holds either value; one final xor then flips both. The
         * trailing shifts run after every pass, mirroring the C++ for-loop
         * increment, so pass k's contribution lands at nibble 15-k. */
        for _ in 0..WIDTH {
            let v = ((p >> (WIDTH << 2)) & 15) as i32;
            diff ^= ((-(((v == x) as i32) | ((v == y) as i32))) & (x ^ y)) as u64;
            diff <<= 4;
            p <<= 4;
        }
        self.0 ^= diff;
    }
}

impl core::fmt::Debug for Permuter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Permuter({:#018x})", self.0)
    }
}

/// Debug oracle: the 15 slot nibbles must be a permutation of {0,...,14}
/// no matter the size. Mirrors the `seen` check in the reference's
/// unparse().
pub fn is_valid_permutation(x: u64) -> bool {
    let mut seen: u32 = 0;
    let mut p = x >> 4;
    for _ in 0..WIDTH {
        seen |= 1u32 << (p & 15);
        p >>= 4;
    }
    seen == (1u32 << WIDTH) - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    const W: i32 = WIDTH;

    fn assert_valid(p: &Permuter) {
        assert!(is_valid_permutation(p.value()), "invalid permutation: {:?}", p);
    }

    /// Deterministic xorshift so test inputs are reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: i32) -> i32 {
            (self.next() % n as u64) as i32
        }
    }

    /// Arbitrary valid permuter: shuffle the full sorted permuter with
    /// exchanges (validity-preserving by construction), then set the size.
    fn shuffled(rng: &mut Rng, n: i32) -> Permuter {
        let mut p = Permuter::from_value(Permuter::make_sorted(W));
        for _ in 0..64 {
            let i = rng.below(W);
            let j = rng.below(W);
            p.exchange(i, j);
        }
        p.set_size(n);
        assert_valid(&p);
        p
    }

    #[test]
    fn make_empty_contract() {
        let p = Permuter::from_value(Permuter::make_empty());
        assert_eq!(p.value(), 0x0123_4567_89AB_CDE0);
        assert_eq!(p.size(), 0);
        assert_valid(&p);
        // Free slots sit in REVERSE physical order: allocation pops back(),
        // so elements are handed out 0, 1, ..., 14 — the documented
        // "allocated in order 0, 1, ..., width - 1".
        for i in 0..W {
            assert_eq!(p.get(i), W - 1 - i);
        }
        let mut q = p;
        for k in 0..W {
            assert_eq!(q.back(), k);
            assert_eq!(q.insert_from_back(q.size()), k);
            assert_valid(&q);
        }
        assert_eq!(q.size(), 15);
    }

    #[test]
    fn make_sorted_contract() {
        for n in 0..=W {
            let p = Permuter::from_value(Permuter::make_sorted(n));
            assert_eq!(p.size(), n, "n={}", n);
            assert_valid(&p);
            for i in 0..n {
                assert_eq!(p.get(i), i, "n={} i={}", n, i);
            }
            // remaining elements are free, allocated in order n, n+1, ...
            let mut q = p;
            for k in n..W {
                assert_eq!(q.insert_from_back(q.size()), k, "n={} k={}", n, k);
                assert_valid(&q);
            }
        }
        assert_eq!(Permuter::make_sorted(0), Permuter::make_empty());
        assert_eq!(Permuter::make_sorted(15), 0xEDCB_A987_6543_210F);
    }

    #[test]
    fn make_sorted_matches_incremental_inserts() {
        // make_sorted(n) must equal n appends starting from empty, since
        // allocation order is 0, 1, ... — full word equality, not just
        // logical equality.
        for n in 0..=W {
            let mut q = Permuter::from_value(Permuter::make_empty());
            for i in 0..n {
                q.insert_from_back(i);
            }
            assert_eq!(q.value(), Permuter::make_sorted(n), "n={}", n);
        }
    }

    #[test]
    fn insert_from_back_postconditions() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for n in 0..W {
            for trial in 0..8 {
                let p = if trial == 0 {
                    Permuter::from_value(Permuter::make_sorted(n))
                } else {
                    shuffled(&mut rng, n)
                };
                for i in 0..=n {
                    let mut q = p;
                    let x = q.insert_from_back(i);
                    assert_valid(&q);
                    assert_eq!(x, p.back());
                    assert_eq!(q.size(), p.size() + 1);
                    for j in 0..i {
                        assert_eq!(q.get(j), p.get(j));
                        assert_ne!(q.get(j), x);
                    }
                    assert_eq!(q.get(i), x);
                    for j in (i + 1)..q.size() {
                        assert_eq!(q.get(j), p.get(j - 1));
                        assert_ne!(q.get(j), x);
                    }
                }
            }
        }
    }

    #[test]
    fn insert_selected_postconditions() {
        let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
        for n in 0..W {
            for _ in 0..8 {
                let p = shuffled(&mut rng, n);
                for si in n..W {
                    for di in 0..=n {
                        let mut q = p;
                        q.insert_selected(di, si);
                        assert_valid(&q);
                        assert_eq!(q.size(), n + 1);
                        for j in 0..di {
                            assert_eq!(q.get(j), p.get(j));
                        }
                        assert_eq!(q.get(di), p.get(si));
                        for j in (di + 1)..=si {
                            assert_eq!(q.get(j), p.get(j - 1));
                        }
                        for j in (si + 1)..W {
                            assert_eq!(q.get(j), p.get(j));
                        }
                    }
                }
            }
        }
        // si == W-1 must degenerate to insert_from_back (full word equality)
        let p = shuffled(&mut rng, 7);
        for di in 0..=7 {
            let mut a = p;
            let mut b = p;
            a.insert_selected(di, W - 1);
            b.insert_from_back(di);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn remove_postconditions() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for n in 1..=W {
            for _ in 0..8 {
                let p = shuffled(&mut rng, n);
                for i in 0..n {
                    let mut q = p;
                    q.remove(i);
                    assert_valid(&q);
                    assert_eq!(q.size(), n - 1);
                    for j in 0..i {
                        assert_eq!(q.get(j), p.get(j));
                    }
                    for j in i..q.size() {
                        assert_eq!(q.get(j), p.get(j + 1));
                    }
                    assert_eq!(q.get(q.size()), p.get(i));
                    // free slots beyond the parked one stay put
                    for j in (q.size() + 1)..W {
                        assert_eq!(q.get(j), p.get(j));
                    }
                }
            }
        }
    }

    #[test]
    fn remove_to_back_postconditions() {
        let mut rng = Rng(0xFEED_FACE_0123_4567);
        for n in 1..=W {
            for _ in 0..8 {
                let p = shuffled(&mut rng, n);
                for i in 0..n {
                    let mut q = p;
                    q.remove_to_back(i);
                    assert_valid(&q);
                    assert_eq!(q.size(), n - 1);
                    for j in 0..i {
                        assert_eq!(q.get(j), p.get(j));
                    }
                    for j in i..(W - 1) {
                        assert_eq!(q.get(j), p.get(j + 1));
                    }
                    assert_eq!(q.back(), p.get(i));
                }
            }
        }
    }

    #[test]
    fn rotate_postconditions() {
        let mut rng = Rng(0x0F0F_1E1E_2D2D_3C3C);
        for n in 0..=W {
            for _ in 0..4 {
                let p = shuffled(&mut rng, n);
                for i in 0..=n {
                    for j in i..=n {
                        let mut q = p;
                        q.rotate(i, j);
                        assert_valid(&q);
                        assert_eq!(q.size(), n);
                        for k in 0..i {
                            assert_eq!(q.get(k), p.get(k));
                        }
                        // The arithmetic rotates the FULL window [i, W),
                        // free slots included: q[k] = p[i + (k-i + j-i) mod
                        // (W-i)]. (The reference @brief writes the modulus
                        // as size()-i, which only matches the used prefix
                        // when the window is full; the code does W-i.)
                        for k in i..W {
                            let src = i + (k - i + (j - i)) % (W - i);
                            assert_eq!(q.get(k), p.get(src), "n={} i={} j={} k={}", n, i, j, k);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exchange_postconditions() {
        let mut rng = Rng(0xAAAA_BBBB_CCCC_DDDD);
        for n in [0, 3, 15] {
            let p = shuffled(&mut rng, n);
            for i in 0..W {
                for j in 0..W {
                    let mut q = p;
                    q.exchange(i, j);
                    assert_valid(&q);
                    assert_eq!(q.size(), p.size());
                    assert_eq!(q.get(i), p.get(j));
                    assert_eq!(q.get(j), p.get(i));
                    for k in 0..W {
                        if k != i && k != j {
                            assert_eq!(q.get(k), p.get(k));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exchange_values_postconditions() {
        let mut rng = Rng(0x1357_9BDF_0246_8ACE);
        for n in [0, 5, 15] {
            let p = shuffled(&mut rng, n);
            for x in 0..W {
                for y in 0..W {
                    let mut q = p;
                    q.exchange_values(x, y);
                    assert_valid(&q);
                    assert_eq!(q.size(), p.size());
                    for k in 0..W {
                        let expect = if p.get(k) == x {
                            y
                        } else if p.get(k) == y {
                            x
                        } else {
                            p.get(k)
                        };
                        assert_eq!(q.get(k), expect, "x={} y={} k={}", x, y, k);
                    }
                }
            }
        }
    }

    #[test]
    fn accessors() {
        assert_eq!(Permuter::width(), 15);
        let p = Permuter::from_value(Permuter::make_sorted(9));
        assert_eq!(p.value(), Permuter::make_sorted(9));
        assert_eq!(p.size(), 9);
        assert_eq!(Permuter::size_of(p.value()), 9);
        assert_eq!(p.back(), p.get(14));
        // value_from(i) aligns position i at the low nibble
        for i in 0..W {
            let v = p.value_from(i);
            for j in i..W {
                assert_eq!(((v >> ((j - i) << 2)) & 15) as i32, p.get(j));
            }
        }
        let mut q = p;
        q.set_size(3);
        assert_eq!(q.size(), 3);
        assert_valid(&q);
        for i in 0..W {
            assert_eq!(q.get(i), p.get(i));
        }
        assert!(p == Permuter::from_value(p.value()));
        assert!(p != q);
    }

    /// Hand-computed words from stepping the C++ arithmetic on paper.
    /// These pin the exact packing, not just the logical view.
    #[test]
    fn gold_words() {
        assert_eq!(Permuter::make_empty(), 0x0123_4567_89AB_CDE0);
        assert_eq!(Permuter::make_sorted(2), 0x2345_6789_ABCD_E102);

        let mut p = Permuter::from_value(Permuter::make_empty());
        assert_eq!(p.insert_from_back(0), 0);
        assert_eq!(p.value(), 0x1234_5678_9ABC_DE01);

        let mut p = Permuter::from_value(Permuter::make_sorted(2));
        p.insert_selected(0, 3);
        assert_eq!(p.value(), 0x2345_6789_ABCE_10D3);

        let mut p = Permuter::from_value(Permuter::make_sorted(4));
        p.remove(1);
        assert_eq!(p.value(), 0x4567_89AB_CDE1_3203);

        let mut p = Permuter::from_value(Permuter::make_sorted(15));
        p.remove(0);
        assert_eq!(p.value(), 0x0EDC_BA98_7654_321E);

        let mut p = Permuter::from_value(Permuter::make_sorted(15));
        p.remove_to_back(0);
        assert_eq!(p.value(), 0x0EDC_BA98_7654_321E);

        let mut p = Permuter::from_value(Permuter::make_sorted(4));
        p.rotate(1, 2);
        assert_eq!(p.value(), 0x1456_789A_BCDE_3204);

        let mut p = Permuter::from_value(Permuter::make_sorted(15));
        p.rotate(0, 3);
        assert_eq!(p.value(), 0x210E_DCBA_9876_543F);

        let mut p = Permuter::from_value(Permuter::make_sorted(4));
        p.exchange(0, 2);
        assert_eq!(p.value(), 0x4567_89AB_CDE3_0124);
    }

    #[test]
    fn build_leaf_scenario() {
        // Build a leaf: 15 inserts at chosen logical positions, tracked
        // against a plain Vec model (used prefix ++ free suffix).
        let mut p = Permuter::from_value(Permuter::make_empty());
        let mut model: Vec<i32> = (0..15).rev().collect();
        let mut size: usize = 0;
        let positions = [0, 1, 1, 0, 4, 2, 3, 0, 8, 5, 10, 6, 1, 13, 7];
        for (step, &pos) in positions.iter().enumerate() {
            let v = model.remove(14);
            model.insert(pos as usize, v);
            let got = p.insert_from_back(pos);
            size += 1;
            assert_valid(&p);
            assert_eq!(got, v, "step {}", step);
            assert_eq!(p.size() as usize, size);
            for k in 0..15 {
                assert_eq!(p.get(k as i32), model[k], "step {} k {}", step, k);
            }
        }
        assert_eq!(p.size(), 15);
        // now remove a few, alternating remove / remove_to_back
        for &(i, to_back) in &[(3usize, false), (0, true), (7, false), (11, true), (2, false)] {
            let v = model.remove(i);
            if to_back {
                model.push(v);
                p.remove_to_back(i as i32);
            } else {
                model.insert(size - 1, v);
                p.remove(i as i32);
            }
            size -= 1;
            assert_valid(&p);
            assert_eq!(p.size() as usize, size);
            for k in 0..15 {
                assert_eq!(p.get(k as i32), model[k], "i {} k {}", i, k);
            }
        }
    }

    #[test]
    fn model_fuzz() {
        // 20k random ops against the Vec model; validity + full logical
        // equality after every op. Any wrong shift in the packing shows up
        // here within a few steps.
        let mut rng = Rng(0x0BAD_5EED_0BAD_5EED);
        let mut p = Permuter::from_value(Permuter::make_empty());
        let mut model: Vec<i32> = (0..15).rev().collect();
        let mut size: i32 = 0;
        for step in 0..20_000 {
            match rng.below(8) {
                0 if size < 15 => {
                    let i = rng.below(size + 1);
                    let v = model.remove(14);
                    model.insert(i as usize, v);
                    assert_eq!(p.insert_from_back(i), v);
                    size += 1;
                }
                1 if size > 0 => {
                    let i = rng.below(size);
                    let v = model.remove(i as usize);
                    model.insert((size - 1) as usize, v);
                    p.remove(i);
                    size -= 1;
                }
                2 if size > 0 => {
                    let i = rng.below(size);
                    let v = model.remove(i as usize);
                    model.push(v);
                    p.remove_to_back(i);
                    size -= 1;
                }
                3 => {
                    let i = rng.below(15);
                    let j = rng.below(15);
                    model.swap(i as usize, j as usize);
                    p.exchange(i, j);
                }
                4 if size < 15 => {
                    let si = size + rng.below(15 - size);
                    let di = rng.below(size + 1);
                    let v = model.remove(si as usize);
                    model.insert(di as usize, v);
                    p.insert_selected(di, si);
                    size += 1;
                }
                5 => {
                    let i = rng.below(size + 1);
                    let j = i + rng.below(size - i + 1);
                    if i < 15 {
                        model[i as usize..].rotate_left(((j - i) % (15 - i)) as usize);
                    }
                    p.rotate(i, j);
                }
                6 => {
                    let x = rng.below(15);
                    let y = rng.below(15);
                    for e in model.iter_mut() {
                        if *e == x {
                            *e = y;
                        } else if *e == y {
                            *e = x;
                        }
                    }
                    p.exchange_values(x, y);
                }
                7 => {
                    let n = rng.below(16);
                    p.set_size(n);
                    size = n;
                }
                _ => continue,
            }
            assert_eq!(p.size(), size, "step {}", step);
            assert_valid(&p);
            for k in 0..15 {
                assert_eq!(p.get(k), model[k as usize], "step {} k {}", step, k);
            }
        }
    }
}
