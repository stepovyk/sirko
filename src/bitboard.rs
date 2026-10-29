//! Бітборди й таблиці атак. Магічні числа для слонів і тур шукає сам рушій на старті
//! (детермінований генератор: щоразу ті самі числа, кілька мілісекунд).

use crate::types::*;
use std::sync::OnceLock;

pub const FILE_A: Bitboard = 0x0101_0101_0101_0101;
pub const FILE_H: Bitboard = FILE_A << 7;
pub const RANK_1: Bitboard = 0xFF;
pub const RANK_2: Bitboard = RANK_1 << 8;
pub const RANK_4: Bitboard = RANK_1 << 24;
pub const RANK_5: Bitboard = RANK_1 << 32;
pub const RANK_7: Bitboard = RANK_1 << 48;
pub const RANK_8: Bitboard = RANK_1 << 56;

#[inline(always)]
pub fn bit(sq: Square) -> Bitboard {
    1u64 << sq
}

#[inline(always)]
pub fn lsb(b: Bitboard) -> Square {
    b.trailing_zeros() as Square
}

#[inline(always)]
pub fn pop_lsb(b: &mut Bitboard) -> Square {
    let sq = lsb(*b);
    *b &= *b - 1;
    sq
}

#[inline(always)]
pub fn count(b: Bitboard) -> u32 {
    b.count_ones()
}

#[inline(always)]
pub fn more_than_one(b: Bitboard) -> bool {
    b & b.wrapping_sub(1) != 0
}

pub fn file_bb(file: u8) -> Bitboard {
    FILE_A << file
}

pub fn rank_bb(rank: u8) -> Bitboard {
    RANK_1 << (8 * rank)
}

struct Magic {
    mask: Bitboard,
    magic: u64,
    shift: u32,
    offset: usize,
}

pub struct Tables {
    pub knight: [Bitboard; 64],
    pub king: [Bitboard; 64],
    pub pawn: [[Bitboard; 64]; 2],
    /// Поля строго між двома полями на одній лінії (інакше 0).
    pub between: Box<[[Bitboard; 64]; 64]>,
    /// Уся лінія через два поля (інакше 0).
    pub line: Box<[[Bitboard; 64]; 64]>,
    bishop: Vec<Magic>,
    rook: Vec<Magic>,
    slider: Vec<Bitboard>,
}

static TABLES: OnceLock<Tables> = OnceLock::new();

#[inline(always)]
pub fn tables() -> &'static Tables {
    TABLES.get_or_init(build)
}

pub fn init() {
    let _ = tables();
}

#[inline(always)]
pub fn knight_attacks(sq: Square) -> Bitboard {
    tables().knight[sq as usize]
}

#[inline(always)]
pub fn king_attacks(sq: Square) -> Bitboard {
    tables().king[sq as usize]
}

#[inline(always)]
pub fn pawn_attacks(color: usize, sq: Square) -> Bitboard {
    tables().pawn[color][sq as usize]
}

#[inline(always)]
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let t = tables();
    let m = &t.bishop[sq as usize];
    t.slider[m.offset + (((occ & m.mask).wrapping_mul(m.magic)) >> m.shift) as usize]
}

#[inline(always)]
pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let t = tables();
    let m = &t.rook[sq as usize];
    t.slider[m.offset + (((occ & m.mask).wrapping_mul(m.magic)) >> m.shift) as usize]
}

#[inline(always)]
pub fn queen_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    bishop_attacks(sq, occ) | rook_attacks(sq, occ)
}

#[inline(always)]
pub fn between(a: Square, b: Square) -> Bitboard {
    tables().between[a as usize][b as usize]
}

#[inline(always)]
pub fn line(a: Square, b: Square) -> Bitboard {
    tables().line[a as usize][b as usize]
}

/// Атаки фігури виду kind (не пішака) з поля sq.
#[inline(always)]
pub fn attacks(kind: usize, sq: Square, occ: Bitboard) -> Bitboard {
    match kind {
        KNIGHT => knight_attacks(sq),
        BISHOP => bishop_attacks(sq, occ),
        ROOK => rook_attacks(sq, occ),
        QUEEN => queen_attacks(sq, occ),
        KING => king_attacks(sq),
        _ => 0,
    }
}

fn step_attacks(sq: Square, deltas: &[(i8, i8)]) -> Bitboard {
    let (f, r) = (file_of(sq) as i8, rank_of(sq) as i8);
    let mut b = 0;
    for &(df, dr) in deltas {
        let (nf, nr) = (f + df, r + dr);
        if (0..8).contains(&nf) && (0..8).contains(&nr) {
            b |= bit(make_sq(nf as u8, nr as u8));
        }
    }
    b
}

const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ROOK_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// Повільні атаки променем: лише для побудови таблиць.
fn ray_attacks(sq: Square, occ: Bitboard, dirs: &[(i8, i8)]) -> Bitboard {
    let (f, r) = (file_of(sq) as i8, rank_of(sq) as i8);
    let mut b = 0;
    for &(df, dr) in dirs {
        let (mut nf, mut nr) = (f + df, r + dr);
        while (0..8).contains(&nf) && (0..8).contains(&nr) {
            let s = bit(make_sq(nf as u8, nr as u8));
            b |= s;
            if occ & s != 0 {
                break;
            }
            nf += df;
            nr += dr;
        }
    }
    b
}

/// Маска для магії: промені без крайніх полів.
fn relevant_mask(sq: Square, dirs: &[(i8, i8)]) -> Bitboard {
    let (f, r) = (file_of(sq) as i8, rank_of(sq) as i8);
    let mut b = 0;
    for &(df, dr) in dirs {
        let (mut nf, mut nr) = (f + df, r + dr);
        while (0..8).contains(&(nf + df)) && (0..8).contains(&(nr + dr)) {
            b |= bit(make_sq(nf as u8, nr as u8));
            nf += df;
            nr += dr;
        }
    }
    b
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

fn find_magics(dirs: &[(i8, i8)], slider: &mut Vec<Bitboard>, rng: &mut Rng) -> Vec<Magic> {
    let mut out = Vec::with_capacity(64);
    for sq in 0..64u8 {
        let mask = relevant_mask(sq, dirs);
        let bits = count(mask);
        let size = 1usize << bits;
        // Усі підмножини маски (Carry-Rippler) і справжні атаки для кожної.
        let mut occs = Vec::with_capacity(size);
        let mut atts = Vec::with_capacity(size);
        let mut sub: Bitboard = 0;
        loop {
            occs.push(sub);
            atts.push(ray_attacks(sq, sub, dirs));
            sub = sub.wrapping_sub(mask) & mask;
            if sub == 0 {
                break;
            }
        }
        let shift = 64 - bits;
        let mut table = vec![0u64; size];
        let mut used = vec![0u32; size];
        let mut epoch = 0u32;
        let magic = loop {
            let candidate = rng.sparse();
            if count((mask.wrapping_mul(candidate)) >> 56) < 6 {
                continue;
            }
            epoch += 1;
            let mut ok = true;
            for i in 0..size {
                let idx = (occs[i].wrapping_mul(candidate) >> shift) as usize;
                if used[idx] != epoch {
                    used[idx] = epoch;
                    table[idx] = atts[i];
                } else if table[idx] != atts[i] {
                    ok = false;
                    break;
                }
            }
            if ok {
                break candidate;
            }
        };
        let offset = slider.len();
        let mut entries = vec![0u64; size];
        for i in 0..size {
            entries[(occs[i].wrapping_mul(magic) >> shift) as usize] = atts[i];
        }
        slider.extend_from_slice(&entries);
        out.push(Magic {
            mask,
            magic,
            shift,
            offset,
        });
    }
    out
}

fn build() -> Tables {
    let mut knight = [0; 64];
    let mut king = [0; 64];
    let mut pawn = [[0; 64]; 2];
    for sq in 0..64u8 {
        knight[sq as usize] = step_attacks(
            sq,
            &[
                (1, 2),
                (2, 1),
                (2, -1),
                (1, -2),
                (-1, -2),
                (-2, -1),
                (-2, 1),
                (-1, 2),
            ],
        );
        king[sq as usize] = step_attacks(
            sq,
            &[
                (1, 0),
                (1, 1),
                (0, 1),
                (-1, 1),
                (-1, 0),
                (-1, -1),
                (0, -1),
                (1, -1),
            ],
        );
        pawn[WHITE][sq as usize] = step_attacks(sq, &[(1, 1), (-1, 1)]);
        pawn[BLACK][sq as usize] = step_attacks(sq, &[(1, -1), (-1, -1)]);
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut slider = Vec::new();
    let bishop = find_magics(&BISHOP_DIRS, &mut slider, &mut rng);
    let rook = find_magics(&ROOK_DIRS, &mut slider, &mut rng);

    let mut between = Box::new([[0u64; 64]; 64]);
    let mut line = Box::new([[0u64; 64]; 64]);
    for a in 0..64u8 {
        for &(df, dr) in BISHOP_DIRS.iter().chain(ROOK_DIRS.iter()) {
            // Один напрям за раз: проміжні поля до кожного b на цьому промені і вся лінія через a.
            let forward = ray_attacks(a, 0, &[(df, dr)]);
            let backward = ray_attacks(a, 0, &[(-df, -dr)]);
            let full = forward | backward | bit(a);
            let mut b = forward;
            while b != 0 {
                let t = pop_lsb(&mut b);
                between[a as usize][t as usize] =
                    forward & ray_attacks(t, 0, &[(-df, -dr)]) & !bit(a);
                line[a as usize][t as usize] = full;
            }
        }
    }
    Tables {
        knight,
        king,
        pawn,
        between,
        line,
        bishop,
        rook,
        slider,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magics_match_rays() {
        let mut rng = Rng(12345);
        for _ in 0..20000 {
            let occ = rng.next() & rng.next();
            let sq = (rng.next() % 64) as u8;
            assert_eq!(bishop_attacks(sq, occ), ray_attacks(sq, occ, &BISHOP_DIRS));
            assert_eq!(rook_attacks(sq, occ), ray_attacks(sq, occ, &ROOK_DIRS));
        }
    }

    #[test]
    fn between_and_line() {
        assert_eq!(
            between(0, 63),
            (1..63)
                .filter(|s| s % 9 == 0)
                .map(|s| bit(s as u8))
                .fold(0, |a, b| a | b)
        );
        assert_eq!(between(0, 7), 0x7E);
        assert_eq!(between(0, 10), 0);
        assert_eq!(line(0, 7), RANK_1);
    }
}
