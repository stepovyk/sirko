//! Оцінка позиції (для того, хто ходить), у сантипішаках. Кожен доданок має дві ваги –
//! для середини гри й для закінчення – і змішується за матеріальною фазою.

use crate::bitboard::*;
use crate::position::{Position, MAX_PHASE};
use crate::types::*;
use std::sync::OnceLock;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct S(pub i32, pub i32);

impl std::ops::Add for S {
    type Output = S;
    fn add(self, o: S) -> S {
        S(self.0 + o.0, self.1 + o.1)
    }
}
impl std::ops::Sub for S {
    type Output = S;
    fn sub(self, o: S) -> S {
        S(self.0 - o.0, self.1 - o.1)
    }
}
impl std::ops::AddAssign for S {
    fn add_assign(&mut self, o: S) {
        self.0 += o.0;
        self.1 += o.1;
    }
}
impl std::ops::SubAssign for S {
    fn sub_assign(&mut self, o: S) {
        self.0 -= o.0;
        self.1 -= o.1;
    }
}
impl std::ops::Mul<i32> for S {
    type Output = S;
    fn mul(self, k: i32) -> S {
        S(self.0 * k, self.1 * k)
    }
}

pub const VALUE: [S; 6] = [
    S(100, 125),
    S(330, 305),
    S(345, 320),
    S(480, 545),
    S(990, 1010),
    S(0, 0),
];
/// Прості значення фігур для впорядкування ходів і SEE.
pub const SEE_VALUE: [i32; 6] = [100, 320, 330, 500, 950, 20000];

const BISHOP_PAIR: S = S(30, 55);
const TEMPO: i32 = 15;
const ROOK_OPEN: S = S(40, 15);
const ROOK_SEMI: S = S(18, 10);
const DOUBLED: S = S(-10, -25);
const ISOLATED: S = S(-12, -14);
const PHALANX: S = S(6, 4);
const SUPPORTED: S = S(10, 8);
const PASSED: [S; 8] = [
    S(0, 0),
    S(2, 10),
    S(5, 15),
    S(12, 30),
    S(30, 60),
    S(55, 105),
    S(90, 170),
    S(0, 0),
];
const PASSED_FREE: S = S(0, 4); // поле перед прохідним вільне: за кожну горизонталь просування
const KING_SHIELD: S = S(14, 0);
const KING_OPEN_FILE: S = S(-22, 0);
// Рухливість: за кожне доступне поле, від нуля в «середньому» числі полів.
const MOBILITY: [S; 6] = [S(0, 0), S(5, 5), S(5, 6), S(3, 6), S(2, 4), S(0, 0)];
const MOBILITY_BASE: [i32; 6] = [0, 4, 6, 6, 12, 0];
// Атаки на зону короля: вага фігури; підсумок проходить через таблицю небезпеки.
const KING_ATTACK_WEIGHT: [i32; 6] = [0, 7, 5, 9, 14, 0];
const THREAT_BY_PAWN: S = S(45, 35); // пішак б'є фігуру
const THREAT_BY_MINOR: S = S(25, 25); // кінь чи слон б'є туру або ферзя
const HANGING: S = S(18, 12); // фігура під боєм і без захисту

struct Tables {
    pst: [[S; 64]; 12],
    king_danger: [i32; 128],
}

static TABLES: OnceLock<Tables> = OnceLock::new();

fn center(sq: Square) -> i32 {
    // 0 у куті, 6 у центрі.
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    let df = if f < 4 { f } else { 7 - f };
    let dr = if r < 4 { r } else { 7 - r };
    df + dr
}

/// Позиційні таблиці з білого боку (поле – з погляду білих).
fn pst_white(kind: usize, sq: Square) -> S {
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    let c = center(sq);
    match kind {
        PAWN => {
            if r == 0 || r == 7 {
                return S(0, 0);
            }
            let central = if (3..=4).contains(&f) { 1 } else { 0 };
            let near = if f == 2 || f == 5 { 1 } else { 0 };
            S(r * 3 + central * (r * 4) + near * r - 4, (r - 1) * 6)
        }
        KNIGHT => S(c * 9 - 25 + if r == 0 { -8 } else { 0 }, c * 7 - 22),
        BISHOP => {
            let diag = if f == r || f == 7 - r { 6 } else { 0 };
            S(c * 4 - 10 + diag + if r == 0 { -4 } else { 0 }, c * 4 - 12)
        }
        ROOK => S(
            if r == 6 { 18 } else { 0 } + if (2..=5).contains(&f) { 4 } else { 0 },
            if r == 6 { 10 } else { 0 },
        ),
        QUEEN => S(c * 2 - 6, c * 4 - 12),
        KING => {
            // Середина: король ховається в кутку за пішаками. Закінчення: іде в центр.
            let shelter = match f {
                0 | 1 | 6 | 7 => 20,
                2 => 10,
                5 => 0,
                _ => -10,
            };
            S(shelter - r * 18, c * 10 - 28)
        }
        _ => S(0, 0),
    }
}

fn tables() -> &'static Tables {
    TABLES.get_or_init(|| {
        let mut pst = [[S(0, 0); 64]; 12];
        for color in [WHITE, BLACK] {
            for kind in 0..6 {
                for sq in 0..64u8 {
                    pst[color * 6 + kind][sq as usize] =
                        VALUE[kind] + pst_white(kind, relative_sq(color, sq));
                }
            }
        }
        let mut king_danger = [0; 128];
        for (i, d) in king_danger.iter_mut().enumerate() {
            let x = i as i32;
            *d = (x * x / 5).min(600);
        }
        Tables { pst, king_danger }
    })
}

pub fn init() {
    let _ = tables();
}

/// Поля попереду пішака на його й сусідніх вертикалях.
fn forward_span(color: usize, sq: Square) -> Bitboard {
    let f = file_of(sq);
    let mut files = file_bb(f);
    if f > 0 {
        files |= file_bb(f - 1);
    }
    if f < 7 {
        files |= file_bb(f + 1);
    }
    let r = rank_of(sq);
    let ahead = if color == WHITE {
        if r == 7 {
            0
        } else {
            !0u64 << (8 * (r + 1))
        }
    } else if r == 0 {
        0
    } else {
        !0u64 >> (8 * (8 - r))
    };
    files & ahead
}

fn adjacent_files(f: u8) -> Bitboard {
    let mut b = 0;
    if f > 0 {
        b |= file_bb(f - 1);
    }
    if f < 7 {
        b |= file_bb(f + 1);
    }
    b
}

fn pawn_attacks_all(color: usize, pawns: Bitboard) -> Bitboard {
    if color == WHITE {
        ((pawns & !FILE_A) << 7) | ((pawns & !FILE_H) << 9)
    } else {
        ((pawns & !FILE_A) >> 9) | ((pawns & !FILE_H) >> 7)
    }
}

fn pawns_term(p: &Position, color: usize) -> S {
    let us = p.pieces(color, PAWN);
    let them = p.pieces(color ^ 1, PAWN);
    let mut s = S(0, 0);
    let mut b = us;
    let att = pawn_attacks_all(color, us);
    while b != 0 {
        let sq = pop_lsb(&mut b);
        let f = file_of(sq);
        let rr = rank_of(relative_sq(color, sq)) as usize;
        if count(us & file_bb(f)) > 1 {
            s += DOUBLED;
        }
        if us & adjacent_files(f) == 0 {
            s += ISOLATED;
        }
        if att & bit(sq) != 0 {
            s += SUPPORTED;
        }
        let side = if f > 0 { bit(sq - 1) } else { 0 } | if f < 7 { bit(sq + 1) } else { 0 };
        if us & side & rank_bb(rank_of(sq)) != 0 {
            s += PHALANX;
        }
        if them & forward_span(color, sq) == 0 && us & forward_span(color, sq) & file_bb(f) == 0 {
            s += PASSED[rr];
            let stop = if color == WHITE {
                sq + 8
            } else {
                sq.wrapping_sub(8)
            };
            if stop < 64 && p.occupied() & bit(stop) == 0 {
                s += PASSED_FREE * rr as i32;
            }
        }
    }
    s
}

/// Оцінка для білих (S), без змішування фаз.
fn evaluate_white(p: &Position) -> S {
    let t = tables();
    let mut score = S(0, 0);
    let occ = p.occupied();

    for color in [WHITE, BLACK] {
        let sign = if color == WHITE { 1 } else { -1 };
        let mut side = S(0, 0);
        let them = color ^ 1;
        let our_pawns = p.pieces(color, PAWN);
        let their_pawns = p.pieces(them, PAWN);
        let their_pawn_att = pawn_attacks_all(them, their_pawns);
        let our_pawn_att = pawn_attacks_all(color, our_pawns);
        let safe = !p.by_color[color] & !their_pawn_att;
        let their_king = p.king_sq(them);
        let zone = king_attacks(their_king) | bit(their_king);
        let mut attack_units = 0;
        let mut attackers = 0;

        for kind in 0..6 {
            let mut b = p.pieces(color, kind);
            while b != 0 {
                let sq = pop_lsb(&mut b);
                side += t.pst[color * 6 + kind][sq as usize];
                if kind == PAWN || kind == KING {
                    continue;
                }
                // Слони й тури бачать крізь свого ферзя й тури (батарея).
                let x_occ = match kind {
                    BISHOP => occ & !p.pieces(color, QUEEN),
                    ROOK => occ & !(p.pieces(color, QUEEN) | p.pieces(color, ROOK)),
                    _ => occ,
                };
                let a = attacks(kind, sq, x_occ);
                let mob = count(a & safe) as i32 - MOBILITY_BASE[kind];
                side += MOBILITY[kind] * mob;
                let za = count(a & zone) as i32;
                if za > 0 {
                    attackers += 1;
                    attack_units += KING_ATTACK_WEIGHT[kind] * za;
                }
                if kind == ROOK {
                    let file = file_bb(file_of(sq));
                    if file & our_pawns == 0 {
                        side += if file & their_pawns == 0 {
                            ROOK_OPEN
                        } else {
                            ROOK_SEMI
                        };
                    }
                }
                if (kind == KNIGHT || kind == BISHOP)
                    && a & (p.pieces(them, ROOK) | p.pieces(them, QUEEN)) != 0
                {
                    side += THREAT_BY_MINOR;
                }
            }
        }
        if count(p.pieces(color, BISHOP)) >= 2 {
            side += BISHOP_PAIR;
        }
        // Пішак б'є фігуру суперника.
        let hit = our_pawn_att & p.by_color[them] & !p.by_kind[PAWN] & !p.by_kind[KING];
        side += THREAT_BY_PAWN * count(hit) as i32;
        // Небезпека для короля суперника: лише коли нападників хоч двоє і в них є ферзь.
        if attackers >= 2 && p.pieces(color, QUEEN) != 0 {
            let idx = (attack_units as usize).min(127);
            side += S(t.king_danger[idx], t.king_danger[idx] / 4);
        }
        side += pawns_term(p, color);

        // Укриття власного короля: пішаки перед ним і відкриті вертикалі поруч.
        let ksq = p.king_sq(color);
        let kf = file_of(ksq);
        let near_files = adjacent_files(kf) | file_bb(kf);
        let shield_zone = forward_span(color, ksq) & near_files & {
            let r = rank_of(ksq);
            if color == WHITE {
                rank_bb((r + 1).min(7)) | rank_bb((r + 2).min(7))
            } else {
                rank_bb(r.saturating_sub(1)) | rank_bb(r.saturating_sub(2))
            }
        };
        side += KING_SHIELD * count(our_pawns & shield_zone).min(3) as i32;
        for f in kf.saturating_sub(1)..=(kf + 1).min(7) {
            if our_pawns & file_bb(f) == 0 {
                side += KING_OPEN_FILE;
            }
        }
        score += side * sign;
    }
    // Фігури без захисту під атакою: для кожного кольору окремо.
    for color in [WHITE, BLACK] {
        let sign = if color == WHITE { 1 } else { -1 };
        let them = color ^ 1;
        let their_att = all_attacks(p, them);
        let our_att = all_attacks(p, color);
        let weak = p.by_color[color] & !p.by_kind[KING] & !p.by_kind[PAWN] & their_att & !our_att;
        score -= HANGING * (count(weak) as i32 * sign);
    }
    score
}

fn all_attacks(p: &Position, color: usize) -> Bitboard {
    let occ = p.occupied();
    let mut a = pawn_attacks_all(color, p.pieces(color, PAWN)) | king_attacks(p.king_sq(color));
    for kind in [KNIGHT, BISHOP, ROOK, QUEEN] {
        let mut b = p.pieces(color, kind);
        while b != 0 {
            a |= attacks(kind, pop_lsb(&mut b), occ);
        }
    }
    a
}

/// Наскільки закінчення нічийне (з 64): без пішаків мала перевага не виграє.
fn scale(p: &Position, eg: i32) -> i32 {
    let strong = if eg > 0 { WHITE } else { BLACK };
    let weak = strong ^ 1;
    let strong_pawns = count(p.pieces(strong, PAWN));
    let mat = |c: usize| {
        (KNIGHT..=QUEEN)
            .map(|k| count(p.pieces(c, k)) as i32 * SEE_VALUE[k])
            .sum::<i32>()
    };
    let (ms, mw) = (mat(strong), mat(weak));
    if strong_pawns == 0 && ms - mw <= SEE_VALUE[BISHOP] {
        // Легка фігура переваги без пішаків – нічия, тура проти легкої – майже нічия.
        return if ms < SEE_VALUE[ROOK] { 0 } else { 16 };
    }
    // Різнокольорові слони без інших фігур – дуже нічийно.
    if count(p.pieces(WHITE, BISHOP)) == 1
        && count(p.pieces(BLACK, BISHOP)) == 1
        && p.by_kind[KNIGHT] | p.by_kind[ROOK] | p.by_kind[QUEEN] == 0
    {
        const LIGHT: Bitboard = 0x55AA_55AA_55AA_55AA;
        let wl = p.pieces(WHITE, BISHOP) & LIGHT != 0;
        let bl = p.pieces(BLACK, BISHOP) & LIGHT != 0;
        if wl != bl {
            return 24;
        }
    }
    64
}

pub fn evaluate(p: &Position) -> i32 {
    let s = evaluate_white(p);
    let phase = p.phase.min(MAX_PHASE);
    let eg = s.1 * scale(p, s.1) / 64;
    let v = (s.0 * phase + eg * (MAX_PHASE - phase)) / MAX_PHASE;
    let v = if p.side == WHITE { v } else { -v };
    v + TEMPO
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric() {
        crate::bitboard::init();
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ] {
            let p = Position::from_fen(fen).unwrap();
            let m = Position::from_fen(&mirror(fen)).unwrap();
            assert_eq!(evaluate(&p), evaluate(&m), "{fen}");
        }
    }

    fn mirror(fen: &str) -> String {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        let board: Vec<String> = parts[0]
            .split('/')
            .rev()
            .map(|r| {
                r.chars()
                    .map(|c| {
                        if c.is_ascii_uppercase() {
                            c.to_ascii_lowercase()
                        } else {
                            c.to_ascii_uppercase()
                        }
                    })
                    .collect()
            })
            .collect();
        let side = if parts[1] == "w" { "b" } else { "w" };
        let castle: String = if parts[2] == "-" {
            "-".into()
        } else {
            parts[2]
                .chars()
                .map(|c| {
                    if c.is_ascii_uppercase() {
                        c.to_ascii_lowercase()
                    } else {
                        c.to_ascii_uppercase()
                    }
                })
                .collect()
        };
        format!("{} {} {} - 0 1", board.join("/"), side, castle)
    }
}
