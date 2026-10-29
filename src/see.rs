//! Обмін на полі (SEE): чи не програє взяття чи хід матеріал, якщо обидва боки б'ють
//! по черзі найдешевшою фігурою.

use crate::bitboard::*;
use crate::eval::SEE_VALUE;
use crate::position::Position;
use crate::types::*;

/// Чи дає хід щонайменше threshold матеріалу після всіх розумних розмінів.
pub fn see_ge(p: &Position, m: Move, threshold: i32) -> bool {
    if m.is_castle() || m.is_ep() || m.is_promo() {
        return 0 >= threshold;
    }
    let (from, to) = (m.from(), m.to());
    let captured = p.piece_at(to);
    let mut swap = if captured == NO_PIECE {
        0
    } else {
        SEE_VALUE[kind_of(captured)]
    } - threshold;
    if swap < 0 {
        return false;
    }
    swap = SEE_VALUE[kind_of(p.piece_at(from))] - swap;
    if swap <= 0 {
        return true;
    }
    let mut occ = p.occupied() ^ bit(from) ^ bit(to);
    let mut attackers = p.attackers_to(to, occ) & occ;
    let mut side = p.side ^ 1;
    let mut result = true;
    let diag = p.by_kind[BISHOP] | p.by_kind[QUEEN];
    let orth = p.by_kind[ROOK] | p.by_kind[QUEEN];
    loop {
        let mine = attackers & p.by_color[side];
        if mine == 0 {
            break;
        }
        result = !result;
        // Найдешевша фігура, що б'є.
        let mut kind = PAWN;
        let mut b = 0;
        for k in PAWN..=KING {
            b = mine & p.by_kind[k];
            if b != 0 {
                kind = k;
                break;
            }
        }
        if kind == KING {
            // Король б'є лише тоді, коли поле більше ніхто не захищає.
            return if attackers & p.by_color[side ^ 1] != 0 {
                !result
            } else {
                result
            };
        }
        swap = SEE_VALUE[kind] - swap;
        if swap < result as i32 {
            break;
        }
        occ ^= bit(lsb(b));
        if kind == PAWN || kind == BISHOP || kind == QUEEN {
            attackers |= bishop_attacks(to, occ) & diag;
        }
        if kind == ROOK || kind == QUEEN {
            attackers |= rook_attacks(to, occ) & orth;
        }
        attackers &= occ;
        side ^= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(p: &Position, uci: &str) -> Move {
        p.parse_uci(uci).unwrap()
    }

    #[test]
    fn exchanges() {
        crate::bitboard::init();
        let p = Position::from_fen("1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1").unwrap();
        assert!(see_ge(&p, mv(&p, "e1e5"), 0)); // пішак без захисту
        let p =
            Position::from_fen("1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1").unwrap();
        assert!(!see_ge(&p, mv(&p, "d3e5"), 0)); // кінь за пішака, захищеного двічі
        let p = Position::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
        assert!(see_ge(&p, mv(&p, "e4d5"), 100));
        let p = Position::from_fen("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
        assert!(see_ge(&p, mv(&p, "e4d5"), 0));
        assert!(!see_ge(&p, mv(&p, "e4d5"), 1));
    }
}
