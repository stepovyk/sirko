//! Позиція: бітборди, хід, FEN, ключ Zobrist. Класичні шахи й шахи Фішера однаково:
//! рокіровка – «король бере свою туру», права зберігаються з полями тур.

use crate::bitboard::*;
use crate::types::*;
use std::sync::OnceLock;

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub struct Zobrist {
    pub piece: [[u64; 64]; 12],
    pub castle: [u64; 16],
    pub ep: [u64; 8],
    pub side: u64,
}

static ZOBRIST: OnceLock<Zobrist> = OnceLock::new();

pub fn zobrist() -> &'static Zobrist {
    ZOBRIST.get_or_init(|| {
        let mut s: u64 = 0x5EED_0F51_4C0F_FEE1;
        let mut next = || {
            // splitmix64
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut piece = [[0; 64]; 12];
        for p in piece.iter_mut() {
            for k in p.iter_mut() {
                *k = next();
            }
        }
        let mut castle = [0; 16];
        let base: [u64; 4] = [next(), next(), next(), next()];
        for (i, c) in castle.iter_mut().enumerate() {
            for (b, v) in base.iter().enumerate() {
                if i & (1 << b) != 0 {
                    *c ^= v;
                }
            }
        }
        let mut ep = [0; 8];
        for e in ep.iter_mut() {
            *e = next();
        }
        Zobrist {
            piece,
            castle,
            ep,
            side: next(),
        }
    })
}

// Права рокіровки: біт на кожну з чотирьох тур.
pub const WK: u8 = 1;
pub const WQ: u8 = 2;
pub const BK: u8 = 4;
pub const BQ: u8 = 8;

#[derive(Clone, Copy)]
pub struct Position {
    pub by_color: [Bitboard; 2],
    pub by_kind: [Bitboard; 6],
    pub board: [u8; 64],
    pub side: usize,
    pub castling: u8,
    pub ep: Square, // 64 – немає
    pub halfmove: u16,
    pub fullmove: u16,
    pub key: u64,
    pub pawn_key: u64,
    pub checkers: Bitboard,
    pub chess960: bool,
    /// Поле тури для кожного права (WK, WQ, BK, BQ по порядку бітів).
    pub castle_rook: [Square; 4],
    /// Маска прав, що лишаються, коли фігура йде з поля або на поле.
    pub castle_mask: [u8; 64],
    /// Набрана матеріальна фаза (для оцінки).
    pub phase: i32,
}

pub const PHASE_WEIGHT: [i32; 6] = [0, 1, 1, 2, 4, 0];
pub const MAX_PHASE: i32 = 24;

pub struct MoveList {
    pub moves: [Move; 256],
    pub len: usize,
}

impl MoveList {
    pub fn new() -> Self {
        MoveList {
            moves: [Move::NONE; 256],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        self.moves[self.len] = m;
        self.len += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = &Move> {
        self.moves[..self.len].iter()
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl Position {
    pub fn startpos() -> Position {
        Position::from_fen(START_FEN).expect("стартова позиція")
    }

    #[inline(always)]
    pub fn occupied(&self) -> Bitboard {
        self.by_color[WHITE] | self.by_color[BLACK]
    }

    #[inline(always)]
    pub fn pieces(&self, color: usize, kind: usize) -> Bitboard {
        self.by_color[color] & self.by_kind[kind]
    }

    #[inline(always)]
    pub fn king_sq(&self, color: usize) -> Square {
        lsb(self.pieces(color, KING))
    }

    #[inline(always)]
    pub fn piece_at(&self, sq: Square) -> u8 {
        self.board[sq as usize]
    }

    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.checkers != 0
    }

    fn put(&mut self, piece: u8, sq: Square) {
        let (c, k) = (color_of(piece), kind_of(piece));
        self.by_color[c] |= bit(sq);
        self.by_kind[k] |= bit(sq);
        self.board[sq as usize] = piece;
        self.key ^= zobrist().piece[piece as usize][sq as usize];
        if k == PAWN || k == KING {
            self.pawn_key ^= zobrist().piece[piece as usize][sq as usize];
        }
        self.phase += PHASE_WEIGHT[k];
    }

    fn remove(&mut self, sq: Square) -> u8 {
        let piece = self.board[sq as usize];
        let (c, k) = (color_of(piece), kind_of(piece));
        self.by_color[c] &= !bit(sq);
        self.by_kind[k] &= !bit(sq);
        self.board[sq as usize] = NO_PIECE;
        self.key ^= zobrist().piece[piece as usize][sq as usize];
        if k == PAWN || k == KING {
            self.pawn_key ^= zobrist().piece[piece as usize][sq as usize];
        }
        self.phase -= PHASE_WEIGHT[k];
        piece
    }

    /// Усі фігури обох кольорів, що б'ють поле sq при зайнятості occ.
    #[inline(always)]
    pub fn attackers_to(&self, sq: Square, occ: Bitboard) -> Bitboard {
        (pawn_attacks(BLACK, sq) & self.pieces(WHITE, PAWN))
            | (pawn_attacks(WHITE, sq) & self.pieces(BLACK, PAWN))
            | (knight_attacks(sq) & self.by_kind[KNIGHT])
            | (bishop_attacks(sq, occ) & (self.by_kind[BISHOP] | self.by_kind[QUEEN]))
            | (rook_attacks(sq, occ) & (self.by_kind[ROOK] | self.by_kind[QUEEN]))
            | (king_attacks(sq) & self.by_kind[KING])
    }

    #[inline(always)]
    pub fn attacked_by(&self, sq: Square, by: usize, occ: Bitboard) -> bool {
        self.attackers_to(sq, occ) & self.by_color[by] != 0
    }

    /// Свої фігури, зв'язані з власним королем.
    pub fn pinned(&self, color: usize) -> Bitboard {
        let ksq = self.king_sq(color);
        let them = color ^ 1;
        let snipers = (rook_attacks(ksq, 0) & (self.pieces(them, ROOK) | self.pieces(them, QUEEN)))
            | (bishop_attacks(ksq, 0) & (self.pieces(them, BISHOP) | self.pieces(them, QUEEN)));
        let occ = self.occupied();
        let mut pinned = 0;
        let mut s = snipers;
        while s != 0 {
            let sq = pop_lsb(&mut s);
            let blockers = between(ksq, sq) & occ;
            if blockers != 0 && !more_than_one(blockers) {
                pinned |= blockers & self.by_color[color];
            }
        }
        pinned
    }

    fn update_checkers(&mut self) {
        let ksq = self.king_sq(self.side);
        self.checkers = self.attackers_to(ksq, self.occupied()) & self.by_color[self.side ^ 1];
    }

    // ---------- FEN ----------

    pub fn from_fen(fen: &str) -> Result<Position, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(format!("неповний FEN: {fen}"));
        }
        let mut p = Position {
            by_color: [0; 2],
            by_kind: [0; 6],
            board: [NO_PIECE; 64],
            side: WHITE,
            castling: 0,
            ep: 64,
            halfmove: 0,
            fullmove: 1,
            key: 0,
            pawn_key: 0,
            checkers: 0,
            chess960: false,
            castle_rook: [7, 0, 63, 56],
            castle_mask: [15; 64],
            phase: 0,
        };
        let (mut file, mut rank) = (0i32, 7i32);
        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => file += ch as i32 - '0' as i32,
                _ => {
                    let kind = match ch.to_ascii_lowercase() {
                        'p' => PAWN,
                        'n' => KNIGHT,
                        'b' => BISHOP,
                        'r' => ROOK,
                        'q' => QUEEN,
                        'k' => KING,
                        _ => return Err(format!("невідома фігура «{ch}»")),
                    };
                    if !(0..8).contains(&file) || !(0..8).contains(&rank) {
                        return Err("FEN виходить за дошку".into());
                    }
                    let color = if ch.is_ascii_uppercase() {
                        WHITE
                    } else {
                        BLACK
                    };
                    p.put(piece_of(color, kind), make_sq(file as u8, rank as u8));
                    file += 1;
                }
            }
        }
        if count(p.pieces(WHITE, KING)) != 1 || count(p.pieces(BLACK, KING)) != 1 {
            return Err("має бути по одному королю".into());
        }
        p.side = match parts[1] {
            "w" => WHITE,
            "b" => BLACK,
            s => return Err(format!("невідомий бік «{s}»")),
        };
        for ch in parts[2].chars() {
            if ch == '-' {
                break;
            }
            let color = if ch.is_ascii_uppercase() {
                WHITE
            } else {
                BLACK
            };
            let ksq = p.king_sq(color);
            let back = rank_of(ksq);
            let rooks = p.pieces(color, ROOK) & rank_bb(back);
            let rook_sq = match ch.to_ascii_lowercase() {
                'k' => {
                    // Найдальша тура праворуч від короля (X-FEN).
                    let right = rooks & !((bit(ksq) << 1) - 1);
                    if right == 0 {
                        continue;
                    }
                    63 - right.leading_zeros() as Square
                }
                'q' => {
                    let left = rooks & (bit(ksq) - 1);
                    if left == 0 {
                        continue;
                    }
                    lsb(left)
                }
                c @ 'a'..='h' => {
                    p.chess960 = true;
                    make_sq(c as u8 - b'a', back)
                }
                _ => return Err(format!("невідоме право рокіровки «{ch}»")),
            };
            if p.piece_at(rook_sq) != piece_of(color, ROOK) {
                continue;
            }
            let kingside = file_of(rook_sq) > file_of(ksq);
            let right = match (color, kingside) {
                (WHITE, true) => WK,
                (WHITE, false) => WQ,
                (_, true) => BK,
                (_, false) => BQ,
            };
            p.castling |= right;
            p.castle_rook[right.trailing_zeros() as usize] = rook_sq;
            p.castle_mask[rook_sq as usize] &= !right;
            p.castle_mask[ksq as usize] &= !right;
        }
        if parts[3] != "-" {
            let sq = parse_sq(parts[3])
                .ok_or_else(|| format!("невідоме поле взяття на проході «{}»", parts[3]))?;
            // Поле враховуємо, лише якщо взяття справді можливе: так ключ не залежить від запису.
            let us = p.side;
            if pawn_attacks(us ^ 1, sq) & p.pieces(us, PAWN) != 0 {
                p.ep = sq;
            }
        }
        p.halfmove = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        p.fullmove = parts
            .get(5)
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
            .max(1);
        // Нетипове розміщення тур чи короля – це теж шахи Фішера.
        for (i, &rsq) in p.castle_rook.iter().enumerate() {
            if p.castling & (1 << i) != 0 {
                let color = i / 2;
                if p.king_sq(color) != relative_sq(color, 4)
                    || (rsq != relative_sq(color, 7) && rsq != relative_sq(color, 0))
                {
                    p.chess960 = true;
                }
            }
        }
        p.key ^= zobrist().castle[p.castling as usize];
        if p.side == BLACK {
            p.key ^= zobrist().side;
        }
        p.update_checkers();
        if p.ep != 64 {
            p.settle_ep();
        }
        if p.attacked_by(p.king_sq(p.side ^ 1), p.side, p.occupied()) {
            return Err("король того, хто не ходить, під шахом".into());
        }
        Ok(p)
    }

    pub fn fen(&self) -> String {
        let mut s = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let pc = self.piece_at(make_sq(file, rank));
                if pc == NO_PIECE {
                    empty += 1;
                    continue;
                }
                if empty > 0 {
                    s.push((b'0' + empty) as char);
                    empty = 0;
                }
                let ch = b"pnbrqk"[kind_of(pc)] as char;
                s.push(if color_of(pc) == WHITE {
                    ch.to_ascii_uppercase()
                } else {
                    ch
                });
            }
            if empty > 0 {
                s.push((b'0' + empty) as char);
            }
            if rank > 0 {
                s.push('/');
            }
        }
        s.push_str(if self.side == WHITE { " w " } else { " b " });
        if self.castling == 0 {
            s.push('-');
        } else {
            for (i, ch) in ['K', 'Q', 'k', 'q'].iter().enumerate() {
                if self.castling & (1 << i) != 0 {
                    if self.chess960 {
                        let f = (b'a' + file_of(self.castle_rook[i])) as char;
                        s.push(if i < 2 { f.to_ascii_uppercase() } else { f });
                    } else {
                        s.push(*ch);
                    }
                }
            }
        }
        if self.ep == 64 {
            s.push_str(" -");
        } else {
            s.push(' ');
            s.push_str(&sq_name(self.ep));
        }
        s.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        s
    }

    // ---------- Хід ----------

    /// Позиція після ходу (хід має бути легальним).
    pub fn make(&self, m: Move) -> Position {
        let mut p = *self;
        let z = zobrist();
        let us = self.side;
        let them = us ^ 1;
        let (from, to) = (m.from(), m.to());
        p.key ^= z.castle[p.castling as usize];
        if p.ep != 64 {
            p.key ^= z.ep[file_of(p.ep) as usize];
            p.ep = 64;
        }
        p.halfmove += 1;
        if us == BLACK {
            p.fullmove += 1;
        }
        if m.is_castle() {
            let kingside = to > from;
            let (kto, rto) = if kingside {
                (relative_sq(us, 6), relative_sq(us, 5))
            } else {
                (relative_sq(us, 2), relative_sq(us, 3))
            };
            let king = p.remove(from);
            let rook = p.remove(to);
            p.put(king, kto);
            p.put(rook, rto);
        } else {
            let piece = p.board[from as usize];
            if m.is_ep() {
                p.remove(if us == WHITE { to - 8 } else { to + 8 });
                p.halfmove = 0;
            } else if p.board[to as usize] != NO_PIECE {
                p.remove(to);
                p.halfmove = 0;
            }
            p.remove(from);
            if m.is_promo() {
                p.put(piece_of(us, m.promo_kind()), to);
            } else {
                p.put(piece, to);
            }
            if kind_of(piece) == PAWN {
                p.halfmove = 0;
                if m.flag() == DOUBLE_PUSH {
                    // Перевірка, чи взяття легальне, – після того, як черга перейде суперникові.
                    p.ep = if us == WHITE { from + 8 } else { from - 8 };
                }
            }
        }
        p.castling &= self.castle_mask[from as usize] & self.castle_mask[to as usize];
        p.key ^= z.castle[p.castling as usize];
        p.side = them;
        p.key ^= z.side;
        p.update_checkers();
        if p.ep != 64 {
            p.settle_ep();
        }
        p
    }

    /// Поле взяття на проході лишається, лише якщо взяття справді легальне: інакше та сама
    /// позиція мала б інший ключ, і повторення не впізнавалося б (як і в правилах FIDE).
    fn settle_ep(&mut self) {
        let ep = self.ep;
        let us = self.side;
        let mut takers = pawn_attacks(us ^ 1, ep) & self.pieces(us, PAWN);
        let mut legal = false;
        while takers != 0 {
            if self.is_legal(Move::new(pop_lsb(&mut takers), ep, EN_PASSANT)) {
                legal = true;
                break;
            }
        }
        if legal {
            self.key ^= zobrist().ep[file_of(ep) as usize];
        } else {
            self.ep = 64;
        }
    }

    /// Нульовий хід: черга переходить суперникові (для пошуку).
    pub fn make_null(&self) -> Position {
        let mut p = *self;
        let z = zobrist();
        if p.ep != 64 {
            p.key ^= z.ep[file_of(p.ep) as usize];
            p.ep = 64;
        }
        p.side ^= 1;
        p.key ^= z.side;
        p.halfmove += 1;
        p.checkers = 0;
        p
    }

    // ---------- Генерація ходів ----------

    /// Усі псевдолегальні ходи (tactical_only – лише взяття й промоції у ферзя).
    pub fn generate(&self, list: &mut MoveList, tactical_only: bool) {
        let us = self.side;
        let them = us ^ 1;
        let occ = self.occupied();
        let enemies = self.by_color[them];
        let targets = if tactical_only {
            enemies
        } else {
            !self.by_color[us]
        };

        // Під подвійним шахом ходить лише король.
        if !more_than_one(self.checkers) {
            self.pawn_moves(list, tactical_only);
            for kind in [KNIGHT, BISHOP, ROOK, QUEEN] {
                let mut b = self.pieces(us, kind);
                while b != 0 {
                    let from = pop_lsb(&mut b);
                    let mut a = attacks(kind, from, occ) & targets;
                    while a != 0 {
                        let to = pop_lsb(&mut a);
                        list.push(Move::new(
                            from,
                            to,
                            if enemies & bit(to) != 0 {
                                CAPTURE
                            } else {
                                QUIET
                            },
                        ));
                    }
                }
            }
        }
        let ksq = self.king_sq(us);
        let mut a = king_attacks(ksq) & targets;
        while a != 0 {
            let to = pop_lsb(&mut a);
            list.push(Move::new(
                ksq,
                to,
                if enemies & bit(to) != 0 {
                    CAPTURE
                } else {
                    QUIET
                },
            ));
        }
        if !tactical_only && self.checkers == 0 {
            for i in [us * 2, us * 2 + 1] {
                if self.castling & (1 << i) != 0 {
                    let rsq = self.castle_rook[i];
                    if self.castle_path_clear(us, ksq, rsq) {
                        list.push(Move::new(ksq, rsq, CASTLE));
                    }
                }
            }
        }
    }

    /// Поля між королем і турою та до їхніх цільових полів вільні (крім самих короля й тури).
    fn castle_path_clear(&self, us: usize, ksq: Square, rsq: Square) -> bool {
        let kingside = rsq > ksq;
        let (kto, rto) = if kingside {
            (relative_sq(us, 6), relative_sq(us, 5))
        } else {
            (relative_sq(us, 2), relative_sq(us, 3))
        };
        let occ = self.occupied() & !bit(ksq) & !bit(rsq);
        let path =
            (between(ksq, kto) | bit(kto) | between(rsq, rto) | bit(rto)) & !bit(ksq) & !bit(rsq);
        path & occ == 0
    }

    fn pawn_moves(&self, list: &mut MoveList, tactical_only: bool) {
        let us = self.side;
        let them = us ^ 1;
        let pawns = self.pieces(us, PAWN);
        let occ = self.occupied();
        let enemies = self.by_color[them];
        let (up, promo_rank, double_rank): (i8, Bitboard, Bitboard) = if us == WHITE {
            (8, RANK_8, RANK_4)
        } else {
            (-8, RANK_1, RANK_5)
        };
        let shift = |b: Bitboard, d: i8| if d > 0 { b << d } else { b >> (-d) };

        let single = shift(pawns, up) & !occ;
        let mut b = single & promo_rank;
        while b != 0 {
            let to = pop_lsb(&mut b);
            let from = (to as i8 - up) as Square;
            push_promos(list, from, to, false, tactical_only);
        }
        if !tactical_only {
            let mut b = single & !promo_rank;
            while b != 0 {
                let to = pop_lsb(&mut b);
                list.push(Move::new((to as i8 - up) as Square, to, QUIET));
            }
            let mut b = shift(single, up) & !occ & double_rank;
            while b != 0 {
                let to = pop_lsb(&mut b);
                list.push(Move::new((to as i8 - 2 * up) as Square, to, DOUBLE_PUSH));
            }
        }
        let mut b = pawns;
        while b != 0 {
            let from = pop_lsb(&mut b);
            let mut a = pawn_attacks(us, from) & enemies;
            while a != 0 {
                let to = pop_lsb(&mut a);
                if bit(to) & promo_rank != 0 {
                    push_promos(list, from, to, true, tactical_only);
                } else {
                    list.push(Move::new(from, to, CAPTURE));
                }
            }
            if self.ep != 64 && pawn_attacks(us, from) & bit(self.ep) != 0 {
                list.push(Move::new(from, self.ep, EN_PASSANT));
            }
        }
    }

    /// Чи легальний псевдолегальний хід (свій король не лишається під шахом).
    pub fn is_legal(&self, m: Move) -> bool {
        let us = self.side;
        let them = us ^ 1;
        let (from, to) = (m.from(), m.to());
        let ksq = self.king_sq(us);
        let occ = self.occupied();
        if m.is_castle() {
            let kingside = to > from;
            let kto = if kingside {
                relative_sq(us, 6)
            } else {
                relative_sq(us, 2)
            };
            // Король не проходить через биті поля (і не стає під шах після рокіровки).
            let mut path = between(from, kto) | bit(kto);
            let occ_after = occ & !bit(from) & !bit(to);
            while path != 0 {
                let sq = pop_lsb(&mut path);
                if self.attacked_by(sq, them, occ_after) {
                    return false;
                }
            }
            // Тура, що йде з місця, могла закривати лінію на поле, куди стає король.
            let rto = if kingside {
                relative_sq(us, 5)
            } else {
                relative_sq(us, 3)
            };
            return !self.attacked_by(kto, them, occ_after | bit(kto) | bit(rto));
        }
        if from == ksq {
            return !self.attacked_by(to, them, occ & !bit(from));
        }
        if m.is_ep() {
            let cap = if us == WHITE { to - 8 } else { to + 8 };
            let occ2 = (occ & !bit(from) & !bit(cap)) | bit(to);
            let rooks = self.pieces(them, ROOK) | self.pieces(them, QUEEN);
            let bishops = self.pieces(them, BISHOP) | self.pieces(them, QUEEN);
            let checkers_after = (rook_attacks(ksq, occ2) & rooks)
                | (bishop_attacks(ksq, occ2) & bishops)
                | (self.checkers & !bit(cap) & !(rooks | bishops));
            return checkers_after == 0;
        }
        if self.checkers != 0 {
            if more_than_one(self.checkers) {
                return false;
            }
            let checker = lsb(self.checkers);
            if bit(to) & (between(ksq, checker) | self.checkers) == 0 {
                return false;
            }
        }
        // Зв'язана фігура ходить лише по лінії зв'язки.
        let pinned = self.pinned(us);
        pinned & bit(from) == 0 || line(ksq, from) & bit(to) != 0
    }

    /// Легальні ходи.
    pub fn legal_moves(&self) -> MoveList {
        let mut all = MoveList::new();
        self.generate(&mut all, false);
        let mut out = MoveList::new();
        for &m in all.iter() {
            if self.is_legal(m) {
                out.push(m);
            }
        }
        out
    }

    // ---------- Запис ходів UCI ----------

    pub fn move_to_uci(&self, m: Move) -> String {
        let (from, mut to) = (m.from(), m.to());
        if m.is_castle() && !self.chess960 {
            to = if to > from {
                relative_sq(self.side, 6)
            } else {
                relative_sq(self.side, 2)
            };
        }
        let mut s = format!("{}{}", sq_name(from), sq_name(to));
        if m.is_promo() {
            s.push(b"nbrq"[m.promo_kind() - KNIGHT] as char);
        }
        s
    }

    pub fn parse_uci(&self, text: &str) -> Option<Move> {
        let legal = self.legal_moves();
        let found = legal.iter().copied().find(|&m| {
            if self.move_to_uci(m) == text {
                return true;
            }
            // Рокіровку в класиці приймаємо й записом «король бере туру», і навпаки.
            if m.is_castle() {
                let kto = if m.to() > m.from() {
                    relative_sq(self.side, 6)
                } else {
                    relative_sq(self.side, 2)
                };
                let alt1 = format!("{}{}", sq_name(m.from()), sq_name(m.to()));
                let alt2 = format!("{}{}", sq_name(m.from()), sq_name(kto));
                return text == alt1 || text == alt2;
            }
            false
        });
        found
    }

    /// Чи є на дошці матеріал для мату хоч у когось.
    pub fn insufficient_material(&self) -> bool {
        if self.by_kind[PAWN] | self.by_kind[ROOK] | self.by_kind[QUEEN] != 0 {
            return false;
        }
        let minors = self.by_kind[KNIGHT] | self.by_kind[BISHOP];
        if count(minors) <= 1 {
            return true;
        }
        // Лише слони на полях одного кольору.
        const LIGHT: Bitboard = 0x55AA_55AA_55AA_55AA;
        self.by_kind[KNIGHT] == 0
            && (self.by_kind[BISHOP] & LIGHT == 0 || self.by_kind[BISHOP] & !LIGHT == 0)
    }

    /// Кількість ходів без руху пішаків і взять досягла 100 (і це не мат).
    pub fn fifty_move_draw(&self) -> bool {
        self.halfmove >= 100 && (!self.in_check() || self.legal_moves().len > 0)
    }

    pub fn non_pawn_material(&self, color: usize) -> bool {
        self.by_color[color] & !(self.by_kind[PAWN] | self.by_kind[KING]) != 0
    }
}

fn push_promos(list: &mut MoveList, from: Square, to: Square, capture: bool, tactical_only: bool) {
    let base = PROMO | if capture { CAPTURE } else { 0 };
    list.push(Move::new(from, to, base | 3)); // ферзь
    if !tactical_only {
        list.push(Move::new(from, to, base | 2));
        list.push(Move::new(from, to, base | 1));
        list.push(Move::new(from, to, base));
    }
}

pub fn perft(p: &Position, depth: u32) -> u64 {
    let moves = p.legal_moves();
    if depth == 1 {
        return moves.len as u64;
    }
    moves.iter().map(|&m| perft(&p.make(m), depth - 1)).sum()
}
