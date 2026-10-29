//! Базові типи: колір, фігура, поле, хід.

pub type Bitboard = u64;
pub type Square = u8;

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;

pub const PAWN: usize = 0;
pub const KNIGHT: usize = 1;
pub const BISHOP: usize = 2;
pub const ROOK: usize = 3;
pub const QUEEN: usize = 4;
pub const KING: usize = 5;
pub const NO_PIECE: u8 = 12;

/// Фігура на полі: колір * 6 + вид (0..12), NO_PIECE – порожньо.
#[inline(always)]
pub fn piece_of(color: usize, kind: usize) -> u8 {
    (color * 6 + kind) as u8
}

#[inline(always)]
pub fn kind_of(piece: u8) -> usize {
    (piece % 6) as usize
}

#[inline(always)]
pub fn color_of(piece: u8) -> usize {
    (piece / 6) as usize
}

#[inline(always)]
pub fn file_of(sq: Square) -> u8 {
    sq & 7
}

#[inline(always)]
pub fn rank_of(sq: Square) -> u8 {
    sq >> 3
}

#[inline(always)]
pub fn make_sq(file: u8, rank: u8) -> Square {
    rank * 8 + file
}

/// Поле з погляду кольору: для чорних дошка дзеркалиться по горизонталі.
#[inline(always)]
pub fn relative_sq(color: usize, sq: Square) -> Square {
    if color == WHITE {
        sq
    } else {
        sq ^ 56
    }
}

pub fn sq_name(sq: Square) -> String {
    let f = (b'a' + file_of(sq)) as char;
    let r = (b'1' + rank_of(sq)) as char;
    format!("{f}{r}")
}

pub fn parse_sq(s: &str) -> Option<Square> {
    let b = s.as_bytes();
    if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
        return None;
    }
    Some(make_sq(b[0] - b'a', b[1] - b'1'))
}

/// Хід у 16 бітах: звідки (6), куди (6), вид (4).
/// Рокіровка кодується як «король бере свою туру»: так однаково для класики й шахів Фішера.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub struct Move(pub u16);

pub const QUIET: u16 = 0;
pub const DOUBLE_PUSH: u16 = 1;
pub const CASTLE: u16 = 2;
pub const CAPTURE: u16 = 4;
pub const EN_PASSANT: u16 = 5;
pub const PROMO: u16 = 8; // 8..11 – тиха промоція у коня, слона, туру, ферзя; 12..15 – з взяттям

impl Move {
    pub const NONE: Move = Move(0);

    #[inline(always)]
    pub fn new(from: Square, to: Square, flag: u16) -> Move {
        Move(from as u16 | ((to as u16) << 6) | (flag << 12))
    }

    #[inline(always)]
    pub fn from(self) -> Square {
        (self.0 & 63) as Square
    }

    #[inline(always)]
    pub fn to(self) -> Square {
        ((self.0 >> 6) & 63) as Square
    }

    #[inline(always)]
    pub fn flag(self) -> u16 {
        self.0 >> 12
    }

    #[inline(always)]
    pub fn is_capture(self) -> bool {
        self.flag() & CAPTURE != 0 && self.flag() != CASTLE
    }

    #[inline(always)]
    pub fn is_promo(self) -> bool {
        self.flag() & PROMO != 0
    }

    #[inline(always)]
    pub fn is_castle(self) -> bool {
        self.flag() == CASTLE
    }

    #[inline(always)]
    pub fn is_ep(self) -> bool {
        self.flag() == EN_PASSANT
    }

    /// Взяття або промоція: такі ходи не «тихі» для пошуку.
    #[inline(always)]
    pub fn is_tactical(self) -> bool {
        self.is_capture() || self.is_promo()
    }

    /// Вид фігури, на яку перетворюється пішак (KNIGHT..QUEEN).
    #[inline(always)]
    pub fn promo_kind(self) -> usize {
        (self.flag() & 3) as usize + KNIGHT
    }

    #[inline(always)]
    pub fn is_none(self) -> bool {
        self.0 == 0
    }
}
