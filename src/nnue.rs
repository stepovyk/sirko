//! Власна нейромережа оцінки: (768 → H) для кожного боку → 1.
//!
//! Вхід – 768 ознак «колір × фігура × поле» з погляду кожного боку (для чорних дошка
//! віддзеркалюється, а кольори міняються), прихований шар – SCReLU (обрізаний квадрат),
//! вихід – оцінка в сантипішаках. Цілі числа: шар 1 і зсув – ×QA, вихід – ×QB.
//!
//! Вихід – окремий для кожного «кошика» за кількістю фігур на дошці (дебют, середина, закінчення).
//!
//! Файл: «SIRKONN2», H і кількість кошиків B (u32 LE), далі i16 LE: W1[768][H], B1[H], W2[B][2H],
//! і B2[B] (i32 LE). Старий формат «SIRKONN1» (один кошик, без B) теж читається.

use crate::bitboard::pop_lsb;
use crate::position::Position;
use crate::types::*;
use std::sync::OnceLock;

pub const INPUTS: usize = 768;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const SCALE: i32 = 400;
const MAGIC_V1: &[u8; 8] = b"SIRKONN1";
const MAGIC: &[u8; 8] = b"SIRKONN2";
pub const BUCKETS: usize = 8;

pub struct Network {
    pub hidden: usize,
    pub buckets: usize,
    pub w1: Vec<i16>, // [INPUTS][hidden]
    pub b1: Vec<i16>,
    pub w2: Vec<i16>, // [buckets][2 * hidden]: спершу той, хто ходить, потім суперник
    pub b2: Vec<i32>, // [buckets]
}

/// Кошик виходу за кількістю фігур на дошці (2..=32).
#[inline(always)]
pub fn bucket(pieces: u32, buckets: usize) -> usize {
    (pieces.saturating_sub(2) as usize * buckets / 31).min(buckets - 1)
}

/// Індекс ознаки фігури piece на полі sq з погляду кольору persp.
#[inline(always)]
pub fn feature(persp: usize, piece: u8, sq: Square) -> usize {
    let (c, k) = (color_of(piece), kind_of(piece));
    if persp == WHITE {
        c * 384 + k * 64 + sq as usize
    } else {
        (c ^ 1) * 384 + k * 64 + (sq ^ 56) as usize
    }
}

impl Network {
    pub fn from_bytes(data: &[u8]) -> Result<Network, String> {
        let v1 = data.len() >= 12 && &data[..8] == MAGIC_V1;
        if !(v1 || data.len() >= 16 && &data[..8] == MAGIC) {
            return Err("не файл мережі Сірка".into());
        }
        let word = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) as usize;
        let hidden = word(8);
        let (buckets, mut at) = if v1 { (1, 12) } else { (word(12), 16) };
        if hidden == 0 || hidden > 4096 || buckets == 0 || buckets > 32 {
            return Err(format!(
                "неправильні розміри мережі: {hidden} нейронів, {buckets} кошиків"
            ));
        }
        let need = at + 2 * (INPUTS * hidden + hidden + buckets * 2 * hidden) + 4 * buckets;
        if data.len() != need {
            return Err(format!(
                "неправильний розмір мережі: {} байт, очікую {need}",
                data.len()
            ));
        }
        let mut take = |n: usize| -> Vec<i16> {
            let v = data[at..at + 2 * n]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
            at += 2 * n;
            v
        };
        let w1 = take(INPUTS * hidden);
        let b1 = take(hidden);
        let w2 = take(buckets * 2 * hidden);
        let start = need - 4 * buckets;
        let b2 = data[start..]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Network {
            hidden,
            buckets,
            w1,
            b1,
            w2,
            b2,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(12 + 2 * (self.w1.len() + self.b1.len() + self.w2.len()) + 4);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(self.hidden as u32).to_le_bytes());
        out.extend_from_slice(&(self.buckets as u32).to_le_bytes());
        for v in self.w1.iter().chain(&self.b1).chain(&self.w2) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for b in &self.b2 {
            out.extend_from_slice(&b.to_le_bytes());
        }
        out
    }

    /// Накопичувачі обох боків з нуля.
    pub fn accumulate(&self, p: &Position, acc: &mut [Vec<i16>; 2]) {
        for (persp, a) in acc.iter_mut().enumerate() {
            a.clear();
            a.extend_from_slice(&self.b1);
            let mut occ = p.occupied();
            while occ != 0 {
                let sq = pop_lsb(&mut occ);
                let f = feature(persp, p.piece_at(sq), sq);
                let row = &self.w1[f * self.hidden..(f + 1) * self.hidden];
                for (x, w) in a.iter_mut().zip(row) {
                    *x += *w;
                }
            }
        }
    }

    /// Накопичувач після ходу: копія попереднього, мінус зняті ознаки, плюс додані.
    #[inline(always)]
    pub fn update(&self, src: &[i16], dst: &mut [i16], adds: &[usize], subs: &[usize]) {
        let h = self.hidden;
        dst.copy_from_slice(src);
        for &f in adds {
            for (x, w) in dst.iter_mut().zip(&self.w1[f * h..(f + 1) * h]) {
                *x += *w;
            }
        }
        for &f in subs {
            for (x, w) in dst.iter_mut().zip(&self.w1[f * h..(f + 1) * h]) {
                *x -= *w;
            }
        }
    }

    /// Вихід мережі з готових накопичувачів (us – той, хто ходить).
    pub fn output(&self, us: &[i16], them: &[i16], pieces: u32) -> i32 {
        let h = self.hidden;
        let b = bucket(pieces, self.buckets);
        let w2 = &self.w2[b * 2 * h..(b + 1) * 2 * h];
        let mut sum: i64 = 0;
        for (side, acc) in [us, them].into_iter().enumerate() {
            let w = &w2[side * h..(side + 1) * h];
            // Блоками по 16: кожен добуток не більший за 255·255·127, тож блок вміщається в i32,
            // і компілятор рахує його векторно.
            let mut blocks = acc.chunks_exact(16).zip(w.chunks_exact(16));
            for (cv, cw) in &mut blocks {
                let mut lane: i32 = 0;
                for k in 0..16 {
                    let c = (cv[k] as i32).clamp(0, QA);
                    lane += c * cw[k] as i32 * c;
                }
                sum += lane as i64;
            }
            let tail = h - h % 16;
            for (&v, &wt) in acc[tail..].iter().zip(&w[tail..]) {
                let c = (v as i32).clamp(0, QA);
                sum += (c * wt as i32) as i64 * c as i64;
            }
        }
        let out = sum / QA as i64 + self.b2[b] as i64;
        (out * SCALE as i64 / (QA as i64 * QB as i64)) as i32
    }

    pub fn evaluate(&self, p: &Position) -> i32 {
        let mut acc = [
            Vec::with_capacity(self.hidden),
            Vec::with_capacity(self.hidden),
        ];
        self.accumulate(p, &mut acc);
        self.output(&acc[p.side], &acc[p.side ^ 1], p.occupied().count_ones())
    }
}

static NET: OnceLock<Option<Network>> = OnceLock::new();

/// Мережа, вбудована при збиранні (якщо є), або None – тоді оцінка ручна.
pub fn embedded() -> Option<&'static Network> {
    NET.get_or_init(|| {
        let bytes: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sirko.nnue"));
        if bytes.is_empty() {
            None
        } else {
            Some(Network::from_bytes(bytes).expect("вбудована мережа пошкоджена"))
        }
    })
    .as_ref()
}
