//! Самогра для навчання нейромережі: Сірко грає сам із собою з обмеженням вузлів на хід і
//! записує спокійні позиції з оцінкою пошуку й результатом партії.
//!
//! Запис – 32 байти: зайнятість (8), фігури по 4 біти в порядку бітів зайнятості (16),
//! оцінка для того, хто ходить (i16), результат для білих (0 – поразка, 1 – нічия, 2 – перемога),
//! хто ходить, номер ходу (u16), два байти запасу.

use crate::bitboard::*;
use crate::position::Position;
use crate::search::{think, Limits, Shared, MATE_BOUND};
use crate::types::*;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const RECORD: usize = 32;

#[derive(Clone, Copy)]
pub struct Sample {
    pub occ: u64,
    pub pieces: [u8; 16],
    pub score: i16,
    pub result: u8,
    pub stm: u8,
}

impl Sample {
    pub fn from_position(p: &Position, score: i32) -> Sample {
        let occ = p.occupied();
        let mut pieces = [0u8; 16];
        let mut b = occ;
        let mut i = 0;
        while b != 0 {
            let sq = pop_lsb(&mut b);
            let pc = p.piece_at(sq);
            pieces[i / 2] |= pc << (4 * (i % 2));
            i += 1;
        }
        Sample {
            occ,
            pieces,
            score: score.clamp(-32000, 32000) as i16,
            result: 1,
            stm: p.side as u8,
        }
    }

    pub fn to_bytes(self, fullmove: u16) -> [u8; RECORD] {
        let mut out = [0u8; RECORD];
        out[..8].copy_from_slice(&self.occ.to_le_bytes());
        out[8..24].copy_from_slice(&self.pieces);
        out[24..26].copy_from_slice(&self.score.to_le_bytes());
        out[26] = self.result;
        out[27] = self.stm;
        out[28..30].copy_from_slice(&fullmove.to_le_bytes());
        out
    }

    pub fn from_bytes(b: &[u8]) -> Sample {
        let mut pieces = [0u8; 16];
        pieces.copy_from_slice(&b[8..24]);
        Sample {
            occ: u64::from_le_bytes(b[..8].try_into().expect("8 байт")),
            pieces,
            score: i16::from_le_bytes([b[24], b[25]]),
            result: b[26],
            stm: b[27],
        }
    }

    /// (поле, фігура) для кожної фігури на дошці.
    pub fn squares(&self) -> impl Iterator<Item = (Square, u8)> + '_ {
        let mut b = self.occ;
        let mut i = 0;
        std::iter::from_fn(move || {
            if b == 0 {
                return None;
            }
            let sq = pop_lsb(&mut b);
            let pc = (self.pieces[i / 2] >> (4 * (i % 2))) & 15;
            i += 1;
            Some((sq, pc))
        })
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

pub struct Options {
    pub games: u64,
    pub nodes: u64,
    pub threads: usize,
    pub out: String,
    pub random_plies: usize,
    /// Нових партій після стількох секунд не починати (0 – без обмеження).
    pub seconds: u64,
}

/// Одна партія: записи з результатом.
fn play_game(
    rng: &mut Rng,
    shared: &Arc<Shared>,
    nodes: u64,
    random_plies: usize,
) -> Vec<[u8; RECORD]> {
    let mut pos = Position::startpos();
    let mut keys = vec![pos.key];
    // Випадковий початок: щоразу інша партія.
    for _ in 0..random_plies {
        let moves = pos.legal_moves();
        if moves.len == 0 {
            return Vec::new();
        }
        pos = pos.make(moves.moves[(rng.next() % moves.len as u64) as usize]);
        keys.push(pos.key);
    }
    if pos.legal_moves().len == 0 {
        return Vec::new();
    }
    shared.tt.clear();
    shared.clear_heuristics();
    let mut samples: Vec<(Sample, u16)> = Vec::new();
    let mut win_streak = 0;
    let mut draw_streak = 0;
    let result: u8 = loop {
        let moves = pos.legal_moves();
        if moves.len == 0 {
            break if pos.in_check() {
                if pos.side == WHITE {
                    0
                } else {
                    2
                }
            } else {
                1
            };
        }
        if pos.halfmove >= 100
            || pos.insufficient_material()
            || keys.iter().filter(|&&k| k == pos.key).count() >= 3
        {
            break 1;
        }
        let limits = Limits {
            nodes: Some(nodes),
            depth: Some(64),
            ..Limits::default()
        };
        let r = think(&pos, &keys, &limits, Arc::clone(shared), 1, 0, false);
        if r.best.is_none() {
            break 1;
        }
        let score = r.score;
        // Спокійні позиції: не під шахом, найкращий хід тихий, без мату на дошці.
        if !pos.in_check() && !r.best.is_tactical() && score.abs() < MATE_BOUND {
            samples.push((Sample::from_position(&pos, score), pos.fullmove));
        }
        // Присуд: явна перевага кілька ходів поспіль – перемога; довга рівна гра – нічия.
        let white_score = if pos.side == WHITE { score } else { -score };
        if white_score.abs() >= 2000 {
            win_streak += 1;
            if win_streak >= 6 {
                break if white_score > 0 { 2 } else { 0 };
            }
        } else {
            win_streak = 0;
        }
        if pos.fullmove >= 40 && white_score.abs() <= 8 {
            draw_streak += 1;
            if draw_streak >= 12 {
                break 1;
            }
        } else {
            draw_streak = 0;
        }
        pos = pos.make(r.best);
        keys.push(pos.key);
    };
    samples
        .into_iter()
        .map(|(mut s, fm)| {
            s.result = result;
            s.to_bytes(fm)
        })
        .collect()
}

pub fn run(opt: Options) {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&opt.out)
        .expect("файл для запису");
    let out = Arc::new(Mutex::new(BufWriter::new(file)));
    let games = Arc::new(AtomicU64::new(0));
    let positions = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let seed =
        start.elapsed().as_nanos() as u64 ^ std::process::id() as u64 ^ 0x9E37_79B9_7F4A_7C15;
    std::thread::scope(|scope| {
        for t in 0..opt.threads.max(1) {
            let out = Arc::clone(&out);
            let games = Arc::clone(&games);
            let positions = Arc::clone(&positions);
            let opt = &opt;
            scope.spawn(move || {
                let mut rng =
                    Rng(seed.wrapping_add((t as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03)) | 1);
                let shared = Arc::new(Shared::new(16, 1).with_net(crate::nnue::embedded()));
                let mut last = Instant::now();
                loop {
                    if opt.seconds > 0 && start.elapsed().as_secs() >= opt.seconds {
                        break;
                    }
                    if games.fetch_add(1, Ordering::Relaxed) >= opt.games {
                        break;
                    }
                    let records = play_game(&mut rng, &shared, opt.nodes, opt.random_plies);
                    let n = records.len() as u64;
                    {
                        let mut w = out.lock().expect("запис");
                        for r in &records {
                            w.write_all(r).expect("запис");
                        }
                    }
                    let total = positions.fetch_add(n, Ordering::Relaxed) + n;
                    let done = games.load(Ordering::Relaxed).min(opt.games);
                    if t == 0 && last.elapsed().as_secs() >= 60 {
                        last = Instant::now();
                        let s = start.elapsed().as_secs_f64();
                        eprintln!(
                            "партій {done}, позицій {total}, {:.0} позицій/с",
                            total as f64 / s
                        );
                    }
                }
            });
        }
    });
    out.lock().expect("запис").flush().expect("запис");
    eprintln!("готово: позицій {}", positions.load(Ordering::Relaxed));
}
