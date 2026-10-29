//! Пошук: ітеративне поглиблення з вікнами прагнення, PVS, таблиця переставлень, нульовий хід,
//! скорочення пізніх ходів, відсічення марних ходів, тихий пошук. Кілька потоків шукають
//! одночасно зі спільною таблицею (Lazy SMP); головний потік стежить за часом і пише UCI.

use crate::eval::{evaluate, SEE_VALUE};
use crate::nnue::{feature, Network};
use crate::position::{MoveList, Position};
use crate::see::see_ge;
use crate::tt::*;
use crate::types::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

pub const INF: i32 = 32001;
pub const MATE: i32 = 32000;
pub const MATE_BOUND: i32 = MATE - 256;
pub const MAX_PLY: usize = 128;

#[derive(Clone, Default)]
pub struct Limits {
    pub time: [Option<u64>; 2], // мс на годиннику
    pub inc: [u64; 2],
    pub movestogo: Option<u64>,
    pub movetime: Option<u64>,
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    pub mate: Option<i32>,
    pub infinite: bool,
    pub ponder: bool,
}

pub struct Shared {
    pub tt: Table,
    pub stop: AtomicBool,
    /// Поки true, пошук іде на час суперника (ponder): межі часу не діють, доки не буде ponderhit.
    pub pondering: AtomicBool,
    pub nodes: Vec<AtomicU64>,
    /// Відлік часу на хід (мкс від base): ponderhit переносить його на мить, коли пішов наш годинник.
    base: Instant,
    start_us: AtomicU64,
    /// Нейромережа оцінки (None – ручна оцінка).
    pub net: Option<&'static Network>,
}

impl Shared {
    pub fn new(hash_mb: usize, threads: usize) -> Shared {
        Shared {
            tt: Table::new(hash_mb),
            stop: AtomicBool::new(false),
            pondering: AtomicBool::new(false),
            nodes: (0..threads.max(1)).map(|_| AtomicU64::new(0)).collect(),
            base: Instant::now(),
            start_us: AtomicU64::new(0),
            net: None,
        }
    }

    pub fn with_net(mut self, net: Option<&'static Network>) -> Shared {
        self.net = net;
        self
    }

    pub fn restart_clock(&self) {
        self.start_us
            .store(self.base.elapsed().as_micros() as u64, Ordering::Relaxed);
    }

    pub fn ponderhit(&self) {
        self.restart_clock();
        self.pondering.store(false, Ordering::Relaxed);
    }

    fn elapsed(&self) -> Duration {
        let now = self.base.elapsed().as_micros() as u64;
        Duration::from_micros(now.saturating_sub(self.start_us.load(Ordering::Relaxed)))
    }

    pub fn total_nodes(&self) -> u64 {
        self.nodes.iter().map(|n| n.load(Ordering::Relaxed)).sum()
    }
}

fn lmr_table() -> &'static [[i32; 64]; 64] {
    static T: OnceLock<[[i32; 64]; 64]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0; 64]; 64];
        for (d, row) in t.iter_mut().enumerate().skip(1) {
            for (m, v) in row.iter_mut().enumerate().skip(1) {
                *v = (0.8 + (d as f64).ln() * (m as f64).ln() / 2.3) as i32;
            }
        }
        t
    })
}

/// Межі часу на хід (відлік – у Shared, бо його переносить ponderhit).
pub struct Clock<'a> {
    shared: &'a Shared,
    soft: Option<Duration>,
    hard: Option<Duration>,
}

impl<'a> Clock<'a> {
    pub fn new(shared: &'a Shared, limits: &Limits, side: usize, overhead: u64) -> Clock<'a> {
        shared.restart_clock();
        let start = shared;
        if let Some(mt) = limits.movetime {
            let t = Duration::from_millis(mt.saturating_sub(overhead).max(1));
            return Clock {
                shared: start,
                soft: Some(t),
                hard: Some(t),
            };
        }
        // Годинник лише одного боку (так буває в python-chess): рахуємо за ним, а не думаємо без кінця.
        let side = if limits.time[side].is_none() && limits.time[side ^ 1].is_some() {
            side ^ 1
        } else {
            side
        };
        if limits.infinite || limits.time[side].is_none() {
            return Clock {
                shared: start,
                soft: None,
                hard: None,
            };
        }
        let time = limits.time[side].unwrap_or(0);
        let inc = limits.inc[side];
        let left = time.saturating_sub(overhead).max(1);
        let mtg = limits.movestogo.unwrap_or(30).clamp(1, 40);
        // М'яка межа: частка часу плюс більша частина надбавки; жорстка – не більше 40% годинника.
        let soft = (left / mtg + inc * 3 / 4).min(left / 2);
        let hard = (soft * 4).min(left * 2 / 5).max(soft);
        Clock {
            shared: start,
            soft: Some(Duration::from_millis(soft.max(1))),
            hard: Some(Duration::from_millis(hard.max(1))),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.shared.elapsed()
    }

    /// Чи починати наступну ітерацію (factor – запас, якщо найкращий хід хитається).
    fn continue_iterating(&self, factor: f64) -> bool {
        match self.soft {
            None => true,
            Some(s) => self.elapsed().as_secs_f64() < s.as_secs_f64() * factor * 0.55,
        }
    }

    fn out_of_time(&self) -> bool {
        matches!(self.hard, Some(h) if self.elapsed() >= h)
    }
}

const HISTORY_MAX: i32 = 16384;
const CORR_SIZE: usize = 16384;
const CORR_GRAIN: i32 = 256;
const CORR_MAX: i32 = 128 * CORR_GRAIN;

struct Worker<'a> {
    id: usize,
    shared: &'a Shared,
    clock: &'a Clock<'a>,
    limits: &'a Limits,
    nodes: u64,
    seldepth: usize,
    /// Ключі позицій від початку партії до поточного вузла (для повторень).
    keys: Vec<u64>,
    /// Для кожного ключа – скільки ходів минуло від останнього нульового (крізь нього повторів немає).
    since_null: Vec<u16>,
    /// Скільки ключів з keys – з історії партії (до кореня пошуку).
    root_len: usize,
    killers: [[Move; 2]; MAX_PLY + 4],
    history: Box<[[[i32; 64]; 64]; 2]>,
    counter: Box<[[Move; 64]; 12]>,
    /// Історія продовжень: [фігура й поле попереднього ходу][фігура й поле цього ходу] (для 1 і 2 ходів тому).
    cont: Vec<i32>,
    /// Історія взять: [фігура][поле][хто взятий].
    capture_hist: Box<[[[i32; 6]; 64]; 12]>,
    /// Корекція статичної оцінки за пішаковою структурою: [бік][частина ключа пішаків].
    correction: Box<[[i32; CORR_SIZE]; 2]>,
    stack_move: [Move; MAX_PLY + 4],
    stack_piece: [u8; MAX_PLY + 4],
    eval_stack: [i32; MAX_PLY + 4],
    pv: Box<[[Move; MAX_PLY + 4]; MAX_PLY + 4]>,
    pv_len: [usize; MAX_PLY + 4],
    stopped: bool,
    /// Накопичувачі мережі на кожну глибину: [глибина][бік][нейрон].
    acc: Vec<i16>,
}

pub struct Report {
    pub best: Move,
    pub ponder: Move,
    pub score: i32,
}

#[inline(always)]
fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_BOUND {
        score + ply as i32
    } else if score <= -MATE_BOUND {
        score - ply as i32
    } else {
        score
    }
}

#[inline(always)]
fn score_from_tt(score: i32, ply: usize, halfmove: u16) -> i32 {
    if score >= MATE_BOUND {
        // Мат, якого вже не встигнути до правила 50 ходів, – не мат.
        if MATE - score > 100 - halfmove as i32 {
            return MATE_BOUND - 1;
        }
        score - ply as i32
    } else if score <= -MATE_BOUND {
        if MATE + score > 100 - halfmove as i32 {
            return -MATE_BOUND + 1;
        }
        score + ply as i32
    } else {
        score
    }
}

impl<'a> Worker<'a> {
    fn new(
        id: usize,
        shared: &'a Shared,
        clock: &'a Clock<'a>,
        limits: &'a Limits,
        history_keys: &[u64],
    ) -> Worker<'a> {
        let mut keys = Vec::with_capacity(history_keys.len() + MAX_PLY + 8);
        keys.extend_from_slice(history_keys);
        let since_null = (0..keys.len())
            .map(|i| i.min(u16::MAX as usize) as u16)
            .collect();
        Worker {
            id,
            shared,
            clock,
            limits,
            nodes: 0,
            seldepth: 0,
            root_len: keys.len(),
            since_null,
            keys,
            killers: [[Move::NONE; 2]; MAX_PLY + 4],
            history: Box::new([[[0; 64]; 64]; 2]),
            counter: Box::new([[Move::NONE; 64]; 12]),
            cont: vec![0; 12 * 64 * 12 * 64],
            capture_hist: Box::new([[[0; 6]; 64]; 12]),
            correction: vec![[0; CORR_SIZE]; 2]
                .into_boxed_slice()
                .try_into()
                .expect("розмір"),
            stack_move: [Move::NONE; MAX_PLY + 4],
            stack_piece: [NO_PIECE; MAX_PLY + 4],
            eval_stack: [0; MAX_PLY + 4],
            pv: Box::new([[Move::NONE; MAX_PLY + 4]; MAX_PLY + 4]),
            pv_len: [0; MAX_PLY + 4],
            stopped: false,
            acc: vec![
                0;
                shared
                    .net
                    .map(|n| (MAX_PLY + 8) * 2 * n.hidden)
                    .unwrap_or(0)
            ],
        }
    }

    /// Накопичувачі для кореня – з нуля.
    fn acc_root(&mut self, p: &Position) {
        if let Some(net) = self.shared.net {
            let h = net.hidden;
            let mut acc = [Vec::with_capacity(h), Vec::with_capacity(h)];
            net.accumulate(p, &mut acc);
            self.acc[..h].copy_from_slice(&acc[0]);
            self.acc[h..2 * h].copy_from_slice(&acc[1]);
        }
    }

    /// Накопичувачі глибини ply + 1 після ходу m з позиції p (глибини ply); нульовий хід – копія.
    fn acc_push(&mut self, p: &Position, m: Move, ply: usize) {
        let Some(net) = self.shared.net else { return };
        let h = net.hidden;
        let (lo, hi) = self.acc.split_at_mut((ply + 1) * 2 * h);
        let src = &lo[ply * 2 * h..];
        let dst = &mut hi[..2 * h];
        if m.is_none() {
            dst.copy_from_slice(src);
            return;
        }
        let us = p.side;
        let (from, to) = (m.from(), m.to());
        let piece = p.piece_at(from);
        let mut adds: [(u8, Square); 2] = [(NO_PIECE, 0); 2];
        let mut subs: [(u8, Square); 2] = [(NO_PIECE, 0); 2];
        let (mut na, mut ns) = (0, 0);
        if m.is_castle() {
            let kingside = to > from;
            let (kto, rto) = if kingside {
                (relative_sq(us, 6), relative_sq(us, 5))
            } else {
                (relative_sq(us, 2), relative_sq(us, 3))
            };
            subs[0] = (piece, from);
            subs[1] = (p.piece_at(to), to);
            adds[0] = (piece, kto);
            adds[1] = (p.piece_at(to), rto);
            na = 2;
            ns = 2;
        } else {
            subs[ns] = (piece, from);
            ns += 1;
            adds[na] = (
                if m.is_promo() {
                    piece_of(us, m.promo_kind())
                } else {
                    piece
                },
                to,
            );
            na += 1;
            if m.is_ep() {
                let cap = if us == WHITE { to - 8 } else { to + 8 };
                subs[ns] = (p.piece_at(cap), cap);
                ns += 1;
            } else if p.piece_at(to) != NO_PIECE {
                subs[ns] = (p.piece_at(to), to);
                ns += 1;
            }
        }
        for persp in 0..2 {
            let mut a = [0usize; 2];
            let mut r = [0usize; 2];
            for i in 0..na {
                a[i] = feature(persp, adds[i].0, adds[i].1);
            }
            for i in 0..ns {
                r[i] = feature(persp, subs[i].0, subs[i].1);
            }
            net.update(
                &src[persp * h..(persp + 1) * h],
                &mut dst[persp * h..(persp + 1) * h],
                &a[..na],
                &r[..ns],
            );
        }
    }

    /// Статична оцінка: мережа або ручна; ближче до правила 50 ходів – ближче до нуля.
    #[inline(always)]
    fn static_eval(&self, p: &Position, ply: usize) -> i32 {
        let v = match self.shared.net {
            Some(net) => {
                let h = net.hidden;
                let base = ply * 2 * h;
                let (white, black) = (&self.acc[base..base + h], &self.acc[base + h..base + 2 * h]);
                if p.side == WHITE {
                    net.output(white, black)
                } else {
                    net.output(black, white)
                }
            }
            None => evaluate(p),
        };
        v * (200 - p.halfmove.min(100) as i32) / 200
    }

    #[inline(always)]
    fn check_stop(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        if self.nodes & 1023 == 0 {
            self.shared.nodes[self.id].store(self.nodes, Ordering::Relaxed);
            if self.shared.stop.load(Ordering::Relaxed) {
                self.stopped = true;
            } else if self.id == 0 {
                let pondering = self.shared.pondering.load(Ordering::Relaxed);
                let nodes_out =
                    matches!(self.limits.nodes, Some(n) if self.shared.total_nodes() >= n);
                if nodes_out || (!pondering && self.clock.out_of_time()) {
                    self.shared.stop.store(true, Ordering::Relaxed);
                    self.stopped = true;
                }
            }
        }
        self.stopped
    }

    /// Нічия: правило 50 ходів, мертвий матеріал або повторення. Повтор позиції з дерева пошуку –
    /// одразу нічия (суперник може повторити ще раз), з історії партії до кореня – лише втретє.
    fn is_draw(&self, p: &Position) -> bool {
        if p.halfmove >= 100 && (!p.in_check() || p.legal_moves().len > 0) {
            return true;
        }
        if p.insufficient_material() {
            return true;
        }
        let cur = self.keys.len() as isize - 1;
        // Раніше незворотного ходу чи нульового ходу позиція повторитися не могла.
        let back = (p.halfmove as isize).min(self.since_null.last().map_or(0, |&n| n as isize));
        let oldest = cur - back;
        let mut j = cur - 4;
        let mut earlier = 0;
        while j >= 0 && j >= oldest {
            if self.keys[j as usize] == p.key {
                if j as usize + 1 >= self.root_len {
                    return true;
                }
                earlier += 1;
                if earlier >= 2 {
                    return true;
                }
            }
            j -= 2;
        }
        false
    }

    fn score_moves(
        &self,
        p: &Position,
        list: &MoveList,
        scores: &mut [i32; 256],
        tt_move: Move,
        ply: usize,
    ) {
        let prev = if ply > 0 {
            self.stack_move[ply - 1]
        } else {
            Move::NONE
        };
        let prev_piece = if ply > 0 {
            self.stack_piece[ply - 1]
        } else {
            NO_PIECE
        };
        let countermove = if !prev.is_none() && prev_piece != NO_PIECE {
            self.counter[prev_piece as usize][prev.to() as usize]
        } else {
            Move::NONE
        };
        for (score, &m) in scores.iter_mut().zip(list.iter()) {
            *score = if m == tt_move {
                1 << 30
            } else if m.is_tactical() {
                let victim = if m.is_ep() {
                    PAWN
                } else {
                    kind_of(p.piece_at(m.to()))
                };
                let victim_value = if m.is_capture() { SEE_VALUE[victim] } else { 0 };
                let promo = if m.is_promo() {
                    SEE_VALUE[m.promo_kind()]
                } else {
                    0
                };
                let attacker = kind_of(p.piece_at(m.from()));
                let hist = self.capture_hist[p.piece_at(m.from()) as usize][m.to() as usize]
                    [Self::victim(p, m)];
                let base = victim_value * 10 + promo - SEE_VALUE[attacker] / 10 + hist / 16;
                if see_ge(p, m, -50) {
                    (1 << 28) + base
                } else {
                    -(1 << 20) + base
                }
            } else if m == self.killers[ply][0] {
                (1 << 27) + 2
            } else if m == self.killers[ply][1] {
                (1 << 27) + 1
            } else if m == countermove {
                1 << 26
            } else {
                self.quiet_score(p, m, ply)
            };
        }
    }

    fn update_history(&mut self, side: usize, m: Move, bonus: i32) {
        let h = &mut self.history[side][m.from() as usize][m.to() as usize];
        *h += bonus - *h * bonus.abs() / HISTORY_MAX;
    }

    /// Індекси історії продовжень для ходу (piece, to) щодо ходів 1 і 2 півходи тому.
    #[inline(always)]
    fn cont_slots(&self, ply: usize, piece: u8, to: Square) -> [Option<usize>; 2] {
        let mut out = [None, None];
        for (k, back) in [1usize, 2].into_iter().enumerate() {
            if ply >= back {
                let (pm, pp) = (self.stack_move[ply - back], self.stack_piece[ply - back]);
                if !pm.is_none() && pp != NO_PIECE {
                    out[k] = Some(
                        ((pp as usize * 64 + pm.to() as usize) * 12 + piece as usize) * 64
                            + to as usize,
                    );
                }
            }
        }
        out
    }

    /// Уся історія тихого ходу: власна плюс продовження.
    #[inline(always)]
    fn quiet_score(&self, p: &Position, m: Move, ply: usize) -> i32 {
        let piece = p.piece_at(m.from());
        let mut v = self.history[p.side][m.from() as usize][m.to() as usize];
        for i in self.cont_slots(ply, piece, m.to()).into_iter().flatten() {
            v += self.cont[i];
        }
        v
    }

    fn update_quiet(&mut self, p: &Position, m: Move, ply: usize, bonus: i32) {
        self.update_history(p.side, m, bonus);
        let piece = p.piece_at(m.from());
        for i in self.cont_slots(ply, piece, m.to()).into_iter().flatten() {
            let h = &mut self.cont[i];
            *h += bonus - *h * bonus.abs() / HISTORY_MAX;
        }
    }

    #[inline(always)]
    fn corrected(&self, p: &Position, raw: i32) -> i32 {
        let c = self.correction[p.side][(p.pawn_key as usize) % CORR_SIZE];
        (raw + c / CORR_GRAIN).clamp(-MATE_BOUND + 1, MATE_BOUND - 1)
    }

    fn update_correction(&mut self, p: &Position, diff: i32, depth: i32) {
        let w = (depth + 1).min(16);
        let c = &mut self.correction[p.side][(p.pawn_key as usize) % CORR_SIZE];
        *c = ((*c * (256 - w) + diff * CORR_GRAIN * w) / 256).clamp(-CORR_MAX, CORR_MAX);
    }

    #[inline(always)]
    fn victim(p: &Position, m: Move) -> usize {
        if m.is_ep() || !m.is_capture() {
            PAWN
        } else {
            kind_of(p.piece_at(m.to()))
        }
    }

    fn update_capture(&mut self, p: &Position, m: Move, bonus: i32) {
        let h = &mut self.capture_hist[p.piece_at(m.from()) as usize][m.to() as usize]
            [Self::victim(p, m)];
        *h += bonus - *h * bonus.abs() / HISTORY_MAX;
    }

    fn push(&mut self, ply: usize, m: Move, piece: u8, key: u64) {
        self.stack_move[ply] = m;
        self.stack_piece[ply] = piece;
        let since = if m.is_none() {
            0
        } else {
            self.since_null.last().map_or(0, |&n| n.saturating_add(1))
        };
        self.keys.push(key);
        self.since_null.push(since);
    }

    fn pop(&mut self) {
        self.keys.pop();
        self.since_null.pop();
    }

    fn quiescence(&mut self, p: &Position, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        if self.check_stop() {
            return 0;
        }
        self.pv_len[ply] = 0;
        self.seldepth = self.seldepth.max(ply);
        if ply > 0 && self.is_draw(p) {
            return 0;
        }
        if ply >= MAX_PLY {
            return self.static_eval(p, ply);
        }
        let in_check = p.in_check();
        let tt = self.shared.tt.probe(p.key);
        if let Some(e) = tt {
            let s = score_from_tt(e.score, ply, p.halfmove);
            if (e.bound == BOUND_EXACT)
                || (e.bound == BOUND_LOWER && s >= beta)
                || (e.bound == BOUND_UPPER && s <= alpha)
            {
                return s;
            }
        }
        let mut best;
        let stand;
        if in_check {
            stand = -INF;
            best = -MATE + ply as i32;
        } else {
            stand = tt
                .map(|e| e.eval)
                .unwrap_or_else(|| self.static_eval(p, ply));
            best = stand;
            if stand >= beta {
                return stand;
            }
            if stand > alpha {
                alpha = stand;
            }
        }
        let mut list = MoveList::new();
        p.generate(&mut list, !in_check);
        let mut scores = [0i32; 256];
        let tt_move = tt.map(|e| e.mv).unwrap_or(Move::NONE);
        self.score_moves(p, &list, &mut scores, tt_move, ply);
        let mut best_move = Move::NONE;
        let mut legal = 0;
        for i in 0..list.len {
            pick(&mut list, &mut scores, i);
            let m = list.moves[i];
            if !p.is_legal(m) {
                continue;
            }
            legal += 1;
            if !in_check {
                // Взяття, що програє матеріал, і марні взяття (навіть з виграшем не дотягують до alpha).
                if !see_ge(p, m, 0) {
                    continue;
                }
                let gain = if m.is_capture() {
                    SEE_VALUE[if m.is_ep() {
                        PAWN
                    } else {
                        kind_of(p.piece_at(m.to()))
                    }]
                } else {
                    0
                } + if m.is_promo() {
                    SEE_VALUE[QUEEN] - SEE_VALUE[PAWN]
                } else {
                    0
                };
                if stand + gain + 200 <= alpha && !m.is_promo() {
                    continue;
                }
            }
            let child = p.make(m);
            self.acc_push(p, m, ply);
            self.push(ply, m, p.piece_at(m.from()), child.key);
            let score = -self.quiescence(&child, -beta, -alpha, ply + 1);
            self.pop();
            if self.stopped {
                return 0;
            }
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    best_move = m;
                    if score >= beta {
                        break;
                    }
                }
            }
        }
        if in_check && legal == 0 {
            return -MATE + ply as i32;
        }
        let bound = if best >= beta {
            BOUND_LOWER
        } else {
            BOUND_UPPER
        };
        self.shared.tt.store(
            p.key,
            best_move,
            score_to_tt(best, ply),
            if in_check { 0 } else { stand },
            -1,
            bound,
            false,
        );
        best
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        p: &Position,
        mut alpha: i32,
        mut beta: i32,
        mut depth: i32,
        ply: usize,
        cut: bool,
        excluded: Move,
    ) -> i32 {
        let pv_node = beta - alpha > 1;
        let root = ply == 0;
        self.pv_len[ply] = 0;
        if p.in_check() {
            depth += 1; // шах – завжди дивимося на хід глибше
        }
        if depth <= 0 {
            return self.quiescence(p, alpha, beta, ply);
        }
        self.nodes += 1;
        if self.check_stop() {
            return 0;
        }
        self.seldepth = self.seldepth.max(ply);
        if !root {
            if self.is_draw(p) {
                // Невелике коливання навколо нуля: різні потоки не застрягають в однаковій нічиїй.
                return 1 - (self.nodes & 2) as i32;
            }
            if ply >= MAX_PLY {
                return self.static_eval(p, ply);
            }
            // Найкоротший мат: далі шукати немає сенсу.
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
        }
        let in_check = p.in_check();
        let tt = if excluded.is_none() {
            self.shared.tt.probe(p.key)
        } else {
            None
        };
        let tt_move = tt.map(|e| e.mv).unwrap_or(Move::NONE);
        let tt_pv = pv_node || tt.map(|e| e.pv).unwrap_or(false);
        if let Some(e) = tt {
            let s = score_from_tt(e.score, ply, p.halfmove);
            if !pv_node
                && e.depth >= depth
                && p.halfmove < 90
                && ((e.bound == BOUND_EXACT)
                    || (e.bound == BOUND_LOWER && s >= beta)
                    || (e.bound == BOUND_UPPER && s <= alpha))
            {
                return s;
            }
        }

        // Сира статична оцінка (вона й іде в таблицю) і виправлена історією корекції.
        let raw_eval = if in_check {
            -INF
        } else {
            tt.map(|e| e.eval)
                .unwrap_or_else(|| self.static_eval(p, ply))
        };
        let corrected = if in_check {
            -INF
        } else {
            self.corrected(p, raw_eval)
        };
        let eval = if in_check {
            -INF
        } else if let Some(e) = tt {
            let mut v = corrected;
            let s = score_from_tt(e.score, ply, p.halfmove);
            // Оцінка з таблиці точніша за статичну, якщо межа це дозволяє.
            let tighter = (e.bound == BOUND_LOWER && s > v)
                || (e.bound == BOUND_UPPER && s < v)
                || e.bound == BOUND_EXACT;
            if tighter && s.abs() < MATE_BOUND {
                v = s;
            }
            v
        } else {
            corrected
        };
        self.eval_stack[ply] = eval;
        let improving = !in_check && ply >= 2 && eval > self.eval_stack[ply - 2];
        if ply + 1 < MAX_PLY {
            self.killers[ply + 1] = [Move::NONE; 2];
        }

        if !pv_node && !in_check && excluded.is_none() {
            // Зворотне відсічення марності: статична оцінка вже набагато вища за beta.
            if depth <= 8
                && eval - 85 * (depth - improving as i32) >= beta
                && eval.abs() < MATE_BOUND
            {
                return eval;
            }
            // Нульовий хід: навіть пропустивши хід, ми не нижче beta.
            if depth >= 3
                && eval >= beta
                && ply > 0
                && !self.stack_move[ply - 1].is_none()
                && p.non_pawn_material(p.side)
                && beta.abs() < MATE_BOUND
            {
                let r = 3 + depth / 4 + ((eval - beta) / 200).min(3);
                let child = p.make_null();
                self.acc_push(p, Move::NONE, ply);
                self.push(ply, Move::NONE, NO_PIECE, child.key);
                let score = -self.negamax(
                    &child,
                    -beta,
                    -beta + 1,
                    depth - r,
                    ply + 1,
                    !cut,
                    Move::NONE,
                );
                self.pop();
                if self.stopped {
                    return 0;
                }
                if score >= beta {
                    return if score >= MATE_BOUND { beta } else { score };
                }
            }
        }
        // Без ходу з таблиці на великій глибині – спершу дешевший пошук.
        if tt_move.is_none() && depth >= 5 && (pv_node || cut) {
            depth -= 1;
        }

        let mut list = MoveList::new();
        p.generate(&mut list, false);
        let mut scores = [0i32; 256];
        self.score_moves(p, &list, &mut scores, tt_move, ply);

        let mut best = -INF;
        let mut best_move = Move::NONE;
        let original_alpha = alpha;
        let mut legal = 0;
        let mut quiets_tried: [Move; 64] = [Move::NONE; 64];
        let mut n_quiets = 0;
        let mut captures_tried: [Move; 32] = [Move::NONE; 32];
        let mut n_captures = 0;
        let lmr = lmr_table();

        for i in 0..list.len {
            pick(&mut list, &mut scores, i);
            let m = list.moves[i];
            if m == excluded || !p.is_legal(m) {
                continue;
            }
            legal += 1;
            let quiet = !m.is_tactical();
            let is_killer = m == self.killers[ply][0] || m == self.killers[ply][1];

            if !root && best > -MATE_BOUND && p.non_pawn_material(p.side) {
                if quiet {
                    // Пізні тихі ходи на малій глибині не дивимося зовсім.
                    let lmp = (3 + depth * depth) / (2 - improving as i32);
                    if !pv_node && depth <= 8 && legal > lmp && !in_check {
                        continue;
                    }
                    // Марний тихий хід: оцінка з запасом не дотягує до alpha.
                    if !in_check && depth <= 7 && eval + 90 + 110 * depth <= alpha && !is_killer {
                        continue;
                    }
                    if depth <= 8 && !see_ge(p, m, -60 * depth) {
                        continue;
                    }
                } else if depth <= 6 && !see_ge(p, m, -100 * depth) {
                    continue;
                }
            }

            // Однина: хід з таблиці набагато кращий за решту – дивимося його глибше.
            let mut ext = 0;
            if !root
                && m == tt_move
                && excluded.is_none()
                && depth >= 7
                && ply < MAX_PLY / 2
                && tt
                    .map(|e| {
                        e.depth >= depth - 3 && e.bound != BOUND_UPPER && e.score.abs() < MATE_BOUND
                    })
                    .unwrap_or(false)
            {
                let tt_score = tt
                    .map(|e| score_from_tt(e.score, ply, p.halfmove))
                    .unwrap_or(0);
                let s_beta = tt_score - depth * 2;
                let s = self.negamax(p, s_beta - 1, s_beta, (depth - 1) / 2, ply, cut, m);
                if self.stopped {
                    return 0;
                }
                if s < s_beta {
                    ext = 1;
                } else if s_beta >= beta {
                    return s_beta; // кілька ходів б'ють beta: відсікаємо одразу
                }
            }

            let child = p.make(m);
            let piece = p.piece_at(m.from());
            self.acc_push(p, m, ply);
            self.push(ply, m, piece, child.key);
            let new_depth = depth - 1 + ext;
            let mut score;
            if legal == 1 {
                score = -self.negamax(&child, -beta, -alpha, new_depth, ply + 1, false, Move::NONE);
            } else {
                // Пізні ходи – спершу з меншою глибиною й нульовим вікном.
                let mut r = 0;
                if depth >= 3 && legal > 1 + pv_node as i32 * 2 && (quiet || !see_ge(p, m, 0)) {
                    r = lmr[depth.min(63) as usize][legal.min(63) as usize];
                    r -= pv_node as i32;
                    r -= is_killer as i32;
                    r += cut as i32;
                    r += !improving as i32;
                    r -= child.in_check() as i32;
                    if quiet {
                        r -= self.quiet_score(p, m, ply) / 8000;
                    }
                    r -= tt_pv as i32;
                    r = r.clamp(0, new_depth - 1);
                }
                score = -self.negamax(
                    &child,
                    -alpha - 1,
                    -alpha,
                    new_depth - r,
                    ply + 1,
                    true,
                    Move::NONE,
                );
                if score > alpha && r > 0 {
                    score = -self.negamax(
                        &child,
                        -alpha - 1,
                        -alpha,
                        new_depth,
                        ply + 1,
                        !cut,
                        Move::NONE,
                    );
                }
                if score > alpha && score < beta {
                    score =
                        -self.negamax(&child, -beta, -alpha, new_depth, ply + 1, false, Move::NONE);
                }
            }
            self.pop();
            if self.stopped {
                return 0;
            }
            if quiet && n_quiets < 64 {
                quiets_tried[n_quiets] = m;
                n_quiets += 1;
            } else if !quiet && n_captures < 32 {
                captures_tried[n_captures] = m;
                n_captures += 1;
            }
            if score > best {
                best = score;
                if score > alpha {
                    best_move = m;
                    alpha = score;
                    // Головна лінія.
                    self.pv[ply][0] = m;
                    let len = self.pv_len[ply + 1];
                    for j in 0..len {
                        self.pv[ply][j + 1] = self.pv[ply + 1][j];
                    }
                    self.pv_len[ply] = len + 1;
                    if score >= beta {
                        let bonus = (depth * depth * 16).min(1600);
                        // Взяття, що не дали відсічення, – гірші, ніж здавалося.
                        for &c in &captures_tried[..n_captures] {
                            if c != m {
                                self.update_capture(p, c, -bonus);
                            }
                        }
                        if !quiet {
                            self.update_capture(p, m, bonus);
                        }
                        if quiet {
                            if self.killers[ply][0] != m {
                                self.killers[ply][1] = self.killers[ply][0];
                                self.killers[ply][0] = m;
                            }
                            self.update_quiet(p, m, ply, bonus);
                            for &q in &quiets_tried[..n_quiets] {
                                if q != m {
                                    self.update_quiet(p, q, ply, -bonus);
                                }
                            }
                            if ply > 0
                                && !self.stack_move[ply - 1].is_none()
                                && self.stack_piece[ply - 1] != NO_PIECE
                            {
                                let pp = self.stack_piece[ply - 1] as usize;
                                self.counter[pp][self.stack_move[ply - 1].to() as usize] = m;
                            }
                        }
                        break;
                    }
                }
            }
        }
        if legal == 0 {
            return if !excluded.is_none() {
                alpha
            } else if in_check {
                -MATE + ply as i32
            } else {
                0
            };
        }
        if excluded.is_none() {
            let bound = if best >= beta {
                BOUND_LOWER
            } else if best > original_alpha {
                BOUND_EXACT
            } else {
                BOUND_UPPER
            };
            self.shared.tt.store(
                p.key,
                best_move,
                score_to_tt(best, ply),
                if in_check { 0 } else { raw_eval },
                depth,
                bound,
                tt_pv,
            );
            // Наскільки пошук розійшовся зі статичною оцінкою – запам'ятати для цієї пішакової структури.
            let reliable = !(bound == BOUND_LOWER && best <= corrected)
                && !(bound == BOUND_UPPER && best >= corrected);
            if !in_check
                && (best_move.is_none() || !best_move.is_tactical())
                && reliable
                && best.abs() < MATE_BOUND
            {
                self.update_correction(p, best - raw_eval, depth);
            }
        }
        best
    }
}

/// Вибір найкращого ще не переглянутого ходу (часткове сортування).
#[inline(always)]
fn pick(list: &mut MoveList, scores: &mut [i32; 256], i: usize) {
    let mut best = i;
    for j in i + 1..list.len {
        if scores[j] > scores[best] {
            best = j;
        }
    }
    if best != i {
        list.moves.swap(i, best);
        scores.swap(i, best);
    }
}

fn score_str(score: i32) -> String {
    if score >= MATE_BOUND {
        format!("mate {}", (MATE - score + 1) / 2)
    } else if score <= -MATE_BOUND {
        format!("mate -{}", (MATE + score) / 2)
    } else {
        format!("cp {score}")
    }
}

/// Пошук з кореня: повертає найкращий хід (і хід для обдумування на час суперника).
pub fn think(
    root: &Position,
    history_keys: &[u64],
    limits: &Limits,
    shared: Arc<Shared>,
    threads: usize,
    overhead: u64,
    print: bool,
) -> Report {
    let legal = root.legal_moves();
    if legal.len == 0 {
        return Report {
            best: Move::NONE,
            ponder: Move::NONE,
            score: if root.in_check() { -MATE } else { 0 },
        };
    }
    shared.tt.new_search();
    for n in &shared.nodes {
        n.store(0, Ordering::Relaxed);
    }
    let clock = Clock::new(&shared, limits, root.side, overhead);
    let max_depth = limits
        .depth
        .unwrap_or(MAX_PLY as i32 - 4)
        .clamp(1, MAX_PLY as i32 - 4);
    let mut result = Report {
        best: legal.moves[0],
        ponder: Move::NONE,
        score: 0,
    };

    std::thread::scope(|scope| {
        for id in 1..threads.max(1) {
            let shared = &shared;
            let clock = &clock;
            scope.spawn(move || {
                let mut w = Worker::new(id, shared, clock, limits, history_keys);
                w.acc_root(root);
                let mut depth = 1 + (id as i32 % 2);
                while depth <= max_depth && !w.stopped {
                    w.negamax(root, -INF, INF, depth, 0, false, Move::NONE);
                    depth += 1;
                }
                shared.nodes[id].store(w.nodes, Ordering::Relaxed);
            });
        }

        let mut w = Worker::new(0, &shared, &clock, limits, history_keys);
        w.acc_root(root);
        let mut score = 0;
        let mut stable = 0;
        let mut last_best = Move::NONE;
        for depth in 1..=max_depth {
            // Вікно прагнення навколо минулої оцінки.
            let mut delta = 18;
            let (mut alpha, mut beta) = if depth >= 4 {
                (score - delta, score + delta)
            } else {
                (-INF, INF)
            };
            let mut search_depth = depth;
            loop {
                let s = w.negamax(root, alpha, beta, search_depth, 0, false, Move::NONE);
                if w.stopped {
                    break;
                }
                if s <= alpha {
                    beta = (alpha + beta) / 2;
                    alpha = (s - delta).max(-INF);
                    search_depth = depth;
                } else if s >= beta {
                    beta = (s + delta).min(INF);
                    search_depth = (search_depth - 1).max(depth - 3).max(1);
                } else {
                    score = s;
                    break;
                }
                delta += delta / 2 + 5;
            }
            if w.stopped && depth > 1 {
                break;
            }
            if w.pv_len[0] > 0 {
                let best = w.pv[0][0];
                stable = if best == last_best { stable + 1 } else { 0 };
                last_best = best;
                result.best = best;
                result.ponder = if w.pv_len[0] > 1 {
                    w.pv[0][1]
                } else {
                    Move::NONE
                };
                result.score = score;
            }
            shared.nodes[0].store(w.nodes, Ordering::Relaxed);
            if print {
                let ms = clock.elapsed().as_millis().max(1) as u64;
                let nodes = shared.total_nodes();
                let mut pos = *root;
                let mut pv = String::new();
                for j in 0..w.pv_len[0] {
                    let m = w.pv[0][j];
                    if !pos.legal_moves().iter().any(|&x| x == m) {
                        break;
                    }
                    pv.push(' ');
                    pv.push_str(&pos.move_to_uci(m));
                    pos = pos.make(m);
                }
                println!(
                    "info depth {} seldepth {} score {} nodes {} nps {} hashfull {} time {} pv{}",
                    depth,
                    w.seldepth,
                    score_str(score),
                    nodes,
                    nodes * 1000 / ms,
                    shared.tt.hashfull(),
                    ms,
                    pv
                );
            }
            if let Some(m) = limits.mate {
                if score >= MATE_BOUND && (MATE - score + 1) / 2 <= m {
                    break;
                }
            }
            // Єдиний хід або стабільний найкращий – менше часу; хитання – більше.
            let factor = if legal.len == 1 {
                0.2
            } else if stable >= 4 {
                0.7
            } else if stable == 0 {
                1.5
            } else {
                1.0
            };
            let pondering = shared.pondering.load(Ordering::Relaxed);
            if !pondering && !limits.infinite && !clock.continue_iterating(factor) {
                break;
            }
        }
        // На час суперника чи в нескінченному пошуку відповідь чекає команди stop.
        while (limits.infinite || shared.pondering.load(Ordering::Relaxed))
            && !shared.stop.load(Ordering::Relaxed)
        {
            std::thread::sleep(Duration::from_millis(2));
            if !shared.pondering.load(Ordering::Relaxed) && !limits.infinite && clock.out_of_time()
            {
                break;
            }
        }
        shared.stop.store(true, Ordering::Relaxed);
    });
    shared.stop.store(false, Ordering::Relaxed);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Накопичувачі, оновлені ходами, збігаються з перерахованими з нуля (усі види ходів).
    #[test]
    fn incremental_accumulator() {
        crate::bitboard::init();
        let net: &'static Network = Box::leak(Box::new(test_net()));
        let shared = Shared::new(1, 1).with_net(Some(net));
        let limits = Limits::default();
        let clock = Clock::new(&shared, &limits, WHITE, 0);
        let mut w = Worker::new(0, &shared, &clock, &limits, &[]);
        let mut seed = 77u64;
        for fen in [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "8/8/8/KPp4r/8/8/8/7k w - c6 0 2",
            "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
        ] {
            let mut p = Position::from_fen(fen).unwrap();
            w.acc_root(&p);
            for ply in 0..40 {
                let moves = p.legal_moves();
                if moves.len == 0 {
                    break;
                }
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let m = moves.moves[(seed % moves.len as u64) as usize];
                w.acc_push(&p, m, ply);
                p = p.make(m);
                assert_eq!(
                    w.static_eval(&p, ply + 1),
                    net.evaluate(&p) * (200 - p.halfmove.min(100) as i32) / 200,
                    "{}",
                    p.fen()
                );
            }
        }
    }

    fn test_net() -> Network {
        let h = 16;
        let mut s = 5u64;
        let mut r = |k: i32| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % (2 * k as u64 + 1)) as i32 - k) as i16
        };
        Network {
            hidden: h,
            w1: (0..crate::nnue::INPUTS * h).map(|_| r(60)).collect(),
            b1: (0..h).map(|_| r(60)).collect(),
            w2: (0..2 * h).map(|_| r(60)).collect(),
            b2: 100,
        }
    }
}
