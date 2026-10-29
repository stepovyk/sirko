//! Таблиця переставлень, спільна для всіх потоків пошуку без блокувань:
//! кожен запис – два атомарні слова, ключ зберігається як (ключ ^ дані), тож розірваний
//! одночасним записом запис просто не впізнається.

use crate::types::Move;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

pub const BOUND_NONE: u8 = 0;
pub const BOUND_UPPER: u8 = 1; // справжня оцінка не вища
pub const BOUND_LOWER: u8 = 2; // справжня оцінка не нижча
pub const BOUND_EXACT: u8 = 3;

#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub mv: Move,
    pub score: i32,
    pub eval: i32,
    pub depth: i32,
    pub bound: u8,
    pub pv: bool,
}

struct Slot {
    key: AtomicU64,
    data: AtomicU64,
}

pub struct Table {
    slots: Vec<Slot>,
    age: AtomicU8,
}

const BUCKET: usize = 2; // два записи на кошик: глибший і новіший

fn pack(mv: Move, score: i32, eval: i32, depth: i32, bound: u8, pv: bool, age: u8) -> u64 {
    (mv.0 as u64)
        | ((score as i16 as u16 as u64) << 16)
        | ((eval as i16 as u16 as u64) << 32)
        | (((depth.clamp(-1, 254) + 1) as u8 as u64) << 48)
        | ((bound as u64 & 3) << 56)
        | ((pv as u64) << 58)
        | ((age as u64 & 31) << 59)
}

fn unpack(d: u64) -> (Entry, u8) {
    (
        Entry {
            mv: Move(d as u16),
            score: (d >> 16) as u16 as i16 as i32,
            eval: (d >> 32) as u16 as i16 as i32,
            depth: ((d >> 48) & 0xFF) as i32 - 1,
            bound: ((d >> 56) & 3) as u8,
            pv: (d >> 58) & 1 != 0,
        },
        ((d >> 59) & 31) as u8,
    )
}

impl Table {
    pub fn new(mb: usize) -> Table {
        let n = (mb.max(1) * 1024 * 1024 / 16 / BUCKET).max(1) * BUCKET;
        let mut slots = Vec::with_capacity(n);
        slots.resize_with(n, || Slot {
            key: AtomicU64::new(0),
            data: AtomicU64::new(0),
        });
        Table {
            slots,
            age: AtomicU8::new(0),
        }
    }

    pub fn clear(&self) {
        for s in &self.slots {
            s.key.store(0, Ordering::Relaxed);
            s.data.store(0, Ordering::Relaxed);
        }
        self.age.store(0, Ordering::Relaxed);
    }

    pub fn new_search(&self) {
        let a = self.age.load(Ordering::Relaxed);
        self.age.store((a + 1) & 31, Ordering::Relaxed);
    }

    #[inline(always)]
    fn bucket(&self, key: u64) -> usize {
        // Множення замість остачі: рівномірно на будь-який розмір таблиці.
        let n = (self.slots.len() / BUCKET) as u128;
        (((key as u128) * n) >> 64) as usize * BUCKET
    }

    pub fn probe(&self, key: u64) -> Option<Entry> {
        let b = self.bucket(key);
        for s in &self.slots[b..b + BUCKET] {
            let d = s.data.load(Ordering::Relaxed);
            if s.key.load(Ordering::Relaxed) ^ d == key && d != 0 {
                return Some(unpack(d).0);
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &self,
        key: u64,
        mv: Move,
        score: i32,
        eval: i32,
        depth: i32,
        bound: u8,
        pv: bool,
    ) {
        let age = self.age.load(Ordering::Relaxed);
        let b = self.bucket(key);
        // Той самий ключ – оновити; інакше витіснити найменш цінний (мілкий і старий).
        let mut victim = b;
        let mut worst = i32::MAX;
        for i in b..b + BUCKET {
            let s = &self.slots[i];
            let d = s.data.load(Ordering::Relaxed);
            let k = s.key.load(Ordering::Relaxed) ^ d;
            if k == key || d == 0 {
                victim = i;
                break;
            }
            let (e, a) = unpack(d);
            let stale = ((age as i32 - a as i32) & 31) * 8;
            let value = e.depth - stale;
            if value < worst {
                worst = value;
                victim = i;
            }
        }
        let s = &self.slots[victim];
        let old = s.data.load(Ordering::Relaxed);
        let old_key = s.key.load(Ordering::Relaxed) ^ old;
        let mut mv = mv;
        if old_key == key && old != 0 {
            let (e, _) = unpack(old);
            // Мілкіший запис того ж ключа не затирає глибший, крім точного результату.
            if bound != BOUND_EXACT
                && depth + 3 < e.depth
                && ((age as i32 - ((old >> 59) & 31) as i32) & 31) == 0
            {
                return;
            }
            if mv.is_none() {
                mv = e.mv; // зберегти найкращий хід, якщо новий запис його не має
            }
        }
        let d = pack(mv, score, eval, depth, bound, pv, age);
        s.key.store(key ^ d, Ordering::Relaxed);
        s.data.store(d, Ordering::Relaxed);
    }

    /// Заповненість у проміле (для UCI hashfull): за першою тисячею записів поточного пошуку.
    pub fn hashfull(&self) -> usize {
        let age = self.age.load(Ordering::Relaxed);
        self.slots
            .iter()
            .take(1000)
            .filter(|s| {
                let d = s.data.load(Ordering::Relaxed);
                d != 0 && ((d >> 59) & 31) as u8 == age
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let t = Table::new(1);
        let m = Move(0x1234);
        t.store(0xDEADBEEF, m, -31000, 250, 17, BOUND_LOWER, true);
        let e = t.probe(0xDEADBEEF).unwrap();
        assert_eq!(
            (e.mv, e.score, e.eval, e.depth, e.bound, e.pv),
            (m, -31000, 250, 17, BOUND_LOWER, true)
        );
        assert!(t.probe(0xDEADBEEE).is_none());
        t.store(0xDEADBEEF, Move::NONE, 5, 5, -1, BOUND_UPPER, false);
        // Мілкіший неточний запис не затирає глибший у тому самому пошуку.
        assert_eq!(t.probe(0xDEADBEEF).unwrap().depth, 17);
    }
}
