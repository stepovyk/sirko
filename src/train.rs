//! Власний тренер мережі: Adam на мінівибірках, кілька потоків, ціль – суміш оцінки пошуку
//! й результату партії. Після навчання ваги квантуються в цілі й пишуться у файл мережі.

use crate::datagen::{Sample, RECORD};
use crate::nnue::{bucket, feature, Network, INPUTS, QA, QB};
use crate::types::*;
use std::io::Read;
use std::time::Instant;

pub struct Options {
    pub data: Vec<String>,
    pub hidden: usize,
    pub buckets: usize,
    pub epochs: usize,
    pub batch: usize,
    /// Скільки зразків тримати в пам'яті одночасно.
    pub buffer: usize,
    pub lr: f32,
    pub wdl: f32,
    pub threads: usize,
    pub out: String,
    /// Файл стану: після кожної епохи сюди пишеться все, щоб продовжити навчання в іншому запуску.
    pub state: Option<String>,
    /// Не починати нову епоху, якщо вона не встигне за стільки хвилин від старту.
    pub max_minutes: Option<f64>,
}

/// Межа ваг: після множення на QB = 64 ваги виходу лишаються в ±127, як вимагає рушій.
const CLIP: f32 = 1.98;

struct Model {
    h: usize,
    b: usize,    // кошиків виходу
    p: Vec<f32>, // усі параметри підряд: w1, b1, w2[кошик], b2[кошик]
}

impl Model {
    fn w1(&self) -> usize {
        0
    }
    fn b1(&self) -> usize {
        INPUTS * self.h
    }
    fn w2(&self) -> usize {
        INPUTS * self.h + self.h
    }
    fn b2(&self) -> usize {
        INPUTS * self.h + self.h + self.b * 2 * self.h
    }
    fn len(h: usize, b: usize) -> usize {
        INPUTS * h + h + b * 2 * h + b
    }

    fn new(h: usize, b: usize, seed: u64) -> Model {
        let mut s = seed | 1;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
        };
        let mut p = vec![0.0f32; Model::len(h, b)];
        let m = Model {
            h,
            b,
            p: Vec::new(),
        };
        for x in &mut p[m.w1()..m.b1()] {
            *x = rnd() * 0.15;
        }
        let k = 1.0 / (2.0 * h as f32).sqrt();
        for x in &mut p[m.w2()..m.b2()] {
            *x = rnd() * k;
        }
        Model { h, b, p }
    }

    fn quantize(&self) -> Network {
        let h = self.h;
        let q = |x: f32, k: i32| {
            (x * k as f32)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16
        };
        Network {
            hidden: h,
            buckets: self.b,
            w1: self.p[self.w1()..self.b1()]
                .iter()
                .map(|&x| q(x, QA))
                .collect(),
            b1: self.p[self.b1()..self.w2()]
                .iter()
                .map(|&x| q(x, QA))
                .collect(),
            w2: self.p[self.w2()..self.b2()]
                .iter()
                .map(|&x| q(x, QB))
                .collect(),
            b2: self.p[self.b2()..]
                .iter()
                .map(|&x| (x * (QA * QB) as f32).round() as i32)
                .collect(),
        }
    }
}

/// Ознаки зразка з погляду того, хто ходить, і суперника; ціль у [0, 1].
fn prepare(s: &Sample, wdl: f32) -> ([u16; 32], [u16; 32], usize, f32) {
    let stm = s.stm as usize;
    let mut us = [0u16; 32];
    let mut them = [0u16; 32];
    let mut n = 0;
    for (sq, pc) in s.squares() {
        if n == 32 || pc >= 12 {
            break;
        }
        us[n] = feature(stm, pc, sq) as u16;
        them[n] = feature(stm ^ 1, pc, sq) as u16;
        n += 1;
    }
    let white = s.result as f32 / 2.0;
    let result = if stm == WHITE { white } else { 1.0 - white };
    let eval = 1.0 / (1.0 + (-(s.score as f32) / 400.0).exp());
    (us, them, n, wdl * result + (1.0 - wdl) * eval)
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Втрата на зразку; якщо grad не None – додає градієнт.
fn step(m: &Model, s: &Sample, wdl: f32, grad: Option<&mut [f32]>, acc: &mut [f32]) -> f32 {
    let h = m.h;
    let (us, them, n, target) = prepare(s, wdl);
    let (a_us, a_them) = acc.split_at_mut(h);
    a_us.copy_from_slice(&m.p[m.b1()..m.w2()]);
    a_them.copy_from_slice(&m.p[m.b1()..m.w2()]);
    for i in 0..n {
        let ru = &m.p[us[i] as usize * h..(us[i] as usize + 1) * h];
        let rt = &m.p[them[i] as usize * h..(them[i] as usize + 1) * h];
        for j in 0..h {
            a_us[j] += ru[j];
            a_them[j] += rt[j];
        }
    }
    let bk = bucket(s.occ.count_ones(), m.b);
    let w2o = m.w2() + bk * 2 * h;
    let b2o = m.b2() + bk;
    let w2 = &m.p[w2o..w2o + 2 * h];
    let mut out = m.p[b2o];
    for j in 0..h {
        let cu = a_us[j].clamp(0.0, 1.0);
        let ct = a_them[j].clamp(0.0, 1.0);
        out += cu * cu * w2[j] + ct * ct * w2[h + j];
    }
    let pred = sigmoid(out);
    let err = pred - target;
    let loss = err * err;
    if let Some(g) = grad {
        let dout = 2.0 * err * pred * (1.0 - pred);
        let b1 = m.b1();
        g[b2o] += dout;
        // Градієнт за прихованим шаром – у тих самих буферах acc (вони вже не потрібні).
        for j in 0..h {
            let (vu, vt) = (a_us[j], a_them[j]);
            let cu = vu.clamp(0.0, 1.0);
            let ct = vt.clamp(0.0, 1.0);
            g[w2o + j] += dout * cu * cu;
            g[w2o + h + j] += dout * ct * ct;
            a_us[j] = if vu > 0.0 && vu < 1.0 {
                dout * w2[j] * 2.0 * vu
            } else {
                0.0
            };
            a_them[j] = if vt > 0.0 && vt < 1.0 {
                dout * w2[h + j] * 2.0 * vt
            } else {
                0.0
            };
            g[b1 + j] += a_us[j] + a_them[j];
        }
        for i in 0..n {
            let gu = us[i] as usize * h;
            for j in 0..h {
                g[gu + j] += a_us[j];
            }
            let gt = them[i] as usize * h;
            for j in 0..h {
                g[gt + j] += a_them[j];
            }
        }
    }
    loss
}

/// Додає до out зразки з файлу: пропустивши skip записів і не більше take.
fn read_into(path: &str, skip: usize, take: usize, out: &mut Vec<Sample>) {
    let mut buf = Vec::new();
    std::fs::File::open(path)
        .and_then(|f| {
            f.take(((skip + take) as u64).saturating_mul(RECORD as u64))
                .read_to_end(&mut buf)
        })
        .unwrap_or_else(|e| panic!("{path}: {e}"));
    out.extend(
        buf.chunks_exact(RECORD)
            .skip(skip)
            .take(take)
            .map(Sample::from_bytes),
    );
}

/// xorshift64: стан – одне число, тож його легко зберегти й відновити.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

const STATE_MAGIC: &[u8; 8] = b"SIRKOST1";

/// Збережене навчання: де зупинилися й усе, що потрібно, щоб продовжити так, ніби перерви не було.
struct Saved {
    epochs: usize,
    done: usize,
    t: i32,
    step: usize,
    total_steps: usize,
    seed: u64,
}

fn save_state(path: &str, tr: &Trainer, s: &Saved) {
    let m = &tr.model;
    let mut out = Vec::with_capacity(64 + 12 * m.p.len());
    out.extend_from_slice(STATE_MAGIC);
    for v in [
        m.h as u64,
        m.b as u64,
        s.epochs as u64,
        s.done as u64,
        s.t as u64,
        s.step as u64,
        s.total_steps as u64,
        s.seed,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in [&m.p, &tr.m1, &tr.m2] {
        for x in v.iter() {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    // Спершу тимчасовий файл: обрив посеред запису не зіпсує попередній стан.
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, &out).unwrap_or_else(|e| panic!("{tmp}: {e}"));
    std::fs::rename(&tmp, path).unwrap_or_else(|e| panic!("{path}: {e}"));
}

/// Читає стан у tr; None – файлу немає. Стан з іншими розмірами мережі чи числом епох – помилка.
fn load_state(path: &str, tr: &mut Trainer, epochs: usize) -> Option<Saved> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => panic!("{path}: {e}"),
    };
    let n = tr.model.p.len();
    assert!(
        bytes.len() == 8 + 8 * 8 + 12 * n && &bytes[..8] == STATE_MAGIC,
        "{path}: не стан навчання цієї мережі"
    );
    let u = |i: usize| u64::from_le_bytes(bytes[8 + 8 * i..16 + 8 * i].try_into().unwrap());
    assert!(
        u(0) as usize == tr.model.h && u(1) as usize == tr.model.b && u(2) as usize == epochs,
        "{path}: стан для мережі {}×{} на {} епох, а задано {}×{} на {epochs}",
        u(0),
        u(1),
        u(2),
        tr.model.h,
        tr.model.b
    );
    let floats = &bytes[8 + 8 * 8..];
    for (k, v) in [&mut tr.model.p, &mut tr.m1, &mut tr.m2]
        .into_iter()
        .enumerate()
    {
        for (j, x) in v.iter_mut().enumerate() {
            let o = 4 * (k * n + j);
            *x = f32::from_le_bytes(floats[o..o + 4].try_into().unwrap());
        }
    }
    Some(Saved {
        epochs,
        done: u(3) as usize,
        t: u(4) as i32,
        step: u(5) as usize,
        total_steps: u(6) as usize,
        seed: u(7),
    })
}

/// Стан навчання: модель, моменти Adam і буфери градієнтів потоків.
struct Trainer {
    model: Model,
    m1: Vec<f32>,
    m2: Vec<f32>,
    grads: Vec<Vec<f32>>,
    t: i32,
    step: usize,
    total_steps: usize,
}

impl Trainer {
    /// Один прохід по мінівибірках шматка даних; повертає суму втрат.
    fn train_on(&mut self, data: &[Sample], o: &Options) -> f64 {
        let (beta1, beta2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let threads = self.grads.len();
        let n = self.m1.len();
        let mut total = 0.0f64;
        for batch in data.chunks(o.batch) {
            let chunk = batch.len().div_ceil(threads);
            let model_ref = &self.model;
            let losses: Vec<f64> = std::thread::scope(|scope| {
                let handles: Vec<_> = self
                    .grads
                    .iter_mut()
                    .zip(batch.chunks(chunk.max(1)))
                    .map(|(g, part)| {
                        scope.spawn(move || {
                            g.iter_mut().for_each(|x| *x = 0.0);
                            let mut acc = vec![0.0f32; 2 * model_ref.h];
                            part.iter()
                                .map(|s| step(model_ref, s, o.wdl, Some(g), &mut acc) as f64)
                                .sum::<f64>()
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("потік навчання"))
                    .collect()
            });
            total += losses.iter().sum::<f64>();
            let used = batch.len().div_ceil(chunk.max(1));
            // Adam з косинусним спадом кроку.
            self.t += 1;
            let progress = (self.step as f32 / self.total_steps.max(1) as f32).min(1.0);
            self.step += 1;
            let lr = o.lr * (0.05 + 0.95 * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()));
            let scale = 1.0 / batch.len() as f32;
            let c1 = 1.0 - beta1.powi(self.t);
            let c2 = 1.0 - beta2.powi(self.t);
            let b2 = self.model.b2();
            for k in 0..n {
                let mut g = 0.0f32;
                for gr in self.grads.iter().take(used) {
                    g += gr[k];
                }
                g *= scale;
                self.m1[k] = beta1 * self.m1[k] + (1.0 - beta1) * g;
                self.m2[k] = beta2 * self.m2[k] + (1.0 - beta2) * g * g;
                let upd = lr * (self.m1[k] / c1) / ((self.m2[k] / c2).sqrt() + eps);
                let x = self.model.p[k] - upd;
                self.model.p[k] = if k < b2 { x.clamp(-CLIP, CLIP) } else { x };
            }
        }
        total
    }
}

pub fn run(o: Options) {
    let start = Instant::now();
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    // Розміри файлів; на перевірку – по кілька перших записів з кожного, у навчання вони не йдуть.
    let files: Vec<(String, usize)> = o
        .data
        .iter()
        .map(|p| {
            let len = std::fs::metadata(p)
                .unwrap_or_else(|e| panic!("{p}: {e}"))
                .len();
            (p.clone(), (len / RECORD as u64) as usize)
        })
        .filter(|(_, n)| *n > 0)
        .collect();
    assert!(!files.is_empty(), "немає даних для навчання");
    let total: usize = files.iter().map(|(_, n)| n).sum();
    let per_file = (total / 100)
        .clamp(files.len(), 200_000)
        .div_ceil(files.len());
    let skip = |n: usize| per_file.min(n / 2);
    let mut valid = Vec::new();
    for (path, n) in &files {
        read_into(path, 0, skip(*n), &mut valid);
    }
    let train_n: usize = files.iter().map(|(_, n)| n - skip(*n)).sum();
    // Шматок у пам'яті: файли в довільному порядку, усередині шматка – перемішані зразки.
    let buffer = o.buffer.max(o.batch);
    eprintln!(
        "зразків: навчання {train_n}, перевірка {}, файлів {}, шматок до {buffer}",
        valid.len(),
        files.len()
    );

    let h = o.hidden;
    let b = o.buckets.max(1);
    let n = Model::len(h, b);
    let threads = o.threads.max(1);
    let mut tr = Trainer {
        model: Model::new(h, b, 0x1234_5678),
        m1: vec![0.0f32; n],
        m2: vec![0.0f32; n],
        grads: (0..threads).map(|_| vec![0.0f32; n]).collect(),
        t: 0,
        step: 0,
        // Неповні мінівибірки в кінці шматків дають кілька зайвих кроків: прогрес обмежено одиницею.
        total_steps: train_n.div_ceil(o.batch) * o.epochs,
    };
    let mut first = 0;
    if let Some(path) = &o.state {
        if let Some(sv) = load_state(path, &mut tr, o.epochs) {
            assert_eq!(
                sv.total_steps, tr.total_steps,
                "{path}: стан для інших даних"
            );
            tr.t = sv.t;
            tr.step = sv.step;
            rng = Rng(sv.seed);
            first = sv.done;
            if first < o.epochs {
                eprintln!("продовжую з епохи {}/{}", first + 1, o.epochs);
            } else {
                // Навчання вже скінчилося: мережа – зі стану, щоб файл був і в цьому запуску.
                std::fs::write(&o.out, tr.model.quantize().to_bytes()).expect("запис мережі");
            }
        }
    }
    let mut order: Vec<usize> = (0..files.len()).collect();
    let mut epoch_secs = 0.0f64;

    for epoch in first..o.epochs {
        if let Some(limit) = o.max_minutes {
            let elapsed = start.elapsed().as_secs_f64();
            if epoch > first && elapsed + epoch_secs * 1.05 > limit * 60.0 {
                eprintln!(
                    "час вичерпано: зроблено {epoch} з {} епох, епоха триває {:.0} хв",
                    o.epochs,
                    epoch_secs / 60.0
                );
                return;
            }
        }
        let epoch_start = Instant::now();
        // Порядок файлів у кожній епохі – з поточного стану генератора, тож продовження його повторює.
        order.sort_unstable();
        shuffle(&mut order, &mut rng);
        // Шматок може перевищити межу на один файл: місце під це – одразу, без подвоєння вектора.
        let biggest = files.iter().map(|(_, n)| *n).max().unwrap_or(0);
        let mut buf: Vec<Sample> = Vec::with_capacity(buffer.min(train_n) + biggest);
        let mut train_loss = 0.0f64;
        for (k, &i) in order.iter().enumerate() {
            let (path, n) = &files[i];
            read_into(path, skip(*n), *n, &mut buf);
            if buf.len() >= buffer || k + 1 == order.len() {
                shuffle(&mut buf, &mut rng);
                train_loss += tr.train_on(&buf, &o);
                buf.clear();
            }
        }
        let mut acc = vec![0.0f32; 2 * tr.model.h];
        let vloss: f64 = valid
            .iter()
            .map(|s| step(&tr.model, s, o.wdl, None, &mut acc) as f64)
            .sum::<f64>()
            / valid.len() as f64;
        eprintln!(
            "епоха {}/{}: втрата навчання {:.6}, перевірки {:.6}, {:.0} с",
            epoch + 1,
            o.epochs,
            train_loss / train_n as f64,
            vloss,
            start.elapsed().as_secs_f64()
        );
        std::fs::write(&o.out, tr.model.quantize().to_bytes()).expect("запис мережі");
        if let Some(path) = &o.state {
            let sv = Saved {
                epochs: o.epochs,
                done: epoch + 1,
                t: tr.t,
                step: tr.step,
                total_steps: tr.total_steps,
                seed: rng.0,
            };
            save_state(path, &tr, &sv);
        }
        epoch_secs = epoch_start.elapsed().as_secs_f64();
    }
    eprintln!("навчання завершено");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    /// Квантована мережа рахує те саме, що й дробова (з точністю до округлення).
    #[test]
    fn quantized_matches_float() {
        crate::bitboard::init();
        let model = Model::new(32, 8, 99);
        let net = model.quantize();
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
        ] {
            let p = Position::from_fen(fen).unwrap();
            let s = Sample::from_position(&p, 0);
            let (us, them, n, _) = prepare(&s, 0.0);
            let h = model.h;
            let mut au = model.p[model.b1()..model.w2()].to_vec();
            let mut at = au.clone();
            for i in 0..n {
                for j in 0..h {
                    au[j] += model.p[us[i] as usize * h + j];
                    at[j] += model.p[them[i] as usize * h + j];
                }
            }
            let bk = bucket(p.occupied().count_ones(), model.b);
            let w2 = &model.p[model.w2() + bk * 2 * h..model.w2() + (bk + 1) * 2 * h];
            let mut out = model.p[model.b2() + bk];
            for j in 0..h {
                out += au[j].clamp(0.0, 1.0).powi(2) * w2[j]
                    + at[j].clamp(0.0, 1.0).powi(2) * w2[h + j];
            }
            let float_cp = out * 400.0;
            let int_cp = net.evaluate(&p) as f32;
            assert!(
                (float_cp - int_cp).abs() < 6.0,
                "{fen}: {float_cp} проти {int_cp}"
            );
        }
    }
}
