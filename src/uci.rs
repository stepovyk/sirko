//! Протокол UCI: так із рушієм говорять lichess-bot, python-chess і будь-яка шахова програма.

use crate::nnue::{self, Network};
use crate::position::{Position, START_FEN};
use crate::search::{think, Limits, Shared};
use crate::types::*;
use std::io::{self, BufRead, Write};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::JoinHandle;

pub const NAME: &str = concat!("Sirko ", env!("CARGO_PKG_VERSION"));
const DEFAULT_HASH: usize = 64;
const MAX_HASH: usize = 65536;
const MAX_THREADS: usize = 256;

pub struct Engine {
    pos: Position,
    keys: Vec<u64>,
    shared: Arc<Shared>,
    hash: usize,
    threads: usize,
    overhead: u64,
    chess960: bool,
    /// Остання команда position була зрозуміла (інакше go відповідає «bestmove 0000», а не ходом зі старої позиції).
    valid: bool,
    infinite: bool,
    net: Option<&'static Network>,
    search: Option<JoinHandle<()>>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Engine {
        let pos = Position::startpos();
        let net = nnue::embedded();
        Engine {
            keys: vec![pos.key],
            pos,
            shared: Arc::new(Shared::new(DEFAULT_HASH, 1).with_net(net)),
            net,
            hash: DEFAULT_HASH,
            threads: 1,
            overhead: 30,
            chess960: false,
            valid: true,
            infinite: false,
            search: None,
        }
    }

    fn wait(&mut self) {
        if let Some(h) = self.search.take() {
            let _ = h.join();
        }
    }

    fn stop(&mut self) {
        self.shared.pondering.store(false, Ordering::Relaxed);
        self.shared.stop.store(true, Ordering::Relaxed);
        self.wait();
        self.shared.stop.store(false, Ordering::Relaxed);
    }

    fn resize(&mut self) {
        self.stop();
        self.shared = Arc::new(Shared::new(self.hash, self.threads).with_net(self.net));
    }

    fn set_position(&mut self, words: &[&str]) {
        let mut i = 0;
        let pos = if words.first() == Some(&"startpos") {
            i = 1;
            Position::from_fen(START_FEN)
        } else if words.first() == Some(&"fen") {
            let end = words
                .iter()
                .position(|&w| w == "moves")
                .unwrap_or(words.len());
            i = end;
            Position::from_fen(&words[1..end].join(" "))
        } else {
            Err("невідома позиція".to_string())
        };
        let mut p = match pos {
            Ok(p) => p,
            Err(e) => {
                println!("info string помилка позиції: {e}");
                self.valid = false;
                return;
            }
        };
        p.chess960 |= self.chess960;
        let mut keys = vec![p.key];
        self.valid = true;
        if words.get(i) == Some(&"moves") {
            for text in &words[i + 1..] {
                match p.parse_uci(text) {
                    Some(m) => {
                        p = p.make(m);
                        keys.push(p.key);
                    }
                    None => {
                        println!("info string нелегальний хід {text}");
                        self.valid = false;
                        break;
                    }
                }
            }
        }
        self.pos = p;
        self.keys = keys;
    }

    fn go(&mut self, words: &[&str]) {
        self.stop();
        let mut l = Limits::default();
        let mut it = words.iter();
        while let Some(&w) = it.next() {
            let mut num = || {
                it.next()
                    .and_then(|v| v.parse::<i64>().ok())
                    .unwrap_or(0)
                    .max(0) as u64
            };
            match w {
                "wtime" => l.time[WHITE] = Some(num()),
                "btime" => l.time[BLACK] = Some(num()),
                "winc" => l.inc[WHITE] = num(),
                "binc" => l.inc[BLACK] = num(),
                "movestogo" => l.movestogo = Some(num()),
                "movetime" => l.movetime = Some(num()),
                "depth" => l.depth = Some(num() as i32),
                "nodes" => l.nodes = Some(num()),
                "mate" => l.mate = Some(num() as i32),
                "infinite" => l.infinite = true,
                "ponder" => l.ponder = true,
                _ => {}
            }
        }
        let shared = Arc::clone(&self.shared);
        shared.pondering.store(l.ponder, Ordering::Relaxed);
        self.infinite = l.infinite;
        let valid = self.valid;
        let pos = self.pos;
        let keys = self.keys.clone();
        let threads = self.threads;
        let overhead = self.overhead;
        self.search = Some(std::thread::spawn(move || {
            let r = if valid {
                think(
                    &pos,
                    &keys,
                    &l,
                    Arc::clone(&shared),
                    threads,
                    overhead,
                    true,
                )
            } else {
                crate::search::Report {
                    best: Move::NONE,
                    ponder: Move::NONE,
                    score: 0,
                }
            };
            // На час суперника (і в нескінченному пошуку) відповідь іде лише після stop чи ponderhit,
            // навіть якщо думати вже нема про що.
            while (l.infinite || shared.pondering.load(Ordering::Relaxed))
                && !shared.stop.load(Ordering::Relaxed)
            {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            if r.best.is_none() {
                println!("bestmove 0000");
            } else {
                let after = pos.make(r.best);
                let ponder_ok =
                    !r.ponder.is_none() && after.legal_moves().iter().any(|&m| m == r.ponder);
                if ponder_ok {
                    println!(
                        "bestmove {} ponder {}",
                        pos.move_to_uci(r.best),
                        after.move_to_uci(r.ponder)
                    );
                } else {
                    println!("bestmove {}", pos.move_to_uci(r.best));
                }
            }
            let _ = io::stdout().flush();
        }));
    }

    fn setoption(&mut self, line: &str) {
        let body = line.trim_start_matches("setoption").trim();
        let body = body.strip_prefix("name").unwrap_or(body).trim();
        let (name, value) = match body.find(" value ") {
            Some(i) => (body[..i].trim(), body[i + 7..].trim()),
            None => (body, ""),
        };
        match name.to_ascii_lowercase().as_str() {
            "hash" => {
                if let Ok(v) = value.parse::<usize>() {
                    self.hash = v.clamp(1, MAX_HASH);
                    self.resize();
                }
            }
            "threads" => {
                if let Ok(v) = value.parse::<usize>() {
                    self.threads = v.clamp(1, MAX_THREADS);
                    self.resize();
                }
            }
            "move overhead" => {
                if let Ok(v) = value.parse::<u64>() {
                    self.overhead = v.min(5000);
                }
            }
            "uci_chess960" => {
                self.chess960 = value.eq_ignore_ascii_case("true");
                self.pos.chess960 |= self.chess960;
            }
            "evalfile" => {
                // Інша мережа без перезбирання (для перевірки нових мереж); «<internal>» – вбудована.
                self.net = if value.is_empty() || value == "<internal>" {
                    nnue::embedded()
                } else if value == "<hce>" {
                    None
                } else {
                    match std::fs::read(value)
                        .map_err(|e| e.to_string())
                        .and_then(|b| Network::from_bytes(&b))
                    {
                        Ok(n) => Some(Box::leak(Box::new(n))),
                        Err(e) => {
                            println!("info string мережу {value} не завантажено: {e}");
                            self.net
                        }
                    }
                };
                self.resize();
            }
            "clear hash" => {
                self.stop();
                self.shared.tt.clear();
            }
            other => {
                // Параметри пошуку (для підбору); Ponder та інші – приймаємо мовчки.
                if let Ok(v) = value.parse::<i32>() {
                    crate::params::set(other, v);
                }
            }
        }
    }

    pub fn command(&mut self, line: &str) -> bool {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.first().copied() {
            Some("uci") => {
                println!("id name {NAME}");
                println!("id author Stepovyk");
                println!("option name Hash type spin default {DEFAULT_HASH} min 1 max {MAX_HASH}");
                println!("option name Threads type spin default 1 min 1 max {MAX_THREADS}");
                println!("option name Move Overhead type spin default 30 min 0 max 5000");
                println!("option name Ponder type check default false");
                println!("option name UCI_Chess960 type check default false");
                println!("option name Clear Hash type button");
                println!("option name EvalFile type string default <internal>");
                if crate::params::tuning_enabled() {
                    for p in crate::params::PARAMS.iter() {
                        println!(
                            "option name {} type spin default {} min {} max {}",
                            p.name, p.default, p.min, p.max
                        );
                    }
                }
                println!("uciok");
            }
            Some("isready") => println!("readyok"),
            Some("ucinewgame") => {
                self.stop();
                self.shared.tt.clear();
                self.shared.clear_heuristics();
            }
            Some("setoption") => self.setoption(line),
            Some("position") => {
                self.stop();
                self.set_position(&words[1..]);
            }
            Some("go") => self.go(&words[1..]),
            Some("stop") => self.stop(),
            Some("ponderhit") => self.shared.ponderhit(),
            Some("quit") => {
                self.stop();
                return false;
            }
            Some("d") => println!("{}", self.pos.fen()),
            _ => {}
        }
        let _ = io::stdout().flush();
        true
    }

    pub fn run(&mut self) {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            if !self.command(line.trim()) {
                return;
            }
        }
        // Кінець вводу: скінченний пошук доводимо до відповіді, нескінченний і на час суперника – зупиняємо.
        if self.infinite || self.shared.pondering.load(Ordering::Relaxed) {
            self.stop();
        } else {
            self.wait();
        }
    }
}
