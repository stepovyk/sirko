use sirko::position::{perft, Position, START_FEN};
use sirko::search::{think, Limits, Shared};
use std::sync::Arc;

/// Позиції для заміру: різні стадії партії, тактика й закінчення.
const BENCH: [&str; 12] = [
    START_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "2r3k1/pp3ppp/2n1b3/q2p4/3P4/P1N1PN2/1P3PPP/R2Q1RK1 b - - 0 15",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1",
    "r1bq1rk1/pp2bppp/2n2n2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 9",
    "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1",
    "3r1rk1/1p3pnp/p3pBp1/1qPpP3/1P1P2R1/P2Q3R/6PP/6K1 w - - 0 1",
];

fn main() {
    sirko::bitboard::init();
    sirko::eval::init();
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("perft") => {
            let depth: u32 = args.get(2).and_then(|d| d.parse().ok()).unwrap_or(5);
            let fen = if args.len() > 3 {
                args[3..].join(" ")
            } else {
                START_FEN.to_string()
            };
            let p = Position::from_fen(&fen).expect("FEN");
            let t = std::time::Instant::now();
            let n = perft(&p, depth);
            let s = t.elapsed().as_secs_f64();
            println!("{n} вузлів за {s:.2} с ({:.1} млн/с)", n as f64 / s / 1e6);
        }
        Some("bench") => {
            // Один потік і однакова таблиця: число вузлів – відбиток поведінки пошуку.
            let depth: i32 = args.get(2).and_then(|d| d.parse().ok()).unwrap_or(13);
            let mut nodes = 0;
            let t = std::time::Instant::now();
            for fen in BENCH {
                let p = Position::from_fen(fen).expect("FEN");
                let shared = Arc::new(Shared::new(16, 1).with_net(sirko::nnue::embedded()));
                let limits = Limits {
                    depth: Some(depth),
                    ..Limits::default()
                };
                think(&p, &[p.key], &limits, Arc::clone(&shared), 1, 0, false);
                nodes += shared.total_nodes();
            }
            let s = t.elapsed().as_secs_f64();
            println!(
                "{nodes} вузлів за {s:.2} с ({:.0} тис./с)",
                nodes as f64 / s / 1e3
            );
        }
        Some("datagen") => {
            // datagen ПАРТІЙ ВУЗЛІВ_НА_ХІД ПОТОКІВ ФАЙЛ [ХВИЛИН]
            let num = |i: usize, d: u64| args.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
            sirko::datagen::run(sirko::datagen::Options {
                games: num(2, 1000),
                nodes: num(3, 5000),
                threads: num(4, 1) as usize,
                out: args.get(5).cloned().unwrap_or_else(|| "data.bin".into()),
                random_plies: 8,
                seconds: num(6, 0) * 60,
            });
        }
        Some("train") => {
            // train ВИХІД ПРИХОВАНИХ ЕПОХ [--batch=N --lr=X --wdl=X] ДАНІ...
            let num = |i: usize, d: usize| args.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
            let flag = |name: &str, d: f64| {
                args.iter()
                    .find_map(|a| {
                        a.strip_prefix(&format!("--{name}="))
                            .and_then(|v| v.parse().ok())
                    })
                    .unwrap_or(d)
            };
            sirko::train::run(sirko::train::Options {
                out: args.get(2).cloned().unwrap_or_else(|| "sirko.nnue".into()),
                hidden: num(3, 256),
                epochs: num(4, 10),
                data: args
                    .iter()
                    .skip(5)
                    .filter(|a| !a.starts_with("--"))
                    .cloned()
                    .collect(),
                batch: flag("batch", 16384.0) as usize,
                lr: flag("lr", 0.001) as f32,
                wdl: flag("wdl", 0.4) as f32,
                threads: std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1),
            });
        }
        Some("moves") => {
            // Для звірки з python-chess: на кожен FEN зі stdin – «хід FEN-після» для кожного легального ходу.
            use std::io::BufRead;
            for line in std::io::stdin().lock().lines() {
                let line = line.expect("stdin");
                match Position::from_fen(line.trim()) {
                    Ok(p) => {
                        let mut out: Vec<String> = p
                            .legal_moves()
                            .iter()
                            .map(|&m| format!("{} {}", p.move_to_uci(m), p.make(m).fen()))
                            .collect();
                        out.sort();
                        println!("{}", out.join("|"));
                    }
                    Err(e) => println!("error {e}"),
                }
            }
        }
        _ => sirko::uci::Engine::new().run(),
    }
}
