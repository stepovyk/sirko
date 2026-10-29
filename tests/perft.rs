//! Генерація ходів проти відомих чисел perft (класика й шахи Фішера).

use sirko::position::{perft, Position};

fn check(fen: &str, expected: &[u64]) {
    sirko::bitboard::init();
    let p = Position::from_fen(fen).unwrap();
    // Поле на проході лишається лише для легального взяття, тож звіряються перші три поля.
    assert_eq!(
        p.fen().split_whitespace().take(3).collect::<Vec<_>>(),
        fen.split_whitespace().take(3).collect::<Vec<_>>(),
        "FEN туди й назад"
    );
    for (i, &n) in expected.iter().enumerate() {
        assert_eq!(perft(&p, i as u32 + 1), n, "{fen} глибина {}", i + 1);
    }
}

#[test]
fn startpos() {
    check(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &[20, 400, 8902, 197281, 4865609],
    );
}

#[test]
fn kiwipete() {
    check(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        &[48, 2039, 97862, 4085603],
    );
}

#[test]
fn position3() {
    check(
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        &[14, 191, 2812, 43238, 674624],
    );
}

#[test]
fn position4() {
    check(
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        &[6, 264, 9467, 422333],
    );
    check(
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        &[6, 264, 9467, 422333],
    );
}

#[test]
fn position5() {
    check(
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        &[44, 1486, 62379, 2103487],
    );
}

#[test]
fn position6() {
    check(
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        &[46, 2079, 89890, 3894594],
    );
}

#[test]
fn en_passant_pins_and_checks() {
    // Взяття на проході, що відкриває шах по горизонталі, – нелегальне (числа – з python-chess).
    check("8/8/8/KPp4r/8/8/8/7k w - c6 0 2", &[4, 56, 259, 4225]);
    check("8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1", &[6, 136, 863, 20471]);
    check("8/8/3k4/8/1pP5/8/8/4K2R b - c3 0 1", &[9, 132, 985, 18002]);
}

#[test]
fn chess960() {
    // Відомі позиції Фішера з таблиць perft (X-FEN, права за файлами тур).
    check(
        "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
        &[21, 528, 12189, 326672, 8146062],
    );
    check(
        "2nnrbkr/p1qppppp/8/1ppb4/6PP/3PP3/PPP2P2/BQNNRBKR w HEhe - 1 9",
        &[21, 807, 18002, 667366],
    );
    check(
        "b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9",
        &[20, 479, 10471, 273318],
    );
    check(
        "qbbnnrkr/2pp2pp/p7/1p2pp2/8/P3PP2/1PPP1KPP/QBBNNR1R w hf - 0 9",
        &[22, 593, 13440, 382958],
    );
    check(
        "1nbbnrkr/p1p1ppp1/3p4/1p3P1p/3Pq2P/8/PPP1P1P1/QNBBNRKR w HFhf - 0 9",
        &[28, 1120, 31058, 1171749],
    );
}
