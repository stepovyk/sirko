// Вбудовує мережу nets/sirko.nnue, якщо вона є; без неї рушій грає з ручною оцінкою.
use std::path::Path;

fn main() {
    let out = Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR")).join("sirko.nnue");
    let src = Path::new("nets/sirko.nnue");
    println!("cargo:rerun-if-changed=nets/sirko.nnue");
    if src.exists() {
        std::fs::copy(src, &out).expect("копія мережі");
    } else {
        std::fs::write(&out, []).expect("порожня мережа");
    }
}
