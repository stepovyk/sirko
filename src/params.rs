//! Числа пошуку, які можна підбирати (SPSA). Звичайно вони сталі; з SIRKO_TUNE=1 рушій показує
//! їх як опції UCI, і їх можна міняти без перезбирання.

use std::sync::atomic::{AtomicI32, Ordering};

pub struct Param {
    pub name: &'static str,
    pub default: i32,
    pub min: i32,
    pub max: i32,
    value: AtomicI32,
}

macro_rules! params {
    ($($id:ident = $name:literal, $def:expr, $min:expr, $max:expr;)*) => {
        #[allow(non_camel_case_types, clippy::upper_case_acronyms)]
        #[derive(Clone, Copy)]
        #[repr(usize)]
        pub enum P { $($id),* }
        const COUNT: usize = [$($name),*].len();
        pub static PARAMS: [Param; COUNT] = [$(Param { name: $name, default: $def, min: $min, max: $max, value: AtomicI32::new($def) }),*];
    };
}

params! {
    RfpMargin = "rfp_margin", 83, 40, 160;
    RfpDepth = "rfp_depth", 5, 4, 12;
    NmpBase = "nmp_base", 4, 2, 5;
    NmpEvalDiv = "nmp_eval_div", 198, 100, 400;
    LmpBase = "lmp_base", 5, 1, 8;
    FutBase = "fut_base", 89, 30, 200;
    FutMult = "fut_mult", 111, 50, 200;
    FutDepth = "fut_depth", 7, 4, 10;
    SeeQuiet = "see_quiet", 58, 20, 120;
    SeeCapture = "see_capture", 98, 40, 200;
    LmrBase = "lmr_base", 80, 30, 150;      // ×0,01
    LmrDiv = "lmr_div", 227, 150, 350;      // ×0,01
    LmrHistDiv = "lmr_hist_div", 7589, 3000, 16000;
    SeDepth = "se_depth", 9, 5, 10;
    SeMargin = "se_margin", 3, 1, 4;
    AspDelta = "asp_delta", 17, 8, 40;
    QsDelta = "qs_delta", 224, 80, 400;
    HistMult = "hist_mult", 16, 6, 32;
    HistMax = "hist_max", 1586, 600, 3000;
}

#[inline(always)]
pub fn get(p: P) -> i32 {
    PARAMS[p as usize].value.load(Ordering::Relaxed)
}

/// Встановлює параметр за назвою (у межах). false – немає такого.
pub fn set(name: &str, v: i32) -> bool {
    match PARAMS.iter().find(|p| p.name.eq_ignore_ascii_case(name)) {
        Some(p) => {
            p.value.store(v.clamp(p.min, p.max), Ordering::Relaxed);
            true
        }
        None => false,
    }
}

pub fn tuning_enabled() -> bool {
    std::env::var("SIRKO_TUNE")
        .map(|v| v == "1")
        .unwrap_or(false)
}
