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
    RfpMargin = "rfp_margin", 85, 40, 160;
    RfpDepth = "rfp_depth", 8, 4, 12;
    NmpBase = "nmp_base", 3, 2, 5;
    NmpEvalDiv = "nmp_eval_div", 200, 100, 400;
    LmpBase = "lmp_base", 3, 1, 8;
    FutBase = "fut_base", 90, 30, 200;
    FutMult = "fut_mult", 110, 50, 200;
    FutDepth = "fut_depth", 7, 4, 10;
    SeeQuiet = "see_quiet", 60, 20, 120;
    SeeCapture = "see_capture", 100, 40, 200;
    LmrBase = "lmr_base", 80, 30, 150;      // ×0,01
    LmrDiv = "lmr_div", 230, 150, 350;      // ×0,01
    LmrHistDiv = "lmr_hist_div", 8000, 3000, 16000;
    SeDepth = "se_depth", 7, 5, 10;
    SeMargin = "se_margin", 2, 1, 4;
    AspDelta = "asp_delta", 18, 8, 40;
    QsDelta = "qs_delta", 200, 80, 400;
    HistMult = "hist_mult", 16, 6, 32;
    HistMax = "hist_max", 1600, 600, 3000;
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
