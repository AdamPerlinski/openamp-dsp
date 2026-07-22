#![allow(static_mut_refs)] // single-threaded inside one AudioWorklet — no aliasing possible

// OpenAmp DSP core — wasm32-unknown-unknown, no JS glue, C ABI only.
// Lives inside an AudioWorklet: single-threaded, static buffers, zero
// allocation after init. The worklet copies input into IN_BUF, calls
// process(n), reads OUT_BUF. Blocks are ≤128 frames (one render quantum).
//
// Chain (docs/TONE.md): 20 Hz DC HPF → voicing tightness HPF → noise gate
// (pre-drive, hysteresis + hold) → pre-gain → up to 3 asymmetric tanh stages
// with interstage HPF/LPF → anti-fizz LPF → 3-band tonestack + fixed voicing
// EQ → power tanh → presence shelf → 10 Hz DC blocker → master.
// Every biased stage uses tanh(k·x+b) − tanh(b) so no DC leaks downstream.
//
// Params: 0 drive 0..1 · 1 bass dB · 2 mid dB · 3 treble dB · 4 master 0..1.5
//         5 voicing 0..3 · 6 gate on/off · 7 gate threshold dB · 8 presence dB

const MAX_BLOCK: usize = 128;
const PI: f32 = core::f32::consts::PI;
const OUT_SCALE: f32 = 0.4; // headroom: worklet peaks stay well under 0 dBFS

static mut IN_BUF: [f32; MAX_BLOCK] = [0.0; MAX_BLOCK];
static mut OUT_BUF: [f32; MAX_BLOCK] = [0.0; MAX_BLOCK];
static mut AMP: Option<Amp> = None;

#[derive(Clone, Copy, Default)]
struct Biquad {
    b0: f32, b1: f32, b2: f32, a1: f32, a2: f32,
    z1: f32, z2: f32,
}

impl Biquad {
    #[inline(always)]
    fn tick(&mut self, x: f32) -> f32 {
        // transposed direct form II
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    fn set(&mut self, b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) {
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }

    fn bypass(&mut self) {
        self.set(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    }

    fn low_shelf(&mut self, fs: f32, f0: f32, db: f32) {
        let a = 10f32.powf(db / 40.0);
        let w0 = 2.0 * PI * f0 / fs;
        let (sw, cw) = (w0.sin(), w0.cos());
        let alpha = sw / 2.0 * 2f32.sqrt();
        let ta = 2.0 * a.sqrt() * alpha;
        self.set(
            a * ((a + 1.0) - (a - 1.0) * cw + ta),
            2.0 * a * ((a - 1.0) - (a + 1.0) * cw),
            a * ((a + 1.0) - (a - 1.0) * cw - ta),
            (a + 1.0) + (a - 1.0) * cw + ta,
            -2.0 * ((a - 1.0) + (a + 1.0) * cw),
            (a + 1.0) + (a - 1.0) * cw - ta,
        );
    }

    fn high_shelf(&mut self, fs: f32, f0: f32, db: f32) {
        let a = 10f32.powf(db / 40.0);
        let w0 = 2.0 * PI * f0 / fs;
        let (sw, cw) = (w0.sin(), w0.cos());
        let alpha = sw / 2.0 * 2f32.sqrt();
        let ta = 2.0 * a.sqrt() * alpha;
        self.set(
            a * ((a + 1.0) + (a - 1.0) * cw + ta),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * cw),
            a * ((a + 1.0) + (a - 1.0) * cw - ta),
            (a + 1.0) - (a - 1.0) * cw + ta,
            2.0 * ((a - 1.0) - (a + 1.0) * cw),
            (a + 1.0) - (a - 1.0) * cw - ta,
        );
    }

    fn peaking(&mut self, fs: f32, f0: f32, q: f32, db: f32) {
        let a = 10f32.powf(db / 40.0);
        let w0 = 2.0 * PI * f0 / fs;
        let (sw, cw) = (w0.sin(), w0.cos());
        let alpha = sw / (2.0 * q);
        self.set(
            1.0 + alpha * a,
            -2.0 * cw,
            1.0 - alpha * a,
            1.0 + alpha / a,
            -2.0 * cw,
            1.0 - alpha / a,
        );
    }
}

#[derive(Clone, Copy, Default)]
struct OnePoleHp {
    a: f32,
    x1: f32,
    y1: f32,
}

impl OnePoleHp {
    fn set(&mut self, fs: f32, fc: f32) {
        self.a = (-2.0 * PI * fc / fs).exp();
    }
    #[inline(always)]
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.a * (self.y1 + x - self.x1);
        self.x1 = x;
        self.y1 = y;
        y
    }
}

#[derive(Clone, Copy, Default)]
struct OnePoleLp {
    a: f32,
    y1: f32,
}

impl OnePoleLp {
    fn set(&mut self, fs: f32, fc: f32) {
        self.a = (-2.0 * PI * fc / fs).exp();
    }
    #[inline(always)]
    fn tick(&mut self, x: f32) -> f32 {
        self.y1 = (1.0 - self.a) * x + self.a * self.y1;
        self.y1
    }
}

// One tanh stage description: y = (tanh(k·x + b) − tanh(b)) · post
#[derive(Clone, Copy)]
struct Stage {
    k: f32,
    bias: f32,
    post: f32,
}

// First-order ADAA tanh (Parker et al., DAFx-16): antiderivative differences
// suppress aliasing another octave beyond what 2x oversampling buys.
// f64 internals: ln·cosh differences at small Δu cancel catastrophically in f32.
#[derive(Clone, Copy)]
struct AdaaTanh {
    k: f64,
    bias: f64,
    post: f32,
    tanh_b: f32,
    u1: f64,
    f1: f64,
}

fn lncosh(u: f64) -> f64 {
    if u.abs() > 15.0 {
        u.abs() - core::f64::consts::LN_2
    } else {
        u.cosh().ln()
    }
}

impl AdaaTanh {
    fn new(s: Stage) -> Self {
        let bias = s.bias as f64;
        AdaaTanh {
            k: s.k as f64,
            bias,
            post: s.post,
            tanh_b: s.bias.tanh(),
            u1: bias, // state at silence: u = k·0 + b
            f1: lncosh(bias),
        }
    }

    #[inline(always)]
    fn tick(&mut self, x: f32) -> f32 {
        let u = self.k * x as f64 + self.bias;
        let du = u - self.u1;
        let f = lncosh(u);
        let y = if du.abs() < 1e-5 {
            ((u + self.u1) * 0.5).tanh()
        } else {
            (f - self.f1) / du
        };
        self.u1 = u;
        self.f1 = f;
        (y as f32 - self.tanh_b) * self.post
    }
}

// 31-tap half-band FIR (Kaiser β=8, stopband ≈ −75 dB) for the 2x
// oversampled nonlinear core. Taps computed once at init.
const HB_LEN: usize = 31;

#[derive(Clone, Copy)]
struct HalfbandFir {
    taps: [f32; HB_LEN],
    dl: [f32; HB_LEN],
    idx: usize,
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..25 {
        term *= (x / (2.0 * k as f64)) * (x / (2.0 * k as f64));
        sum += term;
    }
    sum
}

impl HalfbandFir {
    fn new() -> Self {
        let mut taps = [0.0f32; HB_LEN];
        let beta = 8.0;
        let i0b = bessel_i0(beta);
        let m = (HB_LEN - 1) as f64; // 30
        for (n, t) in taps.iter_mut().enumerate() {
            let x = n as f64 - m / 2.0; // −15..15
            let sinc = if x == 0.0 {
                1.0
            } else {
                (core::f64::consts::PI * x / 2.0).sin() / (core::f64::consts::PI * x / 2.0)
            };
            let r = 2.0 * n as f64 / m - 1.0;
            let w = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b;
            *t = (0.5 * sinc * w) as f32;
        }
        HalfbandFir { taps, dl: [0.0; HB_LEN], idx: 0 }
    }

    #[inline(always)]
    fn tick(&mut self, x: f32) -> f32 {
        self.dl[self.idx] = x;
        self.idx = (self.idx + 1) % HB_LEN;
        let mut acc = 0.0f32;
        let mut j = self.idx;
        for k in 0..HB_LEN {
            acc += self.taps[k] * self.dl[j];
            j += 1;
            if j == HB_LEN {
                j = 0;
            }
        }
        acc
    }
}

#[derive(Clone, Copy)]
enum EqKind {
    Peak,
    HighShelf,
}

#[derive(Clone, Copy)]
struct FixedEq {
    kind: EqKind,
    f: f32,
    q: f32,
    db: f32,
}

// Per-voicing topology (docs/TONE.md §1.3)
struct Voicing {
    input_hpf: f32,
    pre_lo_db: f32,
    pre_hi_db: f32,
    stages: [Option<Stage>; 3],
    is_hpf: [Option<f32>; 2], // before stage 2, before stage 3
    is_lpf: [Option<f32>; 2],
    fizz_lpf: f32,
    ts_low_f: f32,
    ts_mid_f: f32,
    ts_mid_q: f32,
    ts_high_f: f32,
    fixed_eq: [Option<FixedEq>; 2],
    power: Stage,
    gate_thresh_db: f32, // default gate threshold for this voicing
}

fn voicing_cfg(id: u32) -> Voicing {
    match id {
        // CLEAN — blackface sparkle, headroom, gentle squash
        0 => Voicing {
            input_hpf: 60.0,
            pre_lo_db: -6.0,
            pre_hi_db: 14.0,
            stages: [Some(Stage { k: 0.55, bias: 0.03, post: 1.0 / 0.55 }), None, None],
            is_hpf: [None, None],
            is_lpf: [None, None],
            fizz_lpf: 9000.0,
            ts_low_f: 100.0,
            ts_mid_f: 480.0,
            ts_mid_q: 0.6,
            ts_high_f: 4500.0,
            fixed_eq: [
                Some(FixedEq { kind: EqKind::HighShelf, f: 7000.0, q: 0.71, db: 2.0 }),
                Some(FixedEq { kind: EqKind::Peak, f: 400.0, q: 1.0, db: -1.5 }),
            ],
            power: Stage { k: 0.8, bias: 0.0, post: 1.0 / 0.8 },
            gate_thresh_db: -50.0,
        },
        // EDGE — tweed breakup, cleans up with soft picking
        1 => Voicing {
            input_hpf: 80.0,
            pre_lo_db: 0.0,
            pre_hi_db: 22.0,
            stages: [
                Some(Stage { k: 1.0, bias: 0.10, post: 1.0 }),
                Some(Stage { k: 1.4, bias: 0.06, post: 1.0 }),
                None,
            ],
            is_hpf: [Some(120.0), None],
            is_lpf: [None, None],
            fizz_lpf: 8000.0,
            ts_low_f: 100.0,
            ts_mid_f: 550.0,
            ts_mid_q: 0.7,
            ts_high_f: 3800.0,
            fixed_eq: [None, None],
            power: Stage { k: 1.0, bias: 0.0, post: 1.0 },
            gate_thresh_db: -50.0,
        },
        // CRUNCH — Marshall midrange bark
        2 => Voicing {
            input_hpf: 100.0,
            pre_lo_db: 6.0,
            pre_hi_db: 32.0,
            stages: [
                Some(Stage { k: 1.0, bias: 0.12, post: 1.0 }),
                Some(Stage { k: 1.8, bias: 0.08, post: 1.0 }),
                Some(Stage { k: 1.2, bias: 0.05, post: 1.0 }),
            ],
            is_hpf: [Some(120.0), None],
            is_lpf: [Some(7500.0), Some(6800.0)],
            fizz_lpf: 7200.0,
            ts_low_f: 110.0,
            ts_mid_f: 800.0,
            ts_mid_q: 0.7,
            ts_high_f: 3500.0,
            fixed_eq: [Some(FixedEq { kind: EqKind::Peak, f: 750.0, q: 1.4, db: 2.0 }), None],
            power: Stage { k: 1.2, bias: 0.0, post: 1.0 },
            gate_thresh_db: -48.0,
        },
        // HIGH-GAIN — tight modern metal
        _ => Voicing {
            input_hpf: 210.0,
            pre_lo_db: 18.0,
            pre_hi_db: 44.0,
            stages: [
                Some(Stage { k: 1.0, bias: 0.12, post: 1.0 }),
                Some(Stage { k: 1.8, bias: -0.08, post: 1.0 }), // alternating asymmetry
                Some(Stage { k: 1.5, bias: 0.10, post: 1.0 }),
            ],
            is_hpf: [Some(140.0), Some(110.0)],
            is_lpf: [Some(6000.0), Some(5600.0)],
            fizz_lpf: 6800.0,
            ts_low_f: 120.0,
            ts_mid_f: 650.0,
            ts_mid_q: 0.9,
            ts_high_f: 3200.0,
            fixed_eq: [Some(FixedEq { kind: EqKind::Peak, f: 450.0, q: 1.2, db: -2.0 }), None],
            power: Stage { k: 1.2, bias: 0.0, post: 1.0 },
            gate_thresh_db: -43.0,
        },
    }
}

// Noise gate: pre-drive, hysteresis + hold, exponential close (docs/TONE.md §1.2)
struct Gate {
    enabled: bool,
    open_thresh: f32,  // linear
    close_thresh: f32, // linear (open − 6 dB)
    env: f32,
    att: f32,       // detector attack coef (0.5 ms)
    rel: f32,       // detector release coef (30 ms)
    gain: f32,
    open: bool,
    hold_left: i32,
    hold_samples: i32, // 40 ms
    open_coef: f32,    // 0.15 ms ramp
    close_coef: f32,   // 90 ms exponential
}

const GATE_FLOOR: f32 = 1e-4; // −80 dB, not −∞

impl Gate {
    fn new(fs: f32) -> Self {
        Gate {
            enabled: false,
            open_thresh: db_to_lin(-50.0),
            close_thresh: db_to_lin(-56.0),
            env: 0.0,
            att: 1.0 - (-1.0 / (0.0005 * fs)).exp(),
            rel: 1.0 - (-1.0 / (0.030 * fs)).exp(),
            gain: 1.0,
            open: true,
            hold_left: 0,
            hold_samples: (0.040 * fs) as i32,
            open_coef: 1.0 - (-1.0 / (0.00015 * fs)).exp(),
            close_coef: 1.0 - (-1.0 / (0.090 * fs)).exp(),
        }
    }

    fn set_thresh_db(&mut self, db: f32) {
        self.open_thresh = db_to_lin(db);
        self.close_thresh = db_to_lin(db - 6.0);
    }

    #[inline(always)]
    fn tick(&mut self, x: f32) -> f32 {
        if !self.enabled {
            return x;
        }
        let a = x.abs();
        let coef = if a > self.env { self.att } else { self.rel };
        self.env += coef * (a - self.env);

        if self.open {
            if self.env < self.close_thresh {
                self.hold_left -= 1;
                if self.hold_left <= 0 {
                    self.open = false;
                }
            } else {
                self.hold_left = self.hold_samples;
            }
        } else if self.env > self.open_thresh {
            self.open = true;
            self.hold_left = self.hold_samples;
        }

        let (target, coef) = if self.open {
            (1.0, self.open_coef)
        } else {
            (GATE_FLOOR, self.close_coef)
        };
        self.gain += coef * (target - self.gain);
        x * self.gain
    }
}

fn db_to_lin(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

struct Amp {
    fs: f32,
    voicing_id: u32,
    v: Voicing,

    drive: f32,
    master: f32,
    pre_gain: f32,
    cur_pre: f32,
    cur_master: f32,
    bass_db: f32,
    mid_db: f32,
    treble_db: f32,
    presence_db: f32,
    eq_dirty: bool,
    user_gate_thresh: Option<f32>,

    gate: Gate,
    hp_dc_in: OnePoleHp,
    hp_tight: OnePoleHp,
    // everything between up and down runs at 2·fs (the oversampled core)
    up: HalfbandFir,
    down: HalfbandFir,
    adaa: [Option<AdaaTanh>; 3],
    power_adaa: AdaaTanh,
    is_hp: [OnePoleHp; 2],
    is_lp: [OnePoleLp; 2],
    lp_fizz: OnePoleLp,
    eq_bass: Biquad,
    eq_mid: Biquad,
    eq_treble: Biquad,
    eq_fixed: [Biquad; 2],
    eq_presence: Biquad,
    dc_block: OnePoleHp,
}

impl Amp {
    fn new(fs: f32) -> Self {
        let mut a = Amp {
            fs,
            voicing_id: 2,
            v: voicing_cfg(2),
            drive: 0.5,
            master: 0.8,
            pre_gain: 1.0,
            cur_pre: 1.0,
            cur_master: 0.8,
            bass_db: 0.0,
            mid_db: 0.0,
            treble_db: 0.0,
            presence_db: 3.0,
            eq_dirty: true,
            user_gate_thresh: None,
            gate: Gate::new(fs),
            hp_dc_in: Default::default(),
            hp_tight: Default::default(),
            up: HalfbandFir::new(),
            down: HalfbandFir::new(),
            adaa: [None, None, None],
            power_adaa: AdaaTanh::new(Stage { k: 1.0, bias: 0.0, post: 1.0 }),
            is_hp: Default::default(),
            is_lp: Default::default(),
            lp_fizz: Default::default(),
            eq_bass: Default::default(),
            eq_mid: Default::default(),
            eq_treble: Default::default(),
            eq_fixed: Default::default(),
            eq_presence: Default::default(),
            dc_block: Default::default(),
        };
        a.hp_dc_in.set(fs, 20.0);
        a.dc_block.set(fs, 10.0);
        a.apply_voicing();
        a
    }

    fn apply_voicing(&mut self) {
        let v = &self.v;
        let fs_os = self.fs * 2.0; // the nonlinear core runs oversampled
        self.hp_tight.set(self.fs, v.input_hpf);
        for i in 0..2 {
            if let Some(f) = v.is_hpf[i] {
                self.is_hp[i].set(fs_os, f);
            }
            if let Some(f) = v.is_lpf[i] {
                self.is_lp[i].set(fs_os, f);
            }
        }
        self.lp_fizz.set(fs_os, v.fizz_lpf);
        for i in 0..2 {
            match v.fixed_eq[i] {
                Some(e) => match e.kind {
                    EqKind::Peak => self.eq_fixed[i].peaking(fs_os, e.f, e.q, e.db),
                    EqKind::HighShelf => self.eq_fixed[i].high_shelf(fs_os, e.f, e.db),
                },
                None => self.eq_fixed[i].bypass(),
            }
        }
        for i in 0..3 {
            self.adaa[i] = v.stages[i].map(AdaaTanh::new);
        }
        self.power_adaa = AdaaTanh::new(v.power);
        let thresh = self.user_gate_thresh.unwrap_or(v.gate_thresh_db);
        self.gate.set_thresh_db(thresh);
        self.pre_gain = self.drive_to_gain();
        self.eq_dirty = true;
    }

    fn drive_to_gain(&self) -> f32 {
        let db = self.v.pre_lo_db + self.drive * (self.v.pre_hi_db - self.v.pre_lo_db);
        db_to_lin(db)
    }

    fn set_param(&mut self, id: u32, val: f32) {
        match id {
            0 => {
                self.drive = val.clamp(0.0, 1.0);
                self.pre_gain = self.drive_to_gain();
            }
            1 => { self.bass_db = val.clamp(-12.0, 12.0); self.eq_dirty = true; }
            2 => { self.mid_db = val.clamp(-12.0, 12.0); self.eq_dirty = true; }
            3 => { self.treble_db = val.clamp(-12.0, 12.0); self.eq_dirty = true; }
            4 => self.master = val.clamp(0.0, 1.5),
            5 => {
                let id = (val as u32).min(3);
                if id != self.voicing_id {
                    self.voicing_id = id;
                    self.v = voicing_cfg(id);
                    self.apply_voicing();
                }
            }
            6 => self.gate.enabled = val > 0.5,
            7 => {
                self.user_gate_thresh = Some(val.clamp(-70.0, -20.0));
                self.gate.set_thresh_db(val.clamp(-70.0, -20.0));
            }
            8 => { self.presence_db = val.clamp(0.0, 6.0); self.eq_dirty = true; }
            _ => {}
        }
    }

    fn update_eq(&mut self) {
        let v = &self.v;
        let fs_os = self.fs * 2.0; // tonestack sits inside the oversampled core
        self.eq_bass.low_shelf(fs_os, v.ts_low_f, self.bass_db);
        self.eq_mid.peaking(fs_os, v.ts_mid_f, v.ts_mid_q, self.mid_db);
        // cap the high-gain treble shelf: post-distortion boosts above +8 dB
        // are a wasp factory (anti-fizz checklist #7)
        let treble = if self.voicing_id == 3 { self.treble_db.min(8.0) } else { self.treble_db };
        self.eq_treble.high_shelf(fs_os, v.ts_high_f, treble);
        self.eq_presence.high_shelf(self.fs, 4600.0, self.presence_db);
        self.eq_dirty = false;
    }

    fn process(&mut self, input: &[f32], out: &mut [f32]) {
        if self.eq_dirty {
            self.update_eq();
        }
        let has_is_hp = [self.v.is_hpf[0].is_some(), self.v.is_hpf[1].is_some()];
        let has_is_lp = [self.v.is_lpf[0].is_some(), self.v.is_lpf[1].is_some()];

        for (o, &s) in out.iter_mut().zip(input.iter()) {
            self.cur_pre += 0.004 * (self.pre_gain - self.cur_pre);
            self.cur_master += 0.004 * (self.master - self.cur_master);

            // base-rate front end
            let mut front = self.hp_dc_in.tick(s);
            front = self.hp_tight.tick(front);
            front = self.gate.tick(front);
            front *= self.cur_pre;

            // 2x oversampled nonlinear core: zero-stuff (gain 2 compensates),
            // run the whole stage chain + tonestack + power per os-sample,
            // decimate through the second half-band
            let mut y = 0.0f32;
            for phase in 0..2 {
                let mut x = self.up.tick(if phase == 0 { 2.0 * front } else { 0.0 });

                if let Some(st) = self.adaa[0].as_mut() {
                    x = st.tick(x);
                }
                if has_is_hp[0] { x = self.is_hp[0].tick(x); }
                if has_is_lp[0] { x = self.is_lp[0].tick(x); }
                if let Some(st) = self.adaa[1].as_mut() {
                    x = st.tick(x);
                }
                if has_is_hp[1] { x = self.is_hp[1].tick(x); }
                if has_is_lp[1] { x = self.is_lp[1].tick(x); }
                if let Some(st) = self.adaa[2].as_mut() {
                    x = st.tick(x);
                }

                x = self.lp_fizz.tick(x);
                x = self.eq_treble.tick(self.eq_mid.tick(self.eq_bass.tick(x)));
                x = self.eq_fixed[0].tick(x);
                x = self.eq_fixed[1].tick(x);
                x = self.power_adaa.tick(x);
                y = self.down.tick(x);
            }

            // back at base rate
            let mut x = self.eq_presence.tick(y);
            x = self.dc_block.tick(x);
            *o = x * self.cur_master * OUT_SCALE;
        }
    }
}

#[no_mangle]
pub extern "C" fn init(sample_rate: f32) {
    unsafe { AMP = Some(Amp::new(sample_rate)) };
}

#[no_mangle]
pub extern "C" fn in_ptr() -> *mut f32 {
    unsafe { IN_BUF.as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn out_ptr() -> *const f32 {
    unsafe { OUT_BUF.as_ptr() }
}

#[no_mangle]
pub extern "C" fn set_param(id: u32, value: f32) {
    if let Some(amp) = unsafe { AMP.as_mut() } {
        amp.set_param(id, value);
    }
}

#[no_mangle]
pub extern "C" fn process(n: usize) {
    let n = n.min(MAX_BLOCK);
    if let Some(amp) = unsafe { AMP.as_mut() } {
        unsafe { amp.process(&IN_BUF[..n], &mut OUT_BUF[..n]) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f32 = 48000.0;

    fn run(amp: &mut Amp, input: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; input.len()];
        for (ic, oc) in input.chunks(MAX_BLOCK).zip(out.chunks_mut(MAX_BLOCK)) {
            amp.process(ic, oc);
        }
        out
    }

    fn sine(freq: f32, amp: f32, secs: f32) -> Vec<f32> {
        (0..(FS * secs) as usize)
            .map(|i| amp * (2.0 * PI * freq * i as f32 / FS).sin())
            .collect()
    }

    fn assert_all_finite(v: &[f32]) {
        assert!(v.iter().all(|x| x.is_finite()), "output contains NaN/inf");
    }

    fn rms(v: &[f32]) -> f32 {
        (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
    }

    fn mean(v: &[f32]) -> f32 {
        v.iter().sum::<f32>() / v.len() as f32
    }

    #[test]
    fn silence_in_silence_out_all_voicings() {
        for voicing in 0..4 {
            let mut amp = Amp::new(FS);
            amp.set_param(5, voicing as f32);
            let out = run(&mut amp, &vec![0.0; 48000]);
            assert_all_finite(&out);
            assert!(
                rms(&out[24000..]) < 1e-4,
                "voicing {voicing} generates output from silence"
            );
        }
    }

    #[test]
    fn guitar_level_sine_passes_sanely_all_voicings() {
        for voicing in 0..4 {
            let mut amp = Amp::new(FS);
            amp.set_param(5, voicing as f32);
            let out = run(&mut amp, &sine(220.0, 0.1, 1.0));
            assert_all_finite(&out);
            let tail = &out[24000..];
            let r = rms(tail);
            assert!(r > 0.005, "voicing {voicing} too quiet: rms {r}");
            assert!(tail.iter().all(|x| x.abs() <= 1.0), "voicing {voicing} exceeds ±1.0");
        }
    }

    #[test]
    fn no_dc_offset_at_any_drive() {
        // biased tanh stages must not leak DC (anti-fizz checklist #6)
        for voicing in 0..4 {
            for drive in [0.0, 0.5, 1.0] {
                let mut amp = Amp::new(FS);
                amp.set_param(5, voicing as f32);
                amp.set_param(0, drive);
                let out = run(&mut amp, &sine(110.0, 0.1, 1.0));
                let dc = mean(&out[24000..]).abs();
                assert!(dc < 0.005, "voicing {voicing} drive {drive}: DC {dc}");
            }
        }
    }

    #[test]
    fn extreme_settings_stay_bounded() {
        let mut amp = Amp::new(FS);
        amp.set_param(5, 3.0);
        amp.set_param(0, 1.0);
        amp.set_param(1, 12.0);
        amp.set_param(2, 12.0);
        amp.set_param(3, 12.0);
        amp.set_param(4, 1.5);
        amp.set_param(8, 6.0);
        let sq: Vec<f32> = (0..96000)
            .map(|i| if (i as f32 * 110.0 / FS).fract() < 0.5 { 1.0 } else { -1.0 })
            .collect();
        let out = run(&mut amp, &sq);
        assert_all_finite(&out);
        assert!(out.iter().all(|x| x.abs() <= 1.5), "unbounded output");
    }

    #[test]
    fn eq_biquads_are_stable() {
        let mut amp = Amp::new(FS);
        amp.set_param(1, -12.0);
        amp.set_param(2, 12.0);
        amp.set_param(3, -12.0);
        amp.set_param(8, 6.0);
        let mut input = vec![0.0f32; 96000];
        input[0] = 1.0;
        let out = run(&mut amp, &input);
        assert_all_finite(&out);
        assert!(rms(&out[72000..]) < 1e-5, "filter ringing does not decay");
    }

    #[test]
    fn param_changes_do_not_click() {
        let mut amp = Amp::new(FS);
        let input = sine(196.0, 0.1, 2.0);
        let mut out = vec![0.0f32; input.len()];
        for (i, (ic, oc)) in input.chunks(MAX_BLOCK).zip(out.chunks_mut(MAX_BLOCK)).enumerate() {
            if i == 200 {
                amp.set_param(0, 0.9);
                amp.set_param(4, 1.2);
            }
            amp.process(ic, oc);
        }
        assert_all_finite(&out);
        let max_step = out.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        assert!(max_step < 0.35, "audible click on param change: step {max_step}");
    }

    #[test]
    fn gate_blocks_hum_passes_signal() {
        let mut amp = Amp::new(FS);
        amp.set_param(5, 3.0); // high-gain: where the gate matters
        amp.set_param(6, 1.0); // gate on
        amp.set_param(7, -43.0);

        // hum only: 50 Hz at −60 dBFS → must be gated to (near) silence
        let hum = sine(50.0, 0.001, 1.0);
        let out_hum = run(&mut amp, &hum);
        assert!(rms(&out_hum[24000..]) < 1e-3, "gate lets −60 dB hum through");

        // real signal: −20 dBFS pluck → must pass
        let mut amp2 = Amp::new(FS);
        amp2.set_param(5, 3.0);
        amp2.set_param(6, 1.0);
        amp2.set_param(7, -43.0);
        let sig = sine(110.0, 0.1, 1.0);
        let out_sig = run(&mut amp2, &sig);
        assert!(rms(&out_sig[24000..]) > 0.01, "gate blocks a real signal");
    }

    #[test]
    fn gate_opens_fast_enough_for_pick_attack() {
        let mut amp = Amp::new(FS);
        amp.set_param(5, 3.0);
        amp.set_param(6, 1.0);
        // silence, then a pluck: the note must be audible within 2 ms
        let mut input = vec![0.0f32; 24000];
        input.extend(sine(110.0, 0.15, 0.5));
        let out = run(&mut amp, &input);
        let attack_zone = &out[24000..24000 + 96]; // first 2 ms of the pluck
        assert!(
            attack_zone.iter().any(|x| x.abs() > 0.01),
            "gate smears the pick attack"
        );
    }

    fn goertzel_power(v: &[f32], fs: f32, f: f32) -> f32 {
        let w = 2.0 * PI * f / fs;
        let coef = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for &x in v {
            let s0 = x + coef * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - coef * s1 * s2) / (v.len() as f32 * v.len() as f32)
    }

    #[test]
    fn aliasing_is_suppressed_at_high_gain() {
        // 6.5 kHz sine at max drive: the 5th harmonic (32.5 kHz) would fold
        // to 15.5 kHz at 48 kHz — 15.5 kHz is NOT a harmonic of 6.5 kHz, so
        // any energy there is pure aliasing. With 2x OS + ADAA it must sit
        // far below the fundamental.
        let mut amp = Amp::new(FS);
        amp.set_param(5, 3.0);
        amp.set_param(0, 1.0);
        let out = run(&mut amp, &sine(6500.0, 0.15, 1.0));
        let tail = &out[24000..];
        let fund = goertzel_power(tail, FS, 6500.0);
        let alias = goertzel_power(tail, FS, 15500.0);
        assert!(fund > 0.0, "no fundamental?");
        let ratio_db = 10.0 * (alias / fund).log10();
        assert!(
            ratio_db < -50.0,
            "alias at 15.5 kHz only {ratio_db:.1} dB below fundamental"
        );
    }

    #[test]
    fn voicing_switch_mid_signal_is_safe() {
        let mut amp = Amp::new(FS);
        let input = sine(196.0, 0.1, 2.0);
        let mut out = vec![0.0f32; input.len()];
        for (i, (ic, oc)) in input.chunks(MAX_BLOCK).zip(out.chunks_mut(MAX_BLOCK)).enumerate() {
            if i % 100 == 99 {
                amp.set_param(5, ((i / 100) % 4) as f32);
            }
            amp.process(ic, oc);
        }
        assert_all_finite(&out);
        assert!(out.iter().all(|x| x.abs() <= 1.5));
    }
}
