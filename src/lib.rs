#![allow(static_mut_refs)] // single-threaded inside one AudioWorklet — no aliasing possible

// OpenAmp DSP core — wasm32-unknown-unknown, no JS glue, C ABI only.
// Lives inside an AudioWorklet: single-threaded, static buffers, zero
// allocation after init. The worklet copies input into IN_BUF, calls
// process(n), reads OUT_BUF. Blocks are ≤128 frames (one render quantum).
//
// v1 voicing "crunch": tight HPF → two asymmetric tanh stages with an
// inter-stage fizz LPF → 3-band tonestack (RBJ biquads) → soft power-amp
// clip → master. Params are smoothed to avoid zipper noise.

const MAX_BLOCK: usize = 128;
const PI: f32 = core::f32::consts::PI;

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

#[derive(Default)]
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

#[derive(Default)]
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

const STAGE1_BIAS: f32 = 0.12; // asymmetry → even harmonics

#[derive(Default)]
struct Amp {
    fs: f32,
    // targets (set_param) and smoothed current values
    drive: f32,
    master: f32,
    pre_gain: f32,
    cur_pre: f32,
    cur_master: f32,
    bass_db: f32,
    mid_db: f32,
    treble_db: f32,
    eq_dirty: bool,

    hp_dc: OnePoleHp,
    hp_tight: OnePoleHp,
    lp_fizz: OnePoleLp,
    eq_bass: Biquad,
    eq_mid: Biquad,
    eq_treble: Biquad,
    bias_off: f32,
}

impl Amp {
    fn new(fs: f32) -> Self {
        let mut a = Amp {
            fs,
            drive: 0.4,
            master: 0.8,
            eq_dirty: true,
            bias_off: STAGE1_BIAS.tanh(),
            ..Default::default()
        };
        a.hp_dc.set(fs, 20.0);
        a.hp_tight.set(fs, 90.0);
        a.lp_fizz.set(fs, 6500.0);
        a.pre_gain = Self::drive_to_gain(a.drive);
        a.cur_pre = a.pre_gain;
        a.cur_master = a.master;
        a
    }

    fn drive_to_gain(drive: f32) -> f32 {
        // 0..1 → -6..+40 dB of pre-gain
        10f32.powf((-6.0 + drive * 46.0) / 20.0)
    }

    fn set_param(&mut self, id: u32, v: f32) {
        match id {
            0 => {
                self.drive = v.clamp(0.0, 1.0);
                self.pre_gain = Self::drive_to_gain(self.drive);
            }
            1 => { self.bass_db = v.clamp(-12.0, 12.0); self.eq_dirty = true; }
            2 => { self.mid_db = v.clamp(-12.0, 12.0); self.eq_dirty = true; }
            3 => { self.treble_db = v.clamp(-12.0, 12.0); self.eq_dirty = true; }
            4 => self.master = v.clamp(0.0, 1.5),
            _ => {}
        }
    }

    fn update_eq(&mut self) {
        self.eq_bass.low_shelf(self.fs, 110.0, self.bass_db);
        self.eq_mid.peaking(self.fs, 550.0, 0.8, self.mid_db);
        self.eq_treble.high_shelf(self.fs, 3200.0, self.treble_db);
        self.eq_dirty = false;
    }

    fn process(&mut self, input: &[f32], out: &mut [f32]) {
        if self.eq_dirty {
            self.update_eq();
        }
        for (o, &s) in out.iter_mut().zip(input.iter()) {
            // param smoothing (~ms-scale, per sample)
            self.cur_pre += 0.004 * (self.pre_gain - self.cur_pre);
            self.cur_master += 0.004 * (self.master - self.cur_master);

            let mut x = self.hp_dc.tick(s);
            x = self.hp_tight.tick(x);
            x *= self.cur_pre;
            // stage 1: asymmetric clip
            x = (x + STAGE1_BIAS).tanh() - self.bias_off;
            x = self.lp_fizz.tick(x);
            // stage 2: symmetric, harder
            x = (1.8 * x).tanh();
            // tonestack
            x = self.eq_treble.tick(self.eq_mid.tick(self.eq_bass.tick(x)));
            // power amp: gentle squash
            x = (1.2 * x).tanh();
            *o = x * self.cur_master;
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

    #[test]
    fn silence_in_silence_out() {
        let mut amp = Amp::new(FS);
        let out = run(&mut amp, &vec![0.0; 48000]);
        assert_all_finite(&out);
        // after the smoothing settles there must be no self-noise / DC
        assert!(rms(&out[24000..]) < 1e-4, "amp generates output from silence");
    }

    #[test]
    fn guitar_level_sine_passes_sanely() {
        let mut amp = Amp::new(FS);
        let out = run(&mut amp, &sine(82.4, 0.1, 1.0)); // low E, DI level
        assert_all_finite(&out);
        let tail = &out[24000..];
        let r = rms(tail);
        assert!(r > 0.01, "output too quiet: rms {r}");
        assert!(tail.iter().all(|x| x.abs() <= 1.5), "output exceeds ±1.5");
    }

    #[test]
    fn extreme_settings_stay_bounded() {
        let mut amp = Amp::new(FS);
        amp.set_param(0, 1.0); // full drive
        amp.set_param(1, 12.0);
        amp.set_param(2, 12.0);
        amp.set_param(3, 12.0);
        amp.set_param(4, 1.5);
        // hostile input: full-scale square wave 110 Hz
        let sq: Vec<f32> = (0..96000)
            .map(|i| if (i as f32 * 110.0 / FS).fract() < 0.5 { 1.0 } else { -1.0 })
            .collect();
        let out = run(&mut amp, &sq);
        assert_all_finite(&out);
        assert!(out.iter().all(|x| x.abs() <= 2.0), "unbounded output");
    }

    #[test]
    fn eq_biquads_are_stable() {
        let mut amp = Amp::new(FS);
        amp.set_param(1, -12.0);
        amp.set_param(2, 12.0);
        amp.set_param(3, -12.0);
        let mut input = vec![0.0f32; 96000];
        input[0] = 1.0; // impulse
        let out = run(&mut amp, &input);
        assert_all_finite(&out);
        // response must decay: last half-second essentially silent
        assert!(rms(&out[72000..]) < 1e-5, "filter ringing does not decay");
    }

    #[test]
    fn param_changes_do_not_click() {
        let mut amp = Amp::new(FS);
        let input = sine(196.0, 0.1, 2.0); // G3
        let mut out = vec![0.0f32; input.len()];
        for (i, (ic, oc)) in input.chunks(MAX_BLOCK).zip(out.chunks_mut(MAX_BLOCK)).enumerate() {
            if i == 200 {
                amp.set_param(0, 0.9); // slam drive mid-stream
                amp.set_param(4, 1.2);
            }
            amp.process(ic, oc);
        }
        assert_all_finite(&out);
        // no sample-to-sample discontinuity beyond what the waveform allows
        let max_step = out.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        assert!(max_step < 0.35, "audible click on param change: step {max_step}");
    }
}
