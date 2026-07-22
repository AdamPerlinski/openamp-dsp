# openamp-dsp

Guitar amp DSP core in pure Rust. Powers [OpenAmp](https://openamp.example)
— a free guitar amp that runs in a browser tab — and compiles unchanged to
`wasm32-unknown-unknown` (for an AudioWorklet) or native (for desktop use).

**No dependencies. No allocation in the audio path. C ABI, ~40 KB of wasm.**

## What's inside

```
20 Hz DC HPF → voicing tightness HPF → noise gate (hysteresis + hold)
  → pre-gain → [2x oversampled core: up to 3 asymmetric tanh stages with
     first-order ADAA, interstage HPF/LPF, anti-fizz LPF, 3-band RBJ
     tonestack, power stage] → presence shelf → DC blocker → master
```

- **4 voicings** — clean / edge / crunch / high-gain — as data (stage count,
  bias, per-stage gain, filter corners), not code branches. See `voicing_cfg`.
- **Anti-aliasing**: the nonlinear core runs at 2× the context rate between a
  pair of 31-tap Kaiser half-band FIRs, and every `tanh` stage uses
  first-order antiderivative anti-derivative aliasing suppression (ADAA,
  Parker et al. DAFx-16) with f64 internals. There's a regression test that
  measures alias energy with Goertzel and fails if it creeps back.
- **Noise gate** tuned for high-gain playing: 0.5 ms detector attack, open
  ramp 0.15 ms (pick transients pass), 40 ms hold, 90 ms exponential close.
- **DC-safe asymmetric stages**: every biased stage computes
  `tanh(k·x + b) − tanh(b)`, so bias never leaks DC downstream.
- **Loudness-matched voicings**: per-voicing makeup gain smoothed through the
  master glide, so switching amps never blasts your ears.

## API (C ABI — works from any host)

```c
void  init(float sample_rate);
float* in_ptr();            // write ≤128 input samples here
const float* out_ptr();     // read processed samples here
void  process(size_t n);    // n ≤ 128
void  set_param(uint32_t id, float value);
```

Params: `0` drive 0..1 · `1..3` bass/mid/treble dB · `4` master ·
`5` voicing 0..3 · `6` gate on · `7` gate threshold dB · `8` presence dB.

## Build

```sh
# wasm (SIMD enabled via .cargo/config.toml)
cargo build --release --target wasm32-unknown-unknown

# tests run native
cargo test --target x86_64-unknown-linux-gnu
```

## Tests

10 tests cover: silence-in/silence-out per voicing, bounded output under
hostile input, biquad stability at extreme EQ, no zipper clicks on parameter
slams, gate behavior (blocks −60 dB hum, passes −20 dB signal, opens within
2 ms of a pick attack), DC-offset sweep across all voicings and drives,
mid-signal voicing switches, and measured aliasing suppression at max gain.

## License

MIT
