<p align="center">
  <img src="https://raw.githubusercontent.com/AdamPerlinski/openamp-dsp/main/docs/logo.svg" alt="openamp-dsp" width="480">
</p>

<p align="center">
  <a href="https://github.com/AdamPerlinski/openamp-dsp/blob/main/LICENSE"><img src="https://img.shields.io/github/license/AdamPerlinski/openamp-dsp?color=f87171" alt="license"></a>
  <img src="https://img.shields.io/badge/wasm-42%20KB-f87171" alt="wasm size">
  <img src="https://img.shields.io/badge/dependencies-0-f87171" alt="zero deps">
  <img src="https://img.shields.io/badge/alloc%20in%20audio%20path-0-f87171" alt="no alloc">
  <a href="https://github.com/AdamPerlinski/openamp-dsp"><img src="https://img.shields.io/github/stars/AdamPerlinski/openamp-dsp?style=social" alt="GitHub stars"></a>
</p>

**You don't need a plugin installer for guitar tone.**

Most guitarists just want to plug in and sound good. You don't need a 2 GB
amp suite for that — a whole tube-style amp fits in **42 KB of WASM** and
runs in a browser tab at 128-sample latency.

openamp-dsp is the amp core behind **OpenAmp** (free browser amp, demo at
launch): 4 voicings, staged asymmetric saturation with real anti-aliasing,
a noise gate that doesn't eat pick attack, and a 3-band tonestack. Pure
Rust, zero dependencies, zero allocation in the audio path. The same crate
compiles to `wasm32` for an AudioWorklet or native for desktop.

```sh
cargo add openamp-dsp   # or: build the C-ABI wasm, see below
```

---

## What Can You Do With It?

### Run a whole amp in an AudioWorklet

```js
// main thread: fetch the wasm, hand it to your worklet
const bytes = await (await fetch('openamp_dsp.wasm')).arrayBuffer();
node.port.postMessage({ type: 'wasm', bytes }, [bytes]);

// inside the AudioWorkletProcessor (no fetch there — instantiate from bytes)
const { instance } = await WebAssembly.instantiate(bytes, {});
const amp = instance.exports;
amp.init(sampleRate);
const input  = new Float32Array(amp.memory.buffer, amp.in_ptr(),  128);
const output = new Float32Array(amp.memory.buffer, amp.out_ptr(), 128);

// per render quantum:
input.set(channelData);
amp.process(128);
channelData.set(output);
```

### Turn a clean DI into four different amps

```js
amp.set_param(5, 0); // CLEAN     — blackface sparkle, headroom
amp.set_param(5, 1); // EDGE      — tweed breakup, cleans up with soft picking
amp.set_param(5, 2); // CRUNCH    — Marshall-ish midrange bark
amp.set_param(5, 3); // HIGH-GAIN — tight modern metal (210 Hz pre-clip HPF)
```

Voicings are **data, not code branches** — stage count, per-stage gain and
bias, interstage filter corners, tonestack centers. Adding a fifth amp is a
table entry, not a refactor.

### Gate hum without eating your pick attack

```js
amp.set_param(6, 1);    // gate on
amp.set_param(7, -43);  // threshold (dB) — crank it for djent chops
```

Detector attack 0.5 ms, opening ramp **0.15 ms** (transients pass unsmeared),
40 ms hold, 90 ms exponential close. There's a test that fails if a pick
attack takes longer than 2 ms to come through.

### Use it native (same crate, no glue)

```rust
// desktop build — e.g. inside a cpal callback
openamp_dsp::init(48_000.0);
openamp_dsp::set_param(0, 0.7); // drive
// copy input → in_ptr(), process(n), read out_ptr()
```

---

## Which Voicing When?

| You want to sound like | Voicing | Tips |
|---|---|---|
| Pop, funk, worship clean | `0` CLEAN | presence up (`8`), drive < 0.3 |
| Blues, indie, "just breaking up" | `1` EDGE | ride your guitar's volume knob |
| AC/DC, classic rock riffs | `2` CRUNCH | mids up (`2`), drive ≈ 0.5 |
| Metal rhythm, drop tunings | `3` HIGH-GAIN | gate on, bass up, mids scooped |
| Doom/sludge | `3` + drive 1.0 | bass +6, treble −3, slow down |
| Djent | `3` + gate −38 dB | LESS drive than you think, mids UP |

## Param Reference

| id | param | range | notes |
|----|-------|-------|-------|
| 0 | drive | 0..1 | mapped to a per-voicing dB range (e.g. +18…+44 for high-gain) |
| 1 | bass | −12..+12 dB | low shelf, per-voicing corner |
| 2 | mid | −12..+12 dB | peaking, per-voicing center/Q |
| 3 | treble | −12..+12 dB | high shelf (capped +8 dB in high-gain — anti-fizz) |
| 4 | master | 0..1.5 | smoothed; includes per-voicing loudness makeup |
| 5 | voicing | 0..3 | clean / edge / crunch / high-gain |
| 6 | gate | 0/1 | pre-drive noise gate |
| 7 | gate threshold | −70..−20 dB | hysteresis: close = open − 6 dB |
| 8 | presence | 0..+6 dB | 4.6 kHz shelf, post power stage |

All params glide per-sample — slam them mid-note, no zipper, no clicks
(tested).

---

## Under the Hood

```
20 Hz DC HPF → tightness HPF (60–210 Hz by voicing) → noise gate → pre-gain
  → ⌈2× OVERSAMPLED CORE⌉ 31-tap Kaiser half-band up
  →   1–3 × asymmetric tanh stages (first-order ADAA, f64 internals)
  →   interstage HPF/LPF (kills intermod mud between stages)
  →   anti-fizz LPF → RBJ tonestack → fixed voicing EQ → power stage
  → ⌊half-band down⌋ → presence shelf → 10 Hz DC blocker → master
```

Why you won't hear bees in a can:

- **2× oversampling + ADAA** (Parker et al., DAFx-16) on every saturation
  stage. A regression test drives a 6.5 kHz tone at max gain and measures
  alias energy at 15.5 kHz with Goertzel — it fails if aliasing creeps back.
- **DC-corrected bias**: every stage is `tanh(k·x + b) − tanh(b)`, so the
  asymmetry that makes even harmonics never leaks DC into the next stage.
- **Gain is distributed** — no stage sees more than ~+14 dB over its
  predecessor. Single-stage tanh at +40 dB is a fuzz pedal, not an amp.
- **Loudness-matched voicings** — makeup gain rides the smoothed master, so
  A/B-ing amps never turns into "louder = better".

## Tests

```sh
cargo test --target x86_64-unknown-linux-gnu
```

10 tests: silence-in/silence-out per voicing · bounded output under
full-scale square waves · biquad stability at extreme EQ · no clicks on
param slams · gate blocks −60 dB hum, passes −20 dB signal, opens < 2 ms ·
zero DC at any drive · mid-note voicing switches · measured alias
suppression at max gain.

## Building the wasm

```sh
cargo build --release --target wasm32-unknown-unknown
# → target/wasm32-unknown-unknown/release/openamp_dsp.wasm (~42 KB)
```

SIMD is enabled via `.cargo/config.toml`. The C ABI surface is 5 functions —
host it from any language.

## License

MIT — take it, ship it, make it scream.
