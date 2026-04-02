#!/usr/bin/env python3
"""
Generate Fourier analysis training examples.

Covers: FFT, frequency detection, signal decomposition, filtering,
power spectral density, aliasing concepts, and signal reconstruction.

Usage:
    python training/scripts/generate_fourier.py --output training/datasets/fourier/basic.jsonl --count 100
"""

import sys
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.executor.python_exec import run_once, PythonExecutor


def execute(code: str) -> str:
    return run_once(code).output_text()


def make_example(ex_id, category, difficulty, query, think, model_text, code):
    output = execute(code)
    parts = []
    if think:
        parts.append(f"<think>\n{think.strip()}\n</think>")
    if model_text:
        parts.append(f"<model>\n{model_text.strip()}\n</model>")
    parts.append(f"<code>\n{code.strip()}\n</code>")
    parts.append(f"<output>\n{output.strip()}\n</output>")
    return {"id": ex_id, "category": category, "difficulty": difficulty,
            "query": query, "response": "\n".join(parts)}


# ─── Generators ───────────────────────────────────────────────────────────────

def gen_identify_frequencies(rng):
    """FFT of a signal with known frequency components"""
    sample_rate = rng.choice([100, 200, 500, 1000])
    duration = rng.choice([1.0, 2.0])
    freqs = sorted(rng.sample([2, 3, 5, 7, 10, 15, 20, 25, 30], rng.randint(2, 3)))
    amps = [round(rng.uniform(0.5, 2.0), 1) for _ in freqs]

    signal_desc = " + ".join(f"{a}·sin(2π·{f}·t)" for a, f in zip(amps, freqs))
    query = f"A signal is composed of: {signal_desc}. Sample rate {sample_rate} Hz, duration {duration}s. Use FFT to find the dominant frequencies."
    think = (
        "Apply FFT to decompose the signal into frequency components. "
        "The magnitude spectrum peaks tell us which frequencies are present."
    )
    model_text = "Build signal → rfft → magnitude spectrum → find peaks above threshold."

    signal_code = " + ".join(f"{a} * np.sin(2 * np.pi * {f} * t)" for a, f in zip(amps, freqs))
    code = (
        f"import numpy as np\n\n"
        f"sample_rate = {sample_rate}\n"
        f"duration = {duration}\n"
        f"n = int(sample_rate * duration)\n"
        f"t = np.linspace(0, duration, n, endpoint=False)\n\n"
        f"# Construct signal\n"
        f"signal = {signal_code}\n\n"
        f"# FFT\n"
        f"fft_vals = np.fft.rfft(signal)\n"
        f"freqs = np.fft.rfftfreq(n, d=1/sample_rate)\n"
        f"magnitudes = np.abs(fft_vals) * 2 / n\n\n"
        f"# Find dominant frequencies (magnitude > 0.1)\n"
        f"threshold = 0.1\n"
        f"dominant = [(round(freqs[i], 1), round(magnitudes[i], 3))\n"
        f"            for i in np.where(magnitudes > threshold)[0]]\n"
        f"dominant.sort(key=lambda x: -x[1])\n"
        f"print(f'Dominant frequencies (Hz, amplitude):')\n"
        f"for freq, amp in dominant[:6]:\n"
        f"    print(f'  {{freq:.1f}} Hz  amp={{amp:.3f}}')"
    )
    return query, think, model_text, code


def gen_nyquist(rng):
    """Aliasing and Nyquist theorem"""
    signal_freq = rng.choice([10, 15, 20, 25, 30])
    sample_rate = rng.choice([15, 20, 25, 35, 50, 100])
    nyquist = sample_rate / 2

    query = (
        f"A signal has frequency {signal_freq} Hz. It is sampled at {sample_rate} Hz. "
        f"Does aliasing occur? If so, what is the alias frequency?"
    )
    think = (
        "Nyquist theorem: to avoid aliasing, sample rate must be > 2 × signal frequency. "
        "If aliasing occurs, alias frequency = |signal_freq - round(signal_freq/sample_rate) * sample_rate|."
    )
    code = (
        f"signal_freq = {signal_freq}\n"
        f"sample_rate = {sample_rate}\n"
        f"nyquist = sample_rate / 2\n\n"
        f"aliasing = signal_freq > nyquist\n"
        f"print(f'Nyquist frequency: {{nyquist}} Hz')\n"
        f"print(f'Signal frequency: {{signal_freq}} Hz')\n"
        f"print(f'Aliasing occurs: {{aliasing}}')\n"
        f"if aliasing:\n"
        f"    alias = abs(signal_freq - round(signal_freq / sample_rate) * sample_rate)\n"
        f"    print(f'Alias frequency: {{alias}} Hz')\n"
        f"else:\n"
        f"    print('Signal is properly captured.')"
    )
    return query, think, "", code


def gen_lowpass_filter(rng):
    """Apply a low-pass filter to remove high-frequency noise"""
    sample_rate = rng.choice([200, 500, 1000])
    signal_freq = rng.choice([5, 10, 15])
    noise_freq = rng.choice([50, 80, 100, 120])
    cutoff = rng.choice([20, 30, 40])

    query = (
        f"A signal at {signal_freq} Hz is contaminated with noise at {noise_freq} Hz. "
        f"Apply a low-pass Butterworth filter with cutoff {cutoff} Hz (sample rate {sample_rate} Hz). "
        f"What is the signal-to-noise ratio before and after?"
    )
    think = (
        f"Apply a digital Butterworth low-pass filter at {cutoff} Hz. "
        f"Measure the power of the signal component vs noise before and after filtering."
    )
    exec_inst = PythonExecutor()
    code = (
        f"import numpy as np\n"
        f"from scipy import signal as sp\n\n"
        f"sample_rate = {sample_rate}\n"
        f"n = sample_rate * 2\n"
        f"t = np.linspace(0, 2, n, endpoint=False)\n\n"
        f"sig = np.sin(2 * np.pi * {signal_freq} * t)\n"
        f"noise = 0.5 * np.sin(2 * np.pi * {noise_freq} * t)\n"
        f"mixed = sig + noise\n\n"
        f"# Design Butterworth low-pass filter\n"
        f"b, a = sp.butter(4, {cutoff}, btype='low', fs=sample_rate)\n"
        f"filtered = sp.filtfilt(b, a, mixed)\n\n"
        f"# SNR before/after\n"
        f"def snr_db(signal_power, noise_power):\n"
        f"    return 10 * np.log10(signal_power / noise_power)\n\n"
        f"sig_power = np.mean(sig**2)\n"
        f"noise_before = np.mean(noise**2)\n"
        f"residual = filtered - sig\n"
        f"noise_after = np.mean(residual**2)\n\n"
        f"print(f'Filter: Butterworth 4th order, cutoff {cutoff} Hz')\n"
        f"print(f'SNR before filter: {{snr_db(sig_power, noise_before):.1f}} dB')\n"
        f"print(f'SNR after filter:  {{snr_db(sig_power, max(noise_after, 1e-10)):.1f}} dB')"
    )
    return query, think, "", code


def gen_dft_by_hand(rng):
    """Compute DFT of a small sequence manually"""
    n = rng.choice([4, 8])
    if n == 4:
        seq = [rng.choice([0, 1, 2, 3]) for _ in range(4)]
        query = f"Compute the Discrete Fourier Transform of the sequence {seq}."
        think = "DFT: X[k] = Σ x[n] · e^(-j·2π·k·n/N) for k=0..N-1. Compute magnitude and phase."
        code = (
            f"import numpy as np\n\n"
            f"x = np.array({seq}, dtype=complex)\n"
            f"N = len(x)\n"
            f"X = np.fft.fft(x)\n\n"
            f"print('DFT coefficients:')\n"
            f"for k, val in enumerate(X):\n"
            f"    mag = abs(val)\n"
            f"    phase = np.angle(val, deg=True)\n"
            f"    print(f'  X[{{k}}] = {{val:.3f}}  |X|={{mag:.3f}}  phase={{phase:.1f}}°')"
        )
    else:
        freq = rng.choice([1, 2])
        seq_code = f"np.sin(2 * np.pi * {freq} * np.arange(8) / 8)"
        query = f"Compute the DFT of an 8-point sequence: x[n] = sin(2π·{freq}·n/8). Which frequency bin has the peak?"
        think = f"An {freq}-cycle sinusoid sampled at 8 points will have a peak at bin {freq} of the DFT."
        code = (
            f"import numpy as np\n\n"
            f"x = np.sin(2 * np.pi * {freq} * np.arange(8) / 8)\n"
            f"X = np.fft.fft(x)\n"
            f"magnitudes = np.abs(X)\n\n"
            f"print('DFT magnitudes:')\n"
            f"for k, m in enumerate(magnitudes):\n"
            f"    bar = '#' * int(m * 2)\n"
            f"    print(f'  bin {{k}}: {{m:.3f}} {{bar}}')\n"
            f"print(f'Peak at bin: {{np.argmax(magnitudes[:len(X)//2+1])}}')"
        )
    return query, think, "", code


def gen_power_spectral_density(rng):
    """Compute and interpret PSD"""
    sample_rate = rng.choice([200, 500])
    freqs = sorted(rng.sample([5, 10, 20, 30, 50], 2))
    query = (
        f"A signal contains components at {freqs[0]} Hz and {freqs[1]} Hz (sample rate {sample_rate} Hz). "
        f"Compute its power spectral density using Welch's method."
    )
    think = (
        "Welch's method estimates the PSD by averaging periodograms of overlapping segments. "
        "Peaks in the PSD show where the signal's power is concentrated."
    )
    code = (
        f"import numpy as np\n"
        f"from scipy import signal as sp\n\n"
        f"fs = {sample_rate}\n"
        f"t = np.linspace(0, 4, 4*fs, endpoint=False)\n"
        f"sig = np.sin(2*np.pi*{freqs[0]}*t) + 0.7*np.sin(2*np.pi*{freqs[1]}*t)\n"
        f"sig += 0.1 * np.random.default_rng(42).normal(size=len(t))\n\n"
        f"freqs_psd, psd = sp.welch(sig, fs=fs, nperseg=256)\n\n"
        f"# Find top 3 PSD peaks\n"
        f"peaks, _ = sp.find_peaks(psd, height=0.01)\n"
        f"top = sorted(peaks, key=lambda i: -psd[i])[:3]\n"
        f"print('Top PSD peaks:')\n"
        f"for i in top:\n"
        f"    print(f'  {{freqs_psd[i]:.1f}} Hz  power={{psd[i]:.4f}} V²/Hz')"
    )
    return query, think, "", code


def gen_multistep_signal_analysis(rng):
    """Full pipeline: generate → FFT → filter → verify"""
    fs = rng.choice([500, 1000])
    sig_f = rng.choice([10, 20, 25])
    noise_f = rng.choice([100, 150, 200])
    cutoff = rng.choice([40, 50, 60])

    query = (
        f"Analyze a signal at {sig_f} Hz with noise at {noise_f} Hz (fs={fs} Hz): "
        f"(1) identify frequencies via FFT, (2) apply low-pass filter at {cutoff} Hz, "
        f"(3) confirm noise is removed."
    )
    think = "Three steps: FFT to see what's there, design and apply filter, FFT again to verify."

    exec_shared = PythonExecutor()
    code1 = (
        f"import numpy as np\n"
        f"from scipy import signal as sp\n\n"
        f"fs = {fs}\n"
        f"t = np.linspace(0, 2, 2*fs, endpoint=False)\n"
        f"sig = np.sin(2*np.pi*{sig_f}*t) + 0.6*np.sin(2*np.pi*{noise_f}*t)\n\n"
        f"fft_vals = np.fft.rfft(sig)\n"
        f"freqs = np.fft.rfftfreq(len(sig), 1/fs)\n"
        f"mags = np.abs(fft_vals)*2/len(sig)\n"
        f"peaks = [(round(freqs[i],1), round(mags[i],3)) for i in np.where(mags > 0.1)[0]]\n"
        f"peaks.sort(key=lambda x: -x[1])\n"
        f"print('Step 1 - Frequencies found:')\n"
        f"for f, a in peaks[:4]:\n"
        f"    print(f'  {{f}} Hz  amp={{a}}')"
    )
    out1 = exec_shared.run(code1).output_text()

    code2 = (
        f"b, a = sp.butter(4, {cutoff}, btype='low', fs=fs)\n"
        f"filtered = sp.filtfilt(b, a, sig)\n\n"
        f"fft2 = np.fft.rfft(filtered)\n"
        f"mags2 = np.abs(fft2)*2/len(filtered)\n"
        f"peaks2 = [(round(freqs[i],1), round(mags2[i],3)) for i in np.where(mags2 > 0.05)[0]]\n"
        f"peaks2.sort(key=lambda x: -x[1])\n"
        f"print('Step 2 - After {cutoff} Hz low-pass filter:')\n"
        f"for f, a in peaks2[:4]:\n"
        f"    print(f'  {{f}} Hz  amp={{a}}')\n"
        f"noise_remaining = mags2[np.argmin(np.abs(freqs - {noise_f}))]\n"
        f"print(f'Noise at {noise_f} Hz remaining: {{noise_remaining:.4f}}')"
    )
    out2 = exec_shared.run(code2).output_text()

    response = (
        f"<think>\n{think}\n</think>\n"
        f"<code>\n{code1.strip()}\n</code>\n"
        f"<output>\n{out1.strip()}\n</output>\n"
        f"<think>\nNow apply the filter and verify noise removal.\n</think>\n"
        f"<code>\n{code2.strip()}\n</code>\n"
        f"<output>\n{out2.strip()}\n</output>\n"
        f"The {sig_f} Hz signal is preserved while the {noise_f} Hz noise is attenuated."
    )
    return {"id": None, "category": "fourier", "difficulty": "advanced",
            "query": query, "response": response}


# ─── Main ─────────────────────────────────────────────────────────────────────

GENERATORS = [
    ("identify_freq",   "intermediate", gen_identify_frequencies,    0.25),
    ("nyquist",         "basic",        gen_nyquist,                 0.15),
    ("lowpass",         "intermediate", gen_lowpass_filter,          0.20),
    ("dft",             "basic",        gen_dft_by_hand,             0.15),
    ("psd",             "intermediate", gen_power_spectral_density,  0.15),
]


def generate_examples(count: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    examples = []
    idx = 0

    multistep_count = max(1, count // 8)
    single_count = count - multistep_count
    total_w = sum(g[3] for g in GENERATORS)

    for _ in range(single_count):
        r = rng.random()
        cumulative = 0.0
        chosen = GENERATORS[0]
        for gen in GENERATORS:
            cumulative += gen[3] / total_w
            if r <= cumulative:
                chosen = gen
                break
        name, difficulty, fn = chosen[0], chosen[1], chosen[2]
        try:
            query, think, model_text, code = fn(rng)
            ex = make_example(f"fourier_{idx:04d}", "fourier", difficulty,
                               query, think, model_text, code)
            examples.append(ex)
            idx += 1
        except Exception as e:
            print(f"  Warning: skipping {name} ({e})")

    for _ in range(multistep_count):
        try:
            ex = gen_multistep_signal_analysis(rng)
            ex["id"] = f"fourier_{idx:04d}"
            examples.append(ex)
            idx += 1
        except Exception as e:
            print(f"  Warning: skipping multistep ({e})")

    rng.shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} Fourier analysis examples...")
    examples = generate_examples(args.count, seed=args.seed)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Written {len(examples)} examples to {output_path}")
    errors = sum(1 for ex in examples if "<code>" in ex["response"] and "<output>" not in ex["response"])
    print("All examples validated OK." if errors == 0 else f"WARNING: {errors} missing outputs")


if __name__ == "__main__":
    main()
