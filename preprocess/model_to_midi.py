#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────────────────
#  transcribe_simple.py      –  inference for the “frames‑only” LSTM model
#
#  Usage:
#     python transcribe_simple.py --wav path/to/audio.wav \
#                                 --model piano_transcriber.keras \
#                                 --out  audio.mid
# ────────────────────────────────────────────────────────────────────────────────
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import pretty_midi as pm
from tqdm import tqdm

# Your spectrogram helper (exactly the one used for training!)
from Spectogram_creator import generateSpectrogram   # noqa


# ────────────────────────────────────────────────────────────────────────────────
# 1.  Parameters  — must match training time
# ────────────────────────────────────────────────────────────────────────────────
MEL_BINS       = 128
BLOCK_FRAMES   = 256                # model input length (time)
SR             = 22_050             # audio rate in Spectogram_creator
HOP_LENGTH     = 512                # hop in samples → 100 fps
ON_THRES       = 0.9               # prob rises above → note ON
OFF_THRES      = 0.8               # prob falls below → note OFF (hysteresis)


# ────────────────────────────────────────────────────────────────────────────────
# 2.  Frame‑probabilities → list[pretty_midi.Note]
# ────────────────────────────────────────────────────────────────────────────────
def frames_to_notes(frames, on_th=ON_THRES, off_th=OFF_THRES):
    """
    Parameters
    ----------
    frames : (T, 88) float32
        Per‑frame probabilities from the network (sigmoid outputs).
    """
    T, P = frames.shape
    notes = []

    for pitch in range(P):
        active = False
        t_on = 0
        for t in range(T):
            p = frames[t, pitch]
            if not active and p >= on_th:
                active = True
                t_on = t
            elif active and p < off_th:
                # close the note
                start_sec = t_on * HOP_LENGTH / SR
                end_sec   = max(start_sec + 0.05, t * HOP_LENGTH / SR)
                velocity  = int(np.clip(frames[t_on:t, pitch].max() * 127, 1, 127))
                notes.append(pm.Note(velocity=velocity,
                                     pitch=pitch + 21,
                                     start=start_sec,
                                     end=end_sec))
                active = False
        # tail note to the end
        if active:
            start_sec = t_on * HOP_LENGTH / SR
            end_sec   = (T * HOP_LENGTH) / SR
            velocity  = int(np.clip(frames[t_on:, pitch].max() * 127, 1, 127))
            notes.append(pm.Note(velocity=velocity,
                                 pitch=pitch + 21,
                                 start=start_sec,
                                 end=end_sec))
    return notes


# ────────────────────────────────────────────────────────────────────────────────
# 3.  End‑to‑end transcription routine
# ────────────────────────────────────────────────────────────────────────────────
def transcribe(model_path: Path, wav_path: Path, out_path: Path):
    # 3‑a  Load model (compiled=False because we only run inference)
    model = tf.keras.models.load_model(model_path, compile=False)

    # 3‑b  Spectrogram  – returns (128 mel × N frames), already log‑scaled & −1…1
    S = generateSpectrogram(str(wav_path))
    if S.shape[0] != MEL_BINS:
        raise ValueError(f"Spectrogram mel bins ({S.shape[0]}) != expected {MEL_BINS}")
    S = S.T.astype(np.float32)                      # (T,128)

    # 3‑c  Pad to multiple of BLOCK_FRAMES
    pad = (-len(S)) % BLOCK_FRAMES
    if pad:
        S = np.pad(S, ((0, pad), (0, 0)), mode='constant')

    n_blocks = len(S) // BLOCK_FRAMES
    frame_prob = np.zeros((len(S), 88), dtype=np.float32)

    # 3‑d  Block‑wise prediction (batch size = 1)
    for i in tqdm(range(n_blocks), desc="Predicting"):
        block = S[i*BLOCK_FRAMES:(i+1)*BLOCK_FRAMES]         # (256,128)
        pred  = model.predict(block[None, ...], verbose=0)   # (1,256,88)
        frame_prob[i*BLOCK_FRAMES:(i+1)*BLOCK_FRAMES] = pred[0]

    # remove padding
    frame_prob = frame_prob[:-pad or None]

    # 3‑e  Turn probabilities into MIDI notes
    notes = frames_to_notes(frame_prob)

    # 3‑f  Write MIDI
    midi_out = pm.PrettyMIDI()
    inst = pm.Instrument(program=0, name="Piano")
    inst.notes = notes
    midi_out.instruments.append(inst)
    midi_out.write(str(out_path))
    print(f"✓  {len(notes)} notes written → {out_path}")


# ────────────────────────────────────────────────────────────────────────────────
# 4.  Minimal CLI
# ────────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Transcribe WAV → MIDI (frames‑only model)")
    parser.add_argument("--wav",   required=True, help="input .wav file")
    parser.add_argument("--model", required=True, help="trained .keras/.h5 model")
    parser.add_argument("--out",   default=None, help="output .mid file (optional)")
    args = parser.parse_args()

    wav_path   = Path(args.wav)
    model_path = Path(args.model)
    out_path   = Path(args.out) if args.out else wav_path.with_suffix(".mid")

    transcribe(model_path, wav_path, out_path)


if __name__ == "__main__":
    main()
