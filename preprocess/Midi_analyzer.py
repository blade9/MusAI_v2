import tensorflow as tf
import numpy as np
import librosa
import pretty_midi

def load_audio_to_spectrogram(wav_path, sr=16000, hop_length=512, n_mels=128):
    y, _ = librosa.load(wav_path, sr=sr)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_S = librosa.power_to_db(S)
    return log_S.T  # (frames, n_mels)

def split_into_segments(spectrogram, segment_length=256):
    segments = []
    for i in range(0, len(spectrogram) - segment_length + 1, segment_length):
        segment = spectrogram[i:i+segment_length]
        segments.append(segment)
    return np.array(segments)

def predict_pianoroll(model, segments, threshold=0.5):
    predictions = model.predict(segments)
    binary_rolls = predictions > threshold
    return np.vstack(binary_rolls)  # shape: (total_frames, 88)

def pianoroll_to_midi(pianoroll, fs=100):
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    for pitch in range(88):
        note_on = None
        for t, val in enumerate(pianoroll[:, pitch]):
            if val and note_on is None:
                note_on = t
            elif not val and note_on is not None:
                start = note_on / fs
                end = t / fs
                note = pretty_midi.Note(velocity=80, pitch=pitch + 21, start=start, end=end)
                piano.notes.append(note)
                note_on = None
        if note_on is not None:
            # End last note if it runs off the end
            note = pretty_midi.Note(velocity=80, pitch=pitch + 21, start=note_on / fs, end=len(pianoroll) / fs)
            piano.notes.append(note)
    pm.instruments.append(piano)
    return pm
