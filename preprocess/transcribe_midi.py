import os
import numpy as np
import tensorflow as tf
import pretty_midi

def load_segments(segment_folder):
    segment_dirs = sorted([
        os.path.join(segment_folder, d) for d in os.listdir(segment_folder)
        if os.path.isdir(os.path.join(segment_folder, d))
    ])
    segments = []
    for seg_dir in segment_dirs:
        seg_path = os.path.join(seg_dir, "spectrogram.npy")
        if os.path.exists(seg_path):
            seg = np.load(seg_path)
            segments.append(seg)
    return segments

def predict_segments(model, segments):
    all_outputs = []
    for seg in segments:
        # Transpose to (frames, features) → (1, 256, 128)
        input_seg = np.expand_dims(seg.T, axis=0).astype(np.float32)
        prediction = model.predict(input_seg, verbose=0)
        all_outputs.append(prediction[0])  # remove batch dimension
    return np.concatenate(all_outputs, axis=0)  # Shape: (N_total_frames, 88)

def piano_roll_to_midi(piano_roll, fs=100, threshold=0.999):
    # fs: frames per second
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    notes_on = np.zeros(88, dtype=bool)
    note_start_times = np.zeros(88)

    for t, frame in enumerate(piano_roll):
        time = t / fs
        for i, pitch_active in enumerate(frame > threshold):
            midi_pitch = i + 21  # MIDI pitch range: 21 to 108
            if pitch_active:
                if not notes_on[i]:
                    notes_on[i] = True
                    note_start_times[i] = time
            else:
                if notes_on[i]:
                    notes_on[i] = False
                    start = note_start_times[i]
                    end = time
                    note = pretty_midi.Note(
                        velocity=90, pitch=midi_pitch,
                        start=start, end=end
                    )
                    piano.notes.append(note)

    # Turn off any remaining notes
    for i in range(88):
        if notes_on[i]:
            pitch = i + 21
            note = pretty_midi.Note(
                velocity=90,
                pitch=pitch,
                start=note_start_times[i],
                end=len(piano_roll)/fs
            )
            piano.notes.append(note)

    midi.instruments.append(piano)
    return midi

def transcribe(song_folder, model_path, output_midi_path):
    print(f"Transcribing from folder: {song_folder}")
    model = tf.keras.models.load_model(model_path)
    segments = load_segments(song_folder)
    if not segments:
        print("No segments found.")
        return

    piano_roll = predict_segments(model, segments)
    midi = piano_roll_to_midi(piano_roll)
    midi.write(output_midi_path)
    print(f"Saved MIDI to {output_midi_path}")

if __name__ == "__main__":
    # Example usage:
    segment_folder = "spectrograms_1751.wav"  # Folder created by your segmenting code
    model_path = "piano_transcriber.keras"  # Trained Keras model
    output_midi = "1751_output.mid"

    transcribe(segment_folder, model_path, output_midi)
