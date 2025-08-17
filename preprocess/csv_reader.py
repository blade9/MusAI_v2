import numpy as np
import pandas as pd
import os
import Spectogram_creator
from pathlib import Path

def findPiano():
    try:
        df = pd.read_csv('data\musicnet_metadata.csv')
    except FileNotFoundError:
        return "File not found"

    piano_ids = [row["id"] for _, row in df.iterrows() if "solo piano" in str(row["ensemble"]).lower()]
    return piano_ids


def label_prep(csv_path, segment_root_dir, n_frames, hop_length=512, sr=44100, n_notes=88):

    sr_multiplier = 22050/sr
    labels = np.zeros((n_frames, n_notes), dtype=np.float32)
    onset = np.zeros_like(labels)
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        midi_note = max(0, int(row["note"]) - 21)
        start_sample = int(row["start_time"] * sr_multiplier)
        end_sample = int(row["end_time"] * sr_multiplier)

        if row["instrument"] > 1:
            continue

        start_frame = int(start_sample / hop_length)
        end_frame = int(end_sample / hop_length)
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)

        labels[start_frame:end_frame, midi_note] = 1.0
        onset[start_frame, midi_note] = 1.0

    for seg_dir in Path(segment_root_dir).glob("segment_*"):
        start, end = map(int, seg_dir.name.split("_")[1:3])

        seg_labels = labels[start:end]
        np.save(seg_dir/"labels", seg_labels)
        print(start, end)

    return labels

def split_labels(labels, segment_length):
    n_frames = labels.shape[0]
    segments = []



wavpath = "data/musicnet/musicnet/train_data/"
csvpath = "data/musicnet/musicnet/train_labels/"


song_list = [
    "1733", "1734", "1749", "1752",
    "1755", "1756", "1757", "1758", "1760", "1763",
    "1764", "1765", "1766", "1768", "1771", "1772", "1773",
    "1775", "1776", "1777", "2194", "2195", "2196", "2198",
    "2200", "2201", "2207", "2208", "2209", "2210", "2211",
    "2212", "2213", "2214", "2215", "2224", "2225", "2227",
    "2228", "2229", "2230", "2231", "2232", "2234", "2237",
    "2238", "2239", "2240", "2247", "2248", "2292", "2300",
    "2302", "2304", "2305", "2307", "2308", "2310",
    "2322", "2325", "2343", "2345", "2346", "2348", "2350",
    "2357", "2358", "2359", "2364", "2371", "2372", "2373",
    "2374", "2388", "2389", "2390", "2391", "2392", "2393",
    "2404", "2405", "2406", "2410", "2411", "2422", "2423",
    "2424", "2436", "2441", "2442", "2443", "2444", "2471",
    "2472", "2473", "2476", "2477", "2478", "2486", "2487",
    "2488", "2490", "2491", "2492", "2509", "2510", "2512",
    "2514", "2516", "2527", "2528", "2529", "2530", "2531",
    "2532", "2533", "2537", "2538", "2540", "2542", "2550",
    "2555", "2557"
]

for element in song_list:
    wavpath += element
    wavpath += ".wav"
    csvpath += element
    csvpath += ".csv"
    output_path = "spectrograms_" + element + ".wav"
    spect = Spectogram_creator.generateSpectrogram(wavpath)
    label = label_prep(csvpath, output_path, spect.shape[1])
    np.savetxt("labels_output.txt", label, fmt="%d")
    wavpath = "data/musicnet/musicnet/train_data/"
    csvpath = "data/musicnet/musicnet/train_labels/"



