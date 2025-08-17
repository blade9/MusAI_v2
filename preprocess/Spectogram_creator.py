import librosa
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def generateSpectrogram(wav_path):
    my_sr = 22050
    my_n_fft = 2048
    y, sr = librosa.load(wav_path, sr=my_sr)

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=my_n_fft, hop_length=512, n_mels=128
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = 2 * (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db)) - 1

    return mel_spec_db


def segmentSpectogram(spectrogram, segment_frames=256, hop_frames=128):
    segments = []
    n_frames = spectrogram.shape[1]

    for start in range(0, n_frames - segment_frames + 1, hop_frames):
        segment = spectrogram[:, start:start + segment_frames]
        segments.append({'start': start, 'end': start+segment_frames, 'data':segment})
    return segments


def visualizeSpectogram(spectrogram, sr=22050, hop_length = 512):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(
        spectrogram,
        hop_length=hop_length, x_axis='time', y_axis='mel'
    )
    plt.colorbar()
    plt.title('Mel Spectrogram')
    plt.show()

def createFolders(input_path, output_path):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Generate spectrogram
    spec = generateSpectrogram(input_path)
    segments = segmentSpectogram(spec)
    for seg in segments:
        # Create segment folder (e.g., "segment_0_256")
        seg_name = f"segment_{seg['start']}_{seg['end']}"
        seg_path = os.path.join(output_path, seg_name)
        os.makedirs(seg_path, exist_ok=True)

        # Save segment data
        np.save(os.path.join(seg_path, "spectrogram.npy"), seg['data'])

        # Save visualization
        '''
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(
            seg['data'],
            hop_length=512,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar()
        plt.title(f'Spectrogram {seg_name}')
        plt.savefig(os.path.join(seg_path, "spectrogram.png"))
        plt.close()
        '''

wavpath = "data/musicnet/musicnet/train_data/"
song_name = '1751.wav'
song_list = [
    "1733.wav", "1734.wav", "1749.wav", "1752.wav",
    "1755.wav", "1756.wav", "1757.wav", "1758.wav", "1760.wav", "1763.wav",
    "1764.wav", "1765.wav", "1766.wav", "1768.wav", "1771.wav", "1772.wav", "1773.wav",
    "1775.wav", "1776.wav", "1777.wav", "2194.wav", "2195.wav", "2196.wav", "2198.wav",
    "2200.wav", "2201.wav", "2207.wav", "2208.wav", "2209.wav", "2210.wav", "2211.wav",
    "2212.wav", "2213.wav", "2214.wav", "2215.wav", "2224.wav", "2225.wav", "2227.wav",
    "2228.wav", "2229.wav", "2230.wav", "2231.wav", "2232.wav", "2234.wav", "2237.wav",
    "2238.wav", "2239.wav", "2240.wav", "2247.wav", "2248.wav", "2292.wav", "2300.wav",
    "2302.wav", "2304.wav", "2305.wav", "2307.wav", "2308.wav", "2310.wav",
    "2322.wav", "2325.wav", "2343.wav", "2345.wav", "2346.wav", "2348.wav", "2350.wav",
    "2357.wav", "2358.wav", "2359.wav", "2364.wav", "2371.wav", "2372.wav", "2373.wav",
    "2374.wav", "2388.wav", "2389.wav", "2390.wav", "2391.wav", "2392.wav", "2393.wav",
    "2404.wav", "2405.wav", "2406.wav", "2410.wav", "2411.wav", "2422.wav", "2423.wav",
    "2424.wav", "2436.wav", "2441.wav", "2442.wav", "2443.wav", "2444.wav", "2471.wav",
    "2472.wav", "2473.wav", "2476.wav", "2477.wav", "2478.wav", "2486.wav", "2487.wav",
    "2488.wav", "2490.wav", "2491.wav", "2492.wav", "2509.wav", "2510.wav", "2512.wav",
    "2514.wav", "2516.wav", "2527.wav", "2528.wav", "2529.wav", "2530.wav", "2531.wav",
    "2532.wav", "2533.wav", "2537.wav", "2538.wav", "2540.wav", "2542.wav", "2550.wav",
    "2555.wav", "2557.wav"
]
for element in song_list:
    wavpath += element
    output_path = "spectrograms_" + element
    createFolders(wavpath, output_path)
    wavpath = "data/musicnet/musicnet/train_data/"

'''
wavpath += song_name
output_path = "spectrograms_" + song_name
createFolders(wavpath, output_path)
'''
print("Ran")
