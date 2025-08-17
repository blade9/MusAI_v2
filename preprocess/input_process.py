import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt


def get_pairs(data_dir):
    segment_dirs = sorted(Path(data_dir).iterdir(),
                          key=lambda x: int(x.name.split("_")[1]))  # Sort by start frame
    return [
        (d / "spectrogram.npy", d / "labels.npy")
        for d in segment_dirs
        if d.is_dir()
    ]


def create_batches(pairs, batch_size=32):
    """Yields (spectrograms, labels) batches"""
    while True:
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_specs = np.array([np.load(spec).T for spec, _ in batch_pairs])
            batch_labels = np.array([np.load(label) for _, label in batch_pairs])
            yield batch_specs.astype('float32'), batch_labels.astype('float32')


# Usage:
data_pairs = get_pairs("spectrograms_1735.wav")
train_batches = create_batches(data_pairs, batch_size=32)






'''
def create_ordered_dataset(data_dir, batch_size=32):
    """
    Creates order-preserving dataset for single-song segments.
    Assumes segment folders are named/numbered sequentially.
    """
    segment_dirs = [d for d in Path(data_dir).iterdir()
                    if d.is_dir() and d.name.startswith("segment_")]
    file_pairs = [
        (str(d / "spectrogram.npy"), str(d / "labels.npy"))
        for d in segment_dirs
        if (d / "spectrogram.npy").exists() and (d / "labels.npy").exists()
    ]
    print(len(file_pairs))

    if not file_pairs:
        raise FileNotFoundError(f"No valid pairs in {data_dir}")

    def load_pair(spec_path, label_path):
        spectrogram = tf.cast(tf.numpy_function(np.load, [spec_path], tf.float32), tf.float32)
        label = tf.cast(tf.numpy_function(np.load, [label_path], tf.float32), tf.float32)
        return spectrogram, label

        # 4. Build dataset

    return (
        tf.data.Dataset.from_tensor_slices(file_pairs)
        .map(lambda x: load_pair(x[0], x[1]), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
'''
if __name__ == "__main__":
    pairs = get_pairs("./spectrograms_1735.wav")
    train_batches = create_batches(pairs, batch_size=32)
    total_samples = 0

    for X_batch, y_batch in train_batches:
        try:
            assert X_batch.shape[1:] == (256, 128)
            assert y_batch.shape[1:] == (88,)
        except AssertionError as e:
            print(f"X_batch shape: {X_batch.shape}")
            print(f"y_batch shape: {y_batch.shape}")
            plt.imshow(X_batch[0].T, aspect='auto', origin='lower')
            plt.colorbar(label='Magnitude')
            plt.title("First Spectrogram in Failed Batch (Frequency vs Time)")
            plt.xlabel("Time (frames)")
            plt.ylabel("Frequency (bins)")
            plt.show()
        total_samples += len(X_batch)

    print(f"Total segments processed: {total_samples}")

