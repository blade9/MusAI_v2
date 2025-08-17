import tensorflow as tf
from models.lstm_model import build_trans_model, compile_model
from preprocess.input_process import get_pairs, create_batches

DATA_DIR = "spectrograms_1735.wav"
VAL_DIR = "spectrograms_1749.wav"
SECOND_DIR = "spectrograms_"
ADDITIONAL_SONG = [
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
BATCH_SIZE = 32
EPOCHS = 50
MODEL_SAVE_PATH = "piano_transcriber.keras"


def main():
    pairs = get_pairs(DATA_DIR)
    #pairs += get_pairs(SECOND_DIR + ADDITIONAL_SONG[0])

    for element in ADDITIONAL_SONG:
        pairs += get_pairs(SECOND_DIR + element)

    print(pairs)
    train_gen = create_batches(pairs, BATCH_SIZE)

    val_pairs = get_pairs(VAL_DIR)
    val_gen = create_batches(val_pairs, BATCH_SIZE)

    # Calculate steps per epoch
    steps_per_epoch = len(pairs) // BATCH_SIZE

    model = build_trans_model()
    compile_model(model)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            save_best_only=True,
            monitor='val_pr_auc',
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_recall',
            patience=10,
            mode='max',
            restore_best_weights=True
        )
    ]

    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=val_gen,
        validation_steps=max(1, steps_per_epoch // 5)
    )


if __name__ == "__main__":
    main()