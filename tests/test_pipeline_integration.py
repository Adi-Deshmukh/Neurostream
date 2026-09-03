import numpy as np
import mne

from neurostream.data.preprocessing import preprocess_session

sfreq = 250.0
info = mne.create_info(22, sfreq, ch_types="eeg")

raw = mne.io.RawArray(
    np.random.default_rng(4).normal(size=(22, 2200)),
    info,
    verbose=False,
)

raw.set_annotations(
    mne.Annotations(
        onset=[0.4],
        duration=[0],
        description=["769"],
    )
)

X, y, metadata = preprocess_session(raw)

print("Shape:", X.shape)
print("Data type:", X.dtype)
print("Labels:", y)
print("Metadata:", metadata)
print("Channel means:", X.mean(axis=-1))
print("Channel standard deviations:", X.std(axis=-1))