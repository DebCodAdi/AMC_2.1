"""Canonical modulation label vocabulary shared by every training dataset.

Why this exists
---------------
Each dataset loader used to invent its own mapping from modulation name to
class index -- the synthetic set used 6 classes, the RadioML 2016 loader an
ad-hoc 13-name dictionary, and the RadioML 2018 loader the raw one-hot position
of the HDF5 file -- and all three fed the same output head.  Index 3 therefore
meant "16QAM" for one dataset and "BPSK" for another, a checkpoint did not
record which space it was trained in, and nothing could interpret its output.
Two specific label-corruption bugs followed from the ad-hoc maps:

* RadioML 2016's ``PAM4`` was missing from the map, so ``dict.get(name, 0)``
  silently labelled every PAM4 example as BPSK (1 of the 11 classes wrong).
* Any name a map did not know fell back to index 0 instead of being rejected.

Now every loader maps *names* into :data:`MOD_VOCAB`, unknown names raise, and
the class list is written into every checkpoint so inference decodes indices
with the vocabulary the model was trained on.
"""
from __future__ import annotations

# Canonical vocabulary.  Order is part of the checkpoint format; append only.
MOD_VOCAB: tuple[str, ...] = (
    # Linear -- PSK
    "BPSK", "QPSK", "OQPSK", "8PSK", "16PSK", "32PSK",
    # Linear -- APSK / QAM
    "16APSK", "32APSK", "64APSK", "128APSK",
    "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
    # Amplitude keying / PAM
    "OOK", "4ASK", "8ASK", "PAM4",
    # Frequency / continuous-phase
    "2FSK", "4FSK", "8FSK", "CPFSK", "GFSK", "GMSK", "MSK",
    # Analog
    "AM-DSB", "AM-DSB-WC", "AM-DSB-SC", "AM-SSB", "AM-SSB-WC", "AM-SSB-SC",
    "FM", "WBFM",
    # No signal
    "NOISE",
)

_INDEX = {name: i for i, name in enumerate(MOD_VOCAB)}

# Spellings used by public datasets that differ from the canonical names.
_ALIASES = {
    "QAM16": "16QAM", "QAM32": "32QAM", "QAM64": "64QAM",
    "QAM128": "128QAM", "QAM256": "256QAM",
    "PSK8": "8PSK", "8-PSK": "8PSK", "BFSK": "2FSK", "FSK": "2FSK",
    "4-FSK": "4FSK", "2-FSK": "2FSK", "NOISE_ONLY": "NOISE",
}

# RadioML 2018.01A (GOLD_XYZ_OSC.0001_1024.hdf5): position i of the one-hot Y
# vector is RML2018_CLASSES_FIXED[i].  This is the order of DeepSig's
# classes-fixed file; the classes.txt shipped with the original release lists a
# different order that does not match the labels in the HDF5 (confirmed by
# independent loaders, e.g. RCSL-TCD/ITU-RadioML and radioML/dataset issue #25).
RML2018_CLASSES_FIXED: tuple[str, ...] = (
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK",
    "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
    "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC", "FM", "GMSK", "OQPSK",
)

# RadioML 2016.10a has these 11 classes (2016.10b drops AM-SSB).
RML2016_CLASSES: tuple[str, ...] = (
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
    "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
)

# Coarse family of each class, used for the L1 level of the report.
FAMILY = {
    **{m: "PSK" for m in ("BPSK", "QPSK", "OQPSK", "8PSK", "16PSK", "32PSK")},
    **{m: "APSK" for m in ("16APSK", "32APSK", "64APSK", "128APSK")},
    **{m: "QAM" for m in ("16QAM", "32QAM", "64QAM", "128QAM", "256QAM")},
    **{m: "ASK" for m in ("OOK", "4ASK", "8ASK", "PAM4")},
    **{m: "FSK" for m in ("2FSK", "4FSK", "8FSK", "CPFSK", "GFSK", "GMSK", "MSK")},
    **{m: "AM" for m in ("AM-DSB", "AM-DSB-WC", "AM-DSB-SC", "AM-SSB", "AM-SSB-WC", "AM-SSB-SC")},
    **{m: "FM" for m in ("FM", "WBFM")},
    "NOISE": "NOISE",
}


def canonical_mod(name: str | bytes) -> str:
    """Canonical spelling of a modulation name; raises ValueError if unknown."""
    if isinstance(name, bytes):
        name = name.decode("latin1")
    key = str(name).strip().upper()
    key = _ALIASES.get(key, key)
    if key not in _INDEX:
        raise ValueError(f"unknown modulation name {name!r}; add it to signal9.train.labels")
    return key


def mod_index(name: str | bytes) -> int:
    """Index of ``name`` in :data:`MOD_VOCAB` (raises ValueError if unknown)."""
    return _INDEX[canonical_mod(name)]


def mod_name(index: int, class_names: tuple[str, ...] | list[str] = MOD_VOCAB) -> str:
    return class_names[int(index)]
