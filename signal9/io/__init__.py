"""I/O package for Signal 9."""
from .sniff import sniff_dtype, sniff_raw_dtype, RAW_DTYPES
from .readers import read_iq_block, iter_raw_iq
from .wav_io import read_wav, wav_kind
from .sigmf_io import read_sigmf, write_sigmf
from .metadata import resolve_metadata, SignalMetadata

__all__ = [
    "sniff_dtype",
    "sniff_raw_dtype",
    "RAW_DTYPES",
    "read_iq_block",
    "iter_raw_iq",
    "read_wav",
    "wav_kind",
    "read_sigmf",
    "write_sigmf",
    "resolve_metadata",
    "SignalMetadata",
]
