"""Grand Corpus schema and metadata specification (SIGNAL INFINITY Rule R17)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import datetime
import json
from typing import Any


@dataclass
class SampleProvenance:
    """Full auditable provenance record (Rule R17).
    
    Samples lacking full provenance are quarantined, not trained on.
    """
    source: str                 # e.g. 'A_real', 'B_public/sig53', 'C_gen', 'D_mined', 'E_adv', 'F_neg'
    generation: int             # generation index (0 for initial, 1..N for mined)
    labelled_by: str            # 'ground_truth_gen', 'crc_oracle', 'protocol_oracle', 'author'
    verified_by: str            # 'crc32', 'external_decoder', 'construction', 'unverified'
    session_id: str             # unique session identifier
    device_id: str              # capture SDR serial or simulator worker ID
    added_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorpusSample:
    """A single windowed sample in the Grand Corpus."""
    sample_id: str              # unique SHA-256 or UUID
    bucket: str                 # A, B, C, D, E, F
    filepath: str               # relative or absolute path to raw / shard
    offset_samples: int         # sample offset in file
    length_samples: int         # default 4096
    fs: float                   # sample rate
    fc: float                   # center frequency
    mod: str                    # e.g. 'BPSK', 'QPSK', 'QAM16', 'NOISE', 'UNKNOWN'
    snr_db: float               # SNR in dB
    symrate: float              # estimated or exact symbol rate (Baud)
    rolloff: float              # excess bandwidth alpha [0.1, 0.8]
    cfo_hz: float               # carrier frequency offset
    fec: str                    # 'none', 'conv_k7_r12', 'rs255_223', 'ldpc648_324'
    scrambler: str              # 'none', 'g3ruh', 'v35', 'ccsds'
    framing: str                # 'raw', 'hdlc', 'ccsds_asm', 'ism'
    is_positive: bool           # True if actual communication signal, False for Bucket F
    provenance: SampleProvenance

    def to_flat_dict(self) -> dict[str, Any]:
        d = asdict(self)
        prov = d.pop("provenance")
        for k, v in prov.items():
            d[f"prov_{k}"] = v
        return d


DUCKDB_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS samples (
    sample_id VARCHAR PRIMARY KEY,
    bucket VARCHAR NOT NULL,
    filepath VARCHAR NOT NULL,
    offset_samples BIGINT NOT NULL,
    length_samples INTEGER NOT NULL,
    fs DOUBLE NOT NULL,
    fc DOUBLE NOT NULL,
    mod VARCHAR NOT NULL,
    snr_db DOUBLE NOT NULL,
    symrate DOUBLE NOT NULL,
    rolloff DOUBLE NOT NULL,
    cfo_hz DOUBLE NOT NULL,
    fec VARCHAR NOT NULL,
    scrambler VARCHAR NOT NULL,
    framing VARCHAR NOT NULL,
    is_positive BOOLEAN NOT NULL,
    prov_source VARCHAR NOT NULL,
    prov_generation INTEGER NOT NULL,
    prov_labelled_by VARCHAR NOT NULL,
    prov_verified_by VARCHAR NOT NULL,
    prov_session_id VARCHAR NOT NULL,
    prov_device_id VARCHAR NOT NULL,
    prov_added_at VARCHAR NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bucket_mod ON samples(bucket, mod);
CREATE INDEX IF NOT EXISTS idx_snr ON samples(snr_db);
CREATE INDEX IF NOT EXISTS idx_provenance ON samples(prov_generation, prov_verified_by);
"""
