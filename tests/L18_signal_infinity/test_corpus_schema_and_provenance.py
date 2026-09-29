"""Tests for Grand Corpus schema, provenance validation (Rule R17) and diversity (Rule R22)."""
import pytest
import numpy as np
from signal9.corpus.schema import CorpusSample, SampleProvenance, DUCKDB_SCHEMA_SQL
from tools.corpus.diversity import compute_diversity_metrics, audit_diversity_floor


def test_provenance_dataclass_fields():
    prov = SampleProvenance(
        source="A_real",
        generation=0,
        labelled_by="crc_oracle",
        verified_by="crc32",
        session_id="sess_001",
        device_id="rtlsdr_v4_01"
    )
    d = prov.to_dict()
    assert d["source"] == "A_real"
    assert d["generation"] == 0
    assert d["verified_by"] == "crc32"
    assert "added_at" in d


def test_corpus_duckdb_indexing(tmp_path):
    pytest.importorskip("duckdb")
    from tools.corpus.index import init_database, index_samples
    db_file = tmp_path / "test_corpus.duckdb"
    con = init_database(db_file)
    
    prov = SampleProvenance("C_gen", 1, "construction", "construction", "s1", "dev1")
    sample = CorpusSample(
        sample_id="samp_001",
        bucket="C",
        filepath="shards/C/001.raw",
        offset_samples=0,
        length_samples=4096,
        fs=48000.0,
        fc=144800000.0,
        mod="QPSK",
        snr_db=15.0,
        symrate=2400.0,
        rolloff=0.35,
        cfo_hz=12.5,
        fec="none",
        scrambler="none",
        framing="raw",
        is_positive=True,
        provenance=prov
    )

    n_indexed = index_samples(con, [sample])
    assert n_indexed == 1
    
    row = con.execute("SELECT sample_id, mod, snr_db, prov_verified_by FROM samples WHERE sample_id='samp_001'").fetchone()
    assert row[0] == "samp_001"
    assert row[1] == "QPSK"
    assert row[2] == 15.0
    assert row[3] == "construction"
    con.close()


def test_diversity_floor_enforcement():
    samples_gen0 = [
        {"mod": "BPSK", "snr_db": 10.0, "prov_device_id": "dev1", "fec": "none", "scrambler": "none"},
        {"mod": "QPSK", "snr_db": 20.0, "prov_device_id": "dev2", "fec": "conv", "scrambler": "g3ruh"},
    ]
    metrics0 = compute_diversity_metrics(samples_gen0)
    assert metrics0["entropy_mod"] > 0.0

    # Diversity non-decreasing should pass
    passed, viols = audit_diversity_floor(metrics0, metrics0)
    assert passed
    assert len(viols) == 0

    # Diversity decreasing should fail (Rule R22)
    degraded_metrics = dict(metrics0)
    degraded_metrics["entropy_mod"] = metrics0["entropy_mod"] - 0.5
    passed_deg, viols_deg = audit_diversity_floor(degraded_metrics, metrics0)
    assert not passed_deg
    assert "Rule R22 violation" in viols_deg[0]
