"""Corpus indexer creating manifests/corpus.duckdb (SIGNAL INFINITY §1.2)."""
from __future__ import annotations
import argparse
from pathlib import Path
import duckdb
from signal9.corpus.schema import DUCKDB_SCHEMA_SQL, CorpusSample, SampleProvenance


def init_database(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(p))
    con.execute(DUCKDB_SCHEMA_SQL)
    return con


def index_samples(con: duckdb.DuckDBPyConnection, samples: list[CorpusSample]) -> int:
    if not samples:
        return 0
    records = [s.to_flat_dict() for s in samples]
    keys = list(records[0].keys())
    cols = ", ".join(keys)
    placeholders = ", ".join(["?" for _ in keys])
    sql = f"INSERT OR REPLACE INTO samples ({cols}) VALUES ({placeholders})"
    
    rows = [[r[k] for k in keys] for r in records]
    con.executemany(sql, rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Index Grand Corpus into DuckDB")
    parser.add_argument("--root", type=str, default="data/corpus", help="Corpus root directory")
    parser.add_argument("--out", type=str, default="manifests/corpus.duckdb", help="DuckDB manifest output")
    args = parser.parse_args()

    con = init_database(args.out)
    count = con.execute("SELECT count(*) FROM samples").fetchone()[0]
    print(f"Grand Corpus DuckDB index initialized at {args.out} (currently {count} samples).")
    con.close()


if __name__ == "__main__":
    main()
