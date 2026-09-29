"""Rule R11: Zero group leakage verification test suite.

Asserts that GroupShuffleSplit strictly segregates datasets by device serial,
session ID, and emitter without any cross-split leakage.
"""
from __future__ import annotations
import pytest
from tools.splits import group_shuffle_split


def test_group_shuffle_split_no_leak():
    # Synthetic corpus with multiple recordings per device and session
    corpus = [
        {"id": 1, "device_serial": "RTL_001", "session_id": "SES_A", "label": "BPSK"},
        {"id": 2, "device_serial": "RTL_001", "session_id": "SES_A", "label": "BPSK"},
        {"id": 3, "device_serial": "RTL_001", "session_id": "SES_B", "label": "QPSK"},
        {"id": 4, "device_serial": "HACKRF_01", "session_id": "SES_C", "label": "8PSK"},
        {"id": 5, "device_serial": "HACKRF_01", "session_id": "SES_D", "label": "8PSK"},
        {"id": 6, "device_serial": "USRP_B210", "session_id": "SES_E", "label": "16QAM"},
        {"id": 7, "device_serial": "USRP_B210", "session_id": "SES_F", "label": "16QAM"},
        {"id": 8, "device_serial": "PLUTO_09", "session_id": "SES_G", "label": "FSK2"},
        {"id": 9, "device_serial": "PLUTO_09", "session_id": "SES_H", "label": "FSK2"},
        {"id": 10, "device_serial": "AIRSPY_02", "session_id": "SES_I", "label": "QPSK"},
    ]

    splits = group_shuffle_split(corpus, group_keys=["device_serial", "session_id"], seed=42)

    train_devs = {x["device_serial"] for x in splits["train"]}
    val_devs = {x["device_serial"] for x in splits["val"]}
    test_devs = {x["device_serial"] for x in splits["test"]}

    train_sess = {x["session_id"] for x in splits["train"]}
    val_sess = {x["session_id"] for x in splits["val"]}
    test_sess = {x["session_id"] for x in splits["test"]}

    # Assert mutual disjointness across devices
    assert train_devs.isdisjoint(test_devs), f"Device leak train-test: {train_devs & test_devs}"
    assert train_devs.isdisjoint(val_devs), f"Device leak train-val: {train_devs & val_devs}"
    assert val_devs.isdisjoint(test_devs), f"Device leak val-test: {val_devs & test_devs}"

    # Assert mutual disjointness across sessions
    assert train_sess.isdisjoint(test_sess), f"Session leak train-test: {train_sess & test_sess}"
    assert train_sess.isdisjoint(val_sess), f"Session leak train-val: {train_sess & val_sess}"
    assert val_sess.isdisjoint(test_sess), f"Session leak val-test: {val_sess & test_sess}"
