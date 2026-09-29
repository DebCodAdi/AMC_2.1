"""NCO downconversion, channelization, and SigMF annotation for detected bursts (Phase 3).

Extracts detected emissions from wideband captures, shifts to baseband via NCO,
resamples to native channel bandwidth, and formats as SigMF annotations.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal

from .cluster import SignalBurst


@dataclass(frozen=True)
class BurstExtraction:
    burst_id: int
    channel_iq: np.ndarray       # Baseband channelized complex samples
    channel_fs: float            # Downsampled channel sample rate
    f_center_hz: float           # Center frequency relative to wideband DC
    duration_s: float
    snr_db: float
    sigmf_metadata: dict         # Standard SigMF format dictionary


def extract_burst(wb_iq: np.ndarray,
                  fs_wb: float,
                  fc_rf: float,
                  burst: SignalBurst,
                  margin_time_s: float = 0.001,
                  target_oversample: float = 2.0) -> BurstExtraction:
    """Extract, downconvert, and decimate a single detected signal burst.

    Parameters
    ----------
    wb_iq : np.ndarray
        Wideband IQ signal.
    fs_wb : float
        Wideband sample rate in Hz.
    fc_rf : float
        SDR center RF frequency in Hz.
    burst : SignalBurst
        Detected burst metadata.
    margin_time_s : float, default 1 ms
        Temporal margin added before and after burst.
    target_oversample : float, default 2.0
        Oversampling factor relative to burst bandwidth for channelized rate.

    Returns
    -------
    BurstExtraction
    """
    N = len(wb_iq)
    t0_s = max(0.0, burst.t_start_s - margin_time_s)
    t1_s = min(N / fs_wb, burst.t_end_s + margin_time_s)

    idx_start = int(np.floor(t0_s * fs_wb))
    idx_end = int(np.ceil(t1_s * fs_wb))
    idx_end = max(idx_start + 16, min(N, idx_end))

    sliced_iq = wb_iq[idx_start:idx_end]
    n_samples = len(sliced_iq)

    # 1. NCO Frequency Shift: rotate by -f_center
    t_vec = (np.arange(n_samples, dtype=np.float64) + idx_start) / fs_wb
    nco = np.exp(-1j * 2.0 * np.pi * burst.f_center_hz * t_vec).astype(np.complex64)
    baseband_shifted = sliced_iq * nco

    # 2. Resample / Decimate to channel sample rate
    target_fs = max(4000.0, burst.bandwidth_hz * target_oversample)
    # Determine integer or rational decimation factor
    if target_fs < fs_wb * 0.9:
        # Decimate using polyphase FIR
        decim_factor = max(1, int(np.floor(fs_wb / target_fs)))
        ch_iq = signal.decimate(baseband_shifted, decim_factor, ftype="fir", zero_phase=True).astype(np.complex64)
        actual_ch_fs = fs_wb / decim_factor
    else:
        ch_iq = baseband_shifted
        actual_ch_fs = fs_wb

    # 3. Assemble SigMF metadata dictionary
    sigmf_meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": float(actual_ch_fs),
            "core:version": "1.0.0",
            "core:description": f"Extracted burst {burst.burst_id} via Stage 0 Survey",
            "core:recorder": "Signal9 Survey Engine",
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": float(fc_rf + burst.f_center_hz),
                "core:datetime": "2026-09-19T00:00:00Z",
            }
        ],
        "annotations": [
            {
                "core:sample_start": 0,
                "core:sample_count": len(ch_iq),
                "core:freq_lower_edge": float(fc_rf + burst.f_lo_hz),
                "core:freq_upper_edge": float(fc_rf + burst.f_hi_hz),
                "core:label": f"burst_{burst.burst_id}",
                "signal9:snr_db": float(burst.snr_db_est),
                "signal9:bandwidth_hz": float(burst.bandwidth_hz),
            }
        ],
    }

    return BurstExtraction(
        burst_id=burst.burst_id,
        channel_iq=ch_iq,
        channel_fs=actual_ch_fs,
        f_center_hz=burst.f_center_hz,
        duration_s=burst.duration_s,
        snr_db=burst.snr_db_est,
        sigmf_metadata=sigmf_meta,
    )
