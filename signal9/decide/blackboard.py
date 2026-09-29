"""Multi-hypothesis blackboard arbiter for demodulation and decoding.

Orchestrates multi-branch exploration:
  - 4 global rotations {0, 90, 180, 270}
  - Track H differential demapping (DEF-024 / IV-L)
  - Bit-order {msb, lsb} and IQ-order {i_first, q_first} sweeps (DEF-023).
    iq_order == "q_first" conjugates the symbols, which is the spectral-mirror
    ambiguity of a recording whose I and Q channels were swapped; bit_order ==
    "lsb" reverses the bit order within each symbol.
  - Interleaver hypotheses {none, 4x8, 8x4}
  - FEC branches {none, conv_k7_r12, rs255_223, ldpc648_324}
  - Windowed CRC-32 search (DEF-022)
  - Early termination via threading.Event (DEF-021 / config.early_exit_on_crc)
  - Deterministic tie-breaking across parallel workers.

Spec reference: §1.2, DEF-020, DEF-021, DEF-022, DEF-023, DEF-024 (PS26147).
"""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from signal9.config import Config
from signal9.status import Status
from signal9.dsp.constellation import symbols_to_bits, llr_to_soft_symbol
from signal9.dsp.differential import differential_demodulate
from signal9.fec import (
    viterbi_decode,
    rs_decode,
    ldpc_decode,
    block_deinterleave,
)
from signal9.decide.crc_search import windowed_crc32_search, find_trailing_crc32
from signal9.decide.plausibility import plausibility_score


_BITS_PER_SYMBOL = {
    "BPSK": 1, "QPSK": 2, "8PSK": 3,
    "16QAM": 4, "QAM16": 4, "64QAM": 6, "QAM64": 6,
}


def _bits_per_symbol(mod_type: str) -> int:
    return _BITS_PER_SYMBOL.get(mod_type.upper().replace("-", ""), 2)


def _reverse_symbol_bits(stream: np.ndarray, bps: int) -> np.ndarray:
    """Reverse the bit order inside every symbol of a flat per-symbol stream.

    Works for hard bits and for LLRs alike because the operation is a pure index
    permutation.  Any trailing partial symbol is passed through untouched so the
    stream length is preserved.
    """
    stream = np.asarray(stream)
    n_full = (len(stream) // bps) * bps
    if n_full == 0:
        return stream
    head = stream[:n_full].reshape(-1, bps)[:, ::-1].reshape(-1)
    if n_full == len(stream):
        return np.ascontiguousarray(head)
    return np.concatenate([head, stream[n_full:]])


@dataclass(frozen=True)
class BranchResult:
    """Outcome of exploring a single demodulation/decoding hypothesis."""
    branch_id: int
    rotation_deg: int
    is_differential: bool
    bit_order: str
    iq_order: str
    interleaver: str
    fec_type: str
    crc_locked: bool
    payload: bytes
    plausibility: float
    status: Status
    latency_ms: float


class BlackboardArbiter:
    """Multi-hypothesis blackboard search engine."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._early_exit_event = threading.Event()

    def arbitrate(self,
                  symbols: np.ndarray,
                  mod_type: str = "QPSK",
                  fec_hint: str | None = None) -> tuple[BranchResult | None, list[BranchResult]]:
        """Run multi-hypothesis exploration over the symbol stream.

        Parameters
        ----------
        symbols : 1D complex array of symbols (1 SPS)
        mod_type : str
        fec_hint : str, optional

        Returns
        -------
        (winning_branch, all_branch_results)
        """
        symbols = np.asarray(symbols, dtype=np.complex64).ravel()
        self._early_exit_event.clear()

        # Build candidate branch definitions
        branches = self._build_branch_specs(mod_type, fec_hint)

        # Cap branches per config
        if len(branches) > self.config.max_branches:
            branches = branches[: self.config.max_branches]

        results: list[BranchResult] = []

        # Execute branches
        # In test mode or when max_workers=1, run sequentially for deterministic debugging
        for b_idx, spec in enumerate(branches):
            if self.config.early_exit_on_crc and self._early_exit_event.is_set():
                break

            res = self._execute_branch(b_idx, symbols, mod_type, spec)
            results.append(res)

            if res.crc_locked and self.config.early_exit_on_crc:
                self._early_exit_event.set()
                break

        # Deterministic tie-breaking:
        # 1. CRC locked branches outrank non-CRC branches
        # 2. Higher plausibility score wins
        # 3. Deterministic order by branch_id
        if not results:
            return None, []

        def sort_key(r: BranchResult):
            return (
                1 if r.crc_locked else 0,
                r.plausibility,
                -r.branch_id  # lower branch_id preferred
            )

        sorted_results = sorted(results, key=sort_key, reverse=True)
        winner = sorted_results[0]
        return winner, results

    def _build_branch_specs(self, mod_type: str, fec_hint: str | None) -> list[dict]:
        """Generate orthogonal hypothesis branches."""
        rotations = [0, 90, 180, 270]
        bit_orders = ["msb"]
        iq_orders = ["i_first"]

        if self.config.enable_bit_order_sweep:
            bit_orders.append("lsb")
            iq_orders.append("q_first")

        fec_candidates = ["none"]
        if fec_hint and fec_hint != "none":
            fec_candidates.insert(0, fec_hint)
        else:
            fec_candidates.extend(["conv_k7_r12", "rs255_223"])
            if self.config.enable_nms_ldpc:
                fec_candidates.append("ldpc648_324")

        interleavers = ["none"]
        if self.config.enable_interleaver_detect:
            interleavers.extend(["block_4x8", "block_8x4"])

        specs: list[dict] = []

        # Standard coherent branches
        for rot in rotations:
            for bo in bit_orders:
                for iqo in iq_orders:
                    for fec in fec_candidates:
                        for intlv in interleavers:
                            specs.append({
                                "rotation": rot,
                                "is_differential": False,
                                "bit_order": bo,
                                "iq_order": iqo,
                                "interleaver": intlv,
                                "fec": fec,
                            })

        # Track H differential branches (IV-L / DEF-024)
        if self.config.enable_differential_track:
            for fec in fec_candidates:
                specs.append({
                    "rotation": 0,
                    "is_differential": True,
                    "bit_order": "msb",
                    "iq_order": "i_first",
                    "interleaver": "none",
                    "fec": fec,
                })

        return specs

    def _execute_branch(self,
                        branch_id: int,
                        symbols: np.ndarray,
                        mod_type: str,
                        spec: dict) -> BranchResult:
        """Execute a single hypothesis branch."""
        t0 = time.perf_counter()

        # 1. Rotation or Differential demapping
        if spec["is_differential"]:
            sym_proc = differential_demodulate(symbols, mod_type)
        else:
            rot_deg = spec["rotation"]
            rot_factor = np.exp(1j * np.deg2rad(rot_deg))
            sym_proc = symbols * rot_factor

        # Spectral-mirror hypothesis (DEF-023).  A recording whose I and Q leads
        # were swapped, or an LSB-tuned SSB capture, delivers the conjugate
        # constellation.  Conjugating before demapping is the whole content of
        # this hypothesis axis; without it the "q_first" branches were identical
        # to the "i_first" ones and doubled the search cost for nothing.
        if spec["iq_order"] == "q_first":
            sym_proc = np.conj(sym_proc)

        # 2. Demap to soft and hard bits
        soft_llrs = symbols_to_bits(sym_proc, mod_type, hard=False)
        hard_bits = symbols_to_bits(sym_proc, mod_type, hard=True)

        # Bit order within each symbol (DEF-023).  The demapper emits MSB-first;
        # "lsb" reverses the bits of every symbol.  This is a permutation, so it
        # must be applied identically to the hard bits and to the LLRs -- applying
        # it to only one of the two made the FEC and the CRC halves of a branch
        # disagree about which stream they were scoring.
        bps = _bits_per_symbol(mod_type)
        if spec["bit_order"] == "lsb" and bps > 1:
            hard_bits = _reverse_symbol_bits(hard_bits, bps)
            soft_llrs = _reverse_symbol_bits(soft_llrs, bps)

        # 3. Deinterleaving
        intlv = spec["interleaver"]
        if intlv == "block_4x8":
            hard_bits = block_deinterleave(hard_bits, 4, 8)
            soft_llrs = block_deinterleave(soft_llrs, 4, 8)
        elif intlv == "block_8x4":
            hard_bits = block_deinterleave(hard_bits, 8, 4)
            soft_llrs = block_deinterleave(soft_llrs, 8, 4)

        # 4. FEC decoding.
        # symbols_to_bits returns textbook LLRs (positive => bit 0) while the
        # Viterbi and LDPC kernels score against 2*bit-1 (positive => bit 1).
        # Without this conversion every coded branch decodes the bit-complement,
        # raises nothing, and the CRC can never lock.
        fec = spec["fec"]
        decoder_soft = llr_to_soft_symbol(soft_llrs)
        decoded_bytes = b""
        if fec == "conv_k7_r12":
            try:
                dec_bits = viterbi_decode(decoder_soft, K=7, gens=[0o171, 0o133], tb_depth=self.config.viterbi_traceback)
                decoded_bytes = np.packbits(dec_bits).tobytes()
            except Exception:
                decoded_bytes = b""
        elif fec == "rs255_223":
            try:
                raw_bytes = np.packbits(hard_bits)
                if len(raw_bytes) >= 255:
                    rec, n_err = rs_decode(raw_bytes[:255], 255, 223)
                    if n_err >= 0:
                        decoded_bytes = rec.tobytes()
            except Exception:
                decoded_bytes = b""
        elif fec == "ldpc648_324":
            try:
                if len(decoder_soft) >= 648:
                    dec_bits, conv, _, _ = ldpc_decode(decoder_soft[:648], 648, 324, max_iter=30)
                    if conv:
                        decoded_bytes = np.packbits(dec_bits).tobytes()
            except Exception:
                decoded_bytes = b""
        else:
            # Uncoded
            decoded_bytes = np.packbits(hard_bits).tobytes()

        # 5. CRC & Protocol verification (IEEE 802.3, AX.25, CCSDS, ISM)
        if self.config.enable_windowed_crc:
            crc_res = windowed_crc32_search(decoded_bytes)
        else:
            crc_res = find_trailing_crc32(decoded_bytes)

        payload_out = crc_res.payload if crc_res.found else decoded_bytes
        crc_locked = bool(crc_res.found)

        # Multi-protocol deframing fallback if standard trailing CRC-32 was absent
        if not crc_locked and len(decoded_bytes) >= 16:
            try:
                from signal9.protocol.ax25 import deframe_ax25
                ax_frames = deframe_ax25(decoded_bytes)
                valid_ax = [f for f in ax_frames if f.crc_valid]
                if valid_ax:
                    crc_locked = True
                    payload_out = valid_ax[0].payload
            except Exception:
                pass

        if not crc_locked and len(decoded_bytes) >= 16:
            try:
                from signal9.protocol.ccsds import deframe_ccsds_tm
                ccsds_frames = deframe_ccsds_tm(decoded_bytes)
                valid_cc = [f for f in ccsds_frames if f.crc_valid]
                if valid_cc:
                    crc_locked = True
                    payload_out = valid_cc[0].payload
            except Exception:
                pass

        if not crc_locked and len(decoded_bytes) >= 8:
            try:
                from signal9.protocol.ism import deframe_ism
                ism_frames = deframe_ism(decoded_bytes)
                valid_ism = [f for f in ism_frames if f.crc_valid]
                if valid_ism:
                    crc_locked = True
                    payload_out = valid_ism[0].payload
            except Exception:
                pass

        # 6. Plausibility scoring (DEF-006)
        plaus = plausibility_score(
            payload_out,
            crc_valid=crc_locked,
            weights=self.config.entropy_plausibility_weights
        )

        # Determine status
        if crc_locked:
            status = Status.SUCCESS_CRC_LOCKED
        elif len(decoded_bytes) > 0:
            status = Status.DEMOD_NO_CRC
        else:
            status = Status.ALL_BRANCHES_FAILED

        t1 = time.perf_counter()
        return BranchResult(
            branch_id=branch_id,
            rotation_deg=spec["rotation"],
            is_differential=spec["is_differential"],
            bit_order=spec["bit_order"],
            iq_order=spec["iq_order"],
            interleaver=spec["interleaver"],
            fec_type=spec["fec"],
            crc_locked=crc_locked,
            payload=payload_out,
            plausibility=plaus,
            status=status,
            latency_ms=(t1 - t0) * 1000.0
        )
