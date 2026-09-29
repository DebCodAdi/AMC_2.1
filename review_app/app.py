"""Signal 9 Active-Learning & Human Review App (Streamlit)."""
import streamlit as st
import numpy as np
from pathlib import Path
import json

def run_app():
    st.set_page_config(page_title="Signal 9 Review App", layout="wide")
    st.title("Signal 9 — RF Signal Review & Active Learning")

    st.sidebar.header("Configuration")
    data_dir = st.sidebar.text_input("Detections Report JSON", "report.json")

    report_path = Path(data_dir)
    if not report_path.exists():
        st.info("No report.json found. Run 'python main.py <file> --save-json report.json' first.")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    st.subheader(f"Analyzing: {report.get('file', 'Unknown')}")
    st.write(f"**Sample Rate**: {report.get('sample_rate_sps', 0)/1e6:.3f} MS/s | **Center Freq**: {report.get('center_freq_hz', 0)/1e6:.3f} MHz")

    signals = report.get("signals", [])
    st.write(f"Total Detections: {len(signals)}")

    for sig in signals:
        with st.expander(f"Track #{sig.get('track_id')} — BW: {sig.get('bandwidth_hz', 0)/1e3:.1f} kHz"):
            c1, c2, c3 = st.columns(3)
            cls = sig.get("classification", {})
            c1.metric("L1 Family", cls.get("l1", {}).get("prediction"), f"{cls.get('l1', {}).get('confidence', 0)*100:.1f}%")
            c2.metric("L2 Modulation", cls.get("l2", {}).get("prediction"), f"{cls.get('l2', {}).get('confidence', 0)*100:.1f}%")
            c3.metric("L3 Protocol", cls.get("l3", {}).get("prediction"), f"{cls.get('l3', {}).get('confidence', 0)*100:.1f}%")

            st.write(f"**SNR**: {sig.get('snr_db', 0):.1f} dB | **Duration**: {sig.get('duration_sec', 0):.2f} s")
            
            review_col1, review_col2 = st.columns(2)
            if review_col1.button(f"Approve as GOLD #{sig.get('track_id')}"):
                st.success("Track marked as GOLD ground truth.")
            if review_col2.button(f"Flag as UNKNOWN #{sig.get('track_id')}"):
                st.warning("Track added to Outlier Exposure queue.")

if __name__ == "__main__":
    run_app()
