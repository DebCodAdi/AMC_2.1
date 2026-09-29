#!/usr/bin/env bash
# RTL-SDR scheduled recording script
set -e

FREQ=$1
RATE=$2
DURATION=$3
OUTFILE=$4

if [ -z "$OUTFILE" ]; then
    echo "Usage: $0 <freq_hz> <rate_hz> <duration_sec> <outfile.cu8>"
    exit 1
fi

echo "Recording $FREQ Hz at $RATE S/s for $DURATION seconds..."
# Capture cu8 raw binary
rtl_sdr -f "$FREQ" -s "$RATE" -n $(echo "$RATE * $DURATION" | bc) "$OUTFILE"
echo "Done."
