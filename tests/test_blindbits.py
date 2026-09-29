import numpy as np
from scipy import signal as sps
import blindbits as bb
rng = np.random.default_rng(7)

def rrc(beta, sps_, span=10):
    t = np.arange(-span*sps_, span*sps_+1)/sps_
    with np.errstate(divide='ignore', invalid='ignore'):
        h = (np.sin(np.pi*t*(1-beta)) + 4*beta*t*np.cos(np.pi*t*(1+beta))) / (np.pi*t*(1-(4*beta*t)**2))
    h[np.isnan(h)] = 0
    h[t==0] = 1-beta+4*beta/np.pi
    for s in (1/(4*beta), -1/(4*beta)):
        i = np.argmin(abs(t-s))
        if abs(t[i]-s) < 1e-9:
            h[i] = beta/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*beta))+(1-2/np.pi)*np.cos(np.pi/(4*beta)))
    return h/np.sqrt(np.sum(h**2))

def make_qpsk(nsym=4000, sps_=8, beta=0.35, snr_db=20, cfo=0.0, phase=0.3):
    pts, labels = bb.CONSTELLATIONS["QPSK"]
    idx = rng.integers(0, 4, nsym)
    s = pts[idx]
    up = np.zeros(nsym*sps_, complex); up[::sps_] = s
    x = np.convolve(up, rrc(beta, sps_))
    x /= np.sqrt(np.mean(abs(x)**2))
    n = np.arange(len(x))
    x = x*np.exp(1j*(2*np.pi*cfo*n + phase))
    nz = (rng.standard_normal(len(x))+1j*rng.standard_normal(len(x)))/np.sqrt(2)*10**(-snr_db/20)
    return (x+nz).astype(np.complex64), idx, labels, pts

# 1) symbol rate
x,_,_,_ = make_qpsk(sps_=8, snr_db=20)
r,q = bb.estimate_symbol_rate(x, fs=8.0)   # fs=8 -> symbol rate should be 1.0
print(f"symbol rate {r:.4f} (expect 1.0), quality {q:.1f}"); assert abs(r-1.0) < 0.02 and q > 20

# 2) full linear-demod chain: MF -> timing -> CFO -> Costas -> LLR -> BER
x, idx, labels, pts = make_qpsk(nsym=4000, sps_=8, snr_db=15, cfo=0.001, phase=0.7)
mf = np.convolve(x, rrc(0.35,8)); mf = mf[:len(x)]
dec = sps.resample_poly(mf, 2, 8)                       # to 2 sps
sy = bb.gardner_timing_recovery(dec, sps_in=2.0, loop_bw=0.002)
f_hat,qq = bb.cfo_mth_power(sy, fs=1.0, M=4)
sy = sy*np.exp(-1j*2*np.pi*f_hat*np.arange(len(sy)))
sy = bb.costas_loop(sy, order=4, loop_bw=0.01)
sy = sy[len(sy)//4:]                                     # drop loop transients
print("EVM after chain:", round(bb.evm_db(sy, pts),1), "dB ; cfo_hat", round(f_hat,5))
assert bb.evm_db(sy, pts) < -12

# 3) LLR sanity on clean AWGN QPSK (no filtering) -> BER close to theory
pts, labels = bb.CONSTELLATIONS["QPSK"]
idx = rng.integers(0,4,20000); s = pts[idx]
for snr in (6, 10):
    nz = (rng.standard_normal(len(s))+1j*rng.standard_normal(len(s)))/np.sqrt(2)*10**(-snr/20)
    llr = bb.symbols_to_llr(s+nz, pts, labels)
    bits_true = np.array([int(c) for i in idx for c in labels[i]], dtype=np.uint8)
    ber = np.mean(bb.llr_to_bits(llr) != bits_true)
    # QPSK: Es/N0 = snr, Eb/N0 = snr-3dB, BER = Q(sqrt(2Eb/N0))
    from math import erfc, sqrt
    theory = 0.5*erfc(sqrt(10**((snr-3)/10)))
    print(f"QPSK BER@{snr}dB = {ber:.4f} (theory {theory:.4f})")
    assert abs(ber-theory) < max(0.15*theory, 2e-3)
# 16QAM constellation + LLR
p16,l16 = bb.qam_constellation(16); assert abs(np.mean(abs(p16)**2)-1) < 1e-9 and len(set(l16))==16

# 4) GF(2) rank + code length detection (Hamming(7,4))
G = np.array([[1,0,0,0,1,1,0],[0,1,0,0,1,0,1],[0,0,1,0,0,1,1],[0,0,0,1,1,1,1]], np.uint8)
msg = rng.integers(0,2,(2000,4)).astype(np.uint8)
cw = (msg@G % 2).astype(np.uint8).reshape(-1)
prof = bb.rank_profile(cw, 5, 16, n_rows=120)
print("rank ratio n=7:", round(prof[7],3), " n=9:", round(prof[9],3))
assert prof[7] < 0.62 and prof[9] > 0.95
det = bb.detect_code_length(cw, 4, 20, n_rows=120)
print("detect_code_length:", det); assert det[0] in (7,14) and det[3] < 0.7
# with an offset, detection must still find n=7
det2 = bb.detect_code_length(np.r_[rng.integers(0,2,3).astype(np.uint8), cw], 4, 20, n_rows=120)
assert det2[0] in (7,14)

# 5) block interleaver detection
D = 5
inter = bb.block_interleave(cw, D, 7)
assert np.array_equal(bb.block_deinterleave(inter, D, 7)[:len(cw)//(D*7)*(D*7)], cw[:len(cw)//(D*7)*(D*7)])
best, scores = bb.detect_block_interleaver(inter, code_n=7, depth_max=12, n_rows=150)
print("interleaver depth:", best, {k: round(v,2) for k,v in sorted(scores.items())})
assert best == D

# 6) Berlekamp-Massey + linear complexity
poly = np.zeros(20, np.uint8); poly[0]=1; poly[14]=1; poly[15]=1     # x^15+x^14+1
st = rng.integers(0,2,15).astype(np.uint8); st[0]=1
seq = bb.lfsr_sequence(poly, st, 400)
L, C = bb.berlekamp_massey(seq)
print("BM L =", L, "poly taps", np.nonzero(C)[0])
assert L == 15 and list(np.nonzero(C)[0]) == [0,14,15]
long = bb.lfsr_sequence(poly, st, 1024)
lc = bb.linear_complexity_profile(long, window=256, step=128)
assert lc.max() <= 20
rand = rng.integers(0,2,1024).astype(np.uint8)
lcr = bb.linear_complexity_profile(rand, window=256, step=128)
print("LC: lfsr", lc[:3], " random", lcr[:3]); assert lcr.min() > 100
# additive descrambling round-trip
data = rng.integers(0,2,400).astype(np.uint8)
scr = np.bitwise_xor(data, bb.lfsr_sequence(poly, st, 400))
assert np.array_equal(bb.try_descramble_additive(scr, poly, st), data)

# 7) frame period + sync word + entropy profile
P, sync = 200, np.array([int(b) for b in "10101011110011010101"], np.uint8)
frames = []
for _ in range(120):
    frames.append(np.r_[sync, rng.integers(0,2,P-len(sync)).astype(np.uint8)])
stream = np.concatenate(frames)
peaks = bb.bit_autocorrelation_periods(stream, 16, 600, top=3)
print("periods:", [(p, round(v,3)) for p,v in peaks]); assert any(p % P == 0 for p,_ in peaks[:3])
sw = bb.find_sync_pattern(stream, P)
print("sync:", sw["offset"], sw["length"], "".join(map(str, sw["word"])))
assert sw["length"] >= len(sync) and np.array_equal(sw["word"][:len(sync)], sync)
ent = bb.field_entropy_profile(stream, P)
assert ent[:len(sync)].mean() < 0.1 and ent[len(sync):].mean() > 0.9

# 8) CRC identification
import binascii
msgs = [bytes(rng.integers(0,256,20).tolist()) for _ in range(4)]
crc_frames = [(m, binascii.crc32(m) & 0xFFFFFFFF) for m in msgs]
hits = bb.identify_crc(crc_frames)
print("crc hits:", hits); assert "CRC-32/ISO-HDLC" in hits
assert bb._crc(b"123456789", 16, 0x1021, 0xFFFF, False, False, 0x0000) == 0x29B1   # published check

# 9) convolutional encode/decode (K=7, 171/133) with errors
info = rng.integers(0,2,600).astype(np.uint8)
code = bb.conv_encode(info, (0o171,0o133), 7)
llr = np.where(code==1, -4.0, 4.0).astype(np.float32)
flip = rng.choice(len(llr), size=int(0.03*len(llr)), replace=False)
llr[flip] *= -1
dec = bb.viterbi_decode(llr, (0o171,0o133), 7)
ber = np.mean(dec[:len(info)-6] != info[:len(info)-6])
print("Viterbi BER after 3% channel errors:", round(float(ber),5)); assert ber < 1e-2
sc_true = bb.conv_code_score(llr, (0o171,0o133), 7, n_check=300)
sc_false = bb.conv_code_score(llr, (0o133,0o171), 5, n_check=300)
print("score true", round(sc_true,3), " wrong-hypothesis", round(sc_false,3))
assert sc_true > 0.95 and sc_false < sc_true - 0.05
print("ALL OK")
