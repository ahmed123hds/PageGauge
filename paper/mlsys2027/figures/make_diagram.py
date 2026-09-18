import os

svg_path = "C:/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090/paper/mlsys2027/figures/architecture_diagram.svg"

p = []
def w(s): p.append(s)

w('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 820" width="1440" height="820" style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">')
w('<defs>')
w('<linearGradient id="pGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#ffffff"/><stop offset="100%" stop-color="#f8fafc"/></linearGradient>')
w('<linearGradient id="sinkGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#f0fdf4"/><stop offset="100%" stop-color="#dcfce7"/></linearGradient>')
w('<linearGradient id="quantGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#eff6ff"/><stop offset="100%" stop-color="#dbeafe"/></linearGradient>')
w('<linearGradient id="tailGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#f0fdfa"/><stop offset="100%" stop-color="#ccfbf1"/></linearGradient>')
w('<linearGradient id="convGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#fff1f2"/><stop offset="100%" stop-color="#ffe4e6"/></linearGradient>')
w('<linearGradient id="pgGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#faf5ff"/><stop offset="100%" stop-color="#f3e8ff"/></linearGradient>')
w('<linearGradient id="goldGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#fefce8"/><stop offset="100%" stop-color="#fef08a"/></linearGradient>')
w('<linearGradient id="tcGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#3b82f6"/><stop offset="100%" stop-color="#1d4ed8"/></linearGradient>')
w('<linearGradient id="barFP16" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#94a3b8"/><stop offset="100%" stop-color="#64748b"/></linearGradient>')
w('<linearGradient id="barPG" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#3b82f6"/><stop offset="100%" stop-color="#10b981"/></linearGradient>')

w('<marker id="arrB" markerWidth="9" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 9 3, 0 6" fill="#2563eb"/></marker>')
w('<marker id="arrR" markerWidth="9" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 9 3, 0 6" fill="#e11d48"/></marker>')
w('<marker id="arrP" markerWidth="9" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 9 3, 0 6" fill="#7c3aed"/></marker>')
w('<marker id="arrT" markerWidth="9" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 9 3, 0 6" fill="#0d9488"/></marker>')
w('<marker id="arrG" markerWidth="9" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 9 3, 0 6" fill="#d97706"/></marker>')
w('</defs>')

# Panel Frames
w('<rect x="15" y="15" width="445" height="790" rx="12" fill="url(#pGrad)" stroke="#cbd5e1" stroke-width="1.5"/>')
w('<rect x="15" y="15" width="445" height="42" rx="12" fill="#2563eb"/><rect x="15" y="45" width="445" height="12" fill="#2563eb"/>')
w('<text x="30" y="41" fill="#ffffff" font-size="15" font-weight="700">(a) Paged Mixed-Precision KV Cache Memory</text>')

w('<rect x="475" y="15" width="520" height="790" rx="12" fill="url(#pGrad)" stroke="#cbd5e1" stroke-width="1.5"/>')
w('<rect x="475" y="15" width="520" height="42" rx="12" fill="#7c3aed"/><rect x="475" y="45" width="520" height="12" fill="#7c3aed"/>')
w('<text x="490" y="41" fill="#ffffff" font-size="15" font-weight="700">(b) Factorized Softmax Contraction Pipeline</text>')

w('<rect x="1010" y="15" width="415" height="790" rx="12" fill="url(#pGrad)" stroke="#cbd5e1" stroke-width="1.5"/>')
w('<rect x="1010" y="15" width="415" height="42" rx="12" fill="#0d9488"/><rect x="1010" y="45" width="415" height="12" fill="#0d9488"/>')
w('<text x="1025" y="41" fill="#ffffff" font-size="15" font-weight="700">(c) Associative Reduction &amp; Global Epilogue</text>')

# PANEL A: MEMORY
w('<g transform="translate(30, 70)"><rect width="415" height="26" rx="5" fill="#f8fafc" stroke="#94a3b8" stroke-dasharray="3 3"/><text x="207" y="17" text-anchor="middle" font-size="11" font-weight="600" fill="#475569">Physical GPU DRAM (GDDR7 / HBM) - 16-Token Page Blocks</text></g>')

# S4 Sinks
w('<g transform="translate(30, 104)"><rect width="415" height="74" rx="8" fill="url(#sinkGrad)" stroke="#10b981" stroke-width="1.2"/>')
w('<text x="14" y="22" font-size="12.5" font-weight="700" fill="#065f46">Exact Attention Sinks (S4: Tokens 0-63)</text>')
w('<rect x="310" y="8" width="92" height="20" rx="4" fill="#10b981"/><text x="356" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#ffffff">FP16 (2B/elem)</text>')
w('<g transform="translate(14, 34)">')
for i, t in enumerate(["0..15", "16..31", "32..47", "48..63"]):
    w(f'<rect x="{i*99}" y="0" width="90" height="28" rx="4" fill="#ffffff" stroke="#10b981" stroke-width="0.8"/>')
    w(f'<text x="{i*99+45}" y="13" text-anchor="middle" font-size="9.5" font-weight="600" fill="#047857">Page {i} [{t}]</text>')
    w(f'<text x="{i*99+45}" y="23" text-anchor="middle" font-size="8" fill="#64748b">16 x 128 FP16</text>')
w('</g></g>')

# Q Quantized Pages 2D Grid
w('<g transform="translate(30, 188)"><rect width="415" height="315" rx="8" fill="url(#quantGrad)" stroke="#2563eb" stroke-width="1.2"/>')
w('<text x="14" y="22" font-size="12.5" font-weight="700" fill="#1e40af">Historical Quantized Pages (Q: N Pages)</text>')
w('<rect x="300" y="8" width="102" height="20" rx="4" fill="#2563eb"/><text x="351" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#ffffff">INT8 (1B/elem)</text>')
w('<g transform="translate(14, 36)"><rect width="387" height="175" rx="6" fill="#ffffff" stroke="#93c5fd" stroke-width="1"/>')
w('<text x="12" y="16" font-size="10.5" font-weight="700" fill="#1e3a8a">Physical Page p Layout (Tokens t .. t+15, Channels d=128)</text>')
w('<g transform="translate(12, 32)">')
w('<text x="45" y="-5" text-anchor="middle" font-size="8.5" font-weight="600" fill="#64748b">Tokens</text>')
w('<text x="175" y="-5" text-anchor="middle" font-size="8.5" font-weight="600" fill="#2563eb">Signed INT8 Codes Z_p in [-127, 127] (d=0..127)</text>')
w('<text x="330" y="-5" text-anchor="middle" font-size="8.5" font-weight="600" fill="#d97706">Scales s_p</text>')

rows = [
    ("Tok 0..3 :", "+14 -82 +103 -04 ... -19 +67", "s_p^K = 0.042", "#dbeafe"),
    ("Tok 4..7 :", "-41 +112 -28 +09 ... -87 -12", "s_p^V = 0.038", "#eff6ff"),
    ("Tok 8..11:", "+38 -06 +74 -95 ... +44 -62", None, "#dbeafe"),
    ("Tok 12..15:", "-16 -63 +04 +88 ... -09 +31", None, "#eff6ff")
]
for idx, (lbl, val, sc, bg) in enumerate(rows):
    y = idx * 22
    w(f'<rect x="0" y="{y}" width="285" height="19" rx="2" fill="{bg}" stroke="#bfdbfe"/>')
    w(f'<text x="6" y="{y+13}" font-size="9" font-family="monospace" font-weight="600" fill="#1e40af">{lbl}</text>')
    w(f'<text x="75" y="{y+13}" font-size="9" font-family="monospace" fill="#1e3a8a">{val}</text>')
    if sc:
        w(f'<rect x="295" y="{y}" width="68" height="19" rx="2" fill="#fef3c7" stroke="#fde047"/>')
        w(f'<text x="329" y="{y+13}" text-anchor="middle" font-size="8.5" font-weight="700" fill="#b45309">{sc}</text>')
w('<rect x="295" y="44" width="68" height="41" rx="2" fill="#f8fafc" stroke="#cbd5e1"/>')
w('<text x="329" y="60" text-anchor="middle" font-size="8" fill="#64748b">Only 2 FP16</text><text x="329" y="73" text-anchor="middle" font-size="8" fill="#64748b">scalars / page</text>')
w('<text x="142" y="112" text-anchor="middle" font-size="9.5" font-weight="700" fill="#1d4ed8">Signed INT8 Cache Codes: [-127, +127] (1 Byte/element)</text>')
w('<text x="142" y="125" text-anchor="middle" font-size="8.5" fill="#64748b">Scale metadata: 4 Bytes / 2048 Bytes = 0.19% (0.125 B/token overhead)</text>')
w('</g></g>')

# Centers
w('<g transform="translate(14, 220)"><rect width="387" height="46" rx="5" fill="url(#goldGrad)" stroke="#f59e0b" stroke-width="1"/>')
w('<text x="12" y="18" font-size="11" font-weight="700" fill="#92400e">Frozen Channel Centers: c_K, c_V in R^d (FP16 Global Vectors)</text>')
w('<text x="12" y="34" font-size="9" fill="#78350f">Computed once during prefill; shared across all N pages in layer</text></g>')

# Page p+1 ellipsis
w('<g transform="translate(14, 274)"><rect width="387" height="28" rx="4" fill="#ffffff" stroke="#93c5fd" stroke-dasharray="4 3"/>')
w('<text x="193" y="18" text-anchor="middle" font-size="10" font-weight="600" fill="#3b82f6">Physical Page p+1  ...  Page p+N-1  (Identical Block-Strided Layout)</text></g>')
w('</g>')

# Tail Ring Buffer T768
w('<g transform="translate(30, 513)"><rect width="415" height="74" rx="8" fill="url(#tailGrad)" stroke="#0d9488" stroke-width="1.2"/>')
w('<text x="14" y="22" font-size="12.5" font-weight="700" fill="#115e59">Recent Tail Ring Buffer (T768: 768 Tokens)</text>')
w('<rect x="295" y="8" width="105" height="20" rx="4" fill="#0d9488"/><text x="347" y="22" text-anchor="middle" font-size="10" font-weight="700" fill="#ffffff">Centered FP16</text>')
w('<g transform="translate(14, 34)">')
for idx, (t1, t2) in enumerate([("Page N-48..N-33", "K_E, V_E (FP16)"), ("Page N-32..N-17", "Centered via c"), ("Recent Tail (N)", "Guards Dynamic")]):
    w(f'<rect x="{idx*132}" y="0" width="122" height="28" rx="4" fill="#ffffff" stroke="#0d9488" stroke-width="0.8"/>')
    w(f'<text x="{idx*132+61}" y="13" text-anchor="middle" font-size="9" font-weight="600" fill="#0f766e">{t1}</text>')
    w(f'<text x="{idx*132+61}" y="24" text-anchor="middle" font-size="8" fill="#64748b">{t2}</text>')
w('</g></g>')

# Memory footprint comparison
w('<g transform="translate(30, 597)"><rect width="415" height="195" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.2"/>')
w('<text x="14" y="22" font-size="12" font-weight="700" fill="#1e293b">Physical Served KV Cache Memory on RTX 5090</text>')
w('<text x="14" y="37" font-size="9.5" fill="#64748b">Mistral-7B-v0.3 at Context Length 20,480 Tokens (Batch Size B=4)</text>')
w('<g transform="translate(14, 52)"><text x="0" y="12" font-size="10" font-weight="600" fill="#475569">FlashInfer FP16 Baseline</text><text x="387" y="12" text-anchor="end" font-size="10" font-weight="700" fill="#475569">2,752 MiB (100%)</text>')
w('<rect x="0" y="18" width="387" height="18" rx="4" fill="#e2e8f0"/><rect x="0" y="18" width="387" height="18" rx="4" fill="url(#barFP16)"/></g>')
w('<g transform="translate(14, 102)"><text x="0" y="12" font-size="10" font-weight="700" fill="#2563eb">PageGauge INT8 (S4/A0/T768)</text><text x="387" y="12" text-anchor="end" font-size="10" font-weight="700" fill="#059669">1,481 MiB (53.84%)</text>')
w('<rect x="0" y="18" width="387" height="18" rx="4" fill="#e2e8f0"/><rect x="0" y="18" width="208" height="18" rx="4" fill="url(#barPG)"/>')
w('<rect x="208" y="18" width="179" height="18" rx="4" fill="#fee2e2" stroke="#fca5a5" stroke-dasharray="3 2"/><text x="297" y="31" text-anchor="middle" font-size="9" font-weight="700" fill="#dc2626">-46.16% Memory Cut</text></g>')
w('<rect x="14" y="152" width="387" height="28" rx="4" fill="#f0fdf4" stroke="#86efac"/>')
w('<text x="207" y="170" text-anchor="middle" font-size="10.5" font-weight="700" fill="#166534">1.86x Deployment Capacity Multiplier | Enables B=8 Serving without OOM</text></g>')

# PANEL B: HARDWARE PIPELINE
w('<g transform="translate(490, 70)"><rect width="490" height="175" rx="8" fill="url(#convGrad)" stroke="#e11d48" stroke-width="1.2"/>')
w('<text x="16" y="22" font-size="12" font-weight="700" fill="#9f1239">Conventional Fused Low-Bit Attention (Prior Art)</text>')
w('<rect x="375" y="8" width="100" height="20" rx="4" fill="#e11d48"/><text x="425" y="22" text-anchor="middle" font-size="9.5" font-weight="700" fill="#ffffff">ALU Bottleneck</text>')
w('<g transform="translate(16, 38)">')
w('<rect x="0" y="0" width="110" height="54" rx="5" fill="#ffffff" stroke="#f43f5e" stroke-width="1"/>')
w('<text x="55" y="18" text-anchor="middle" font-size="10" font-weight="700" fill="#881337">Load Packed</text><text x="55" y="32" text-anchor="middle" font-size="8.5" fill="#9f1239">Codes Z, Scales s,</text><text x="55" y="45" text-anchor="middle" font-size="8.5" fill="#9f1239">Zero-Points z</text>')
w('<line x1="110" y1="27" x2="140" y2="27" stroke="#e11d48" stroke-width="1.5" marker-end="url(#arrR)"/>')
w('<rect x="140" y="0" width="185" height="54" rx="5" fill="#ffe4e6" stroke="#e11d48" stroke-width="1.5"/>')
w('<text x="232" y="17" text-anchor="middle" font-size="10" font-weight="700" fill="#9f1239">In-Register ALU Dequantization</text>')
w('<text x="232" y="34" text-anchor="middle" font-size="11" font-family="monospace" font-weight="700" fill="#881337">X\' = (Z - z) * s</text>')
w('<text x="232" y="47" text-anchor="middle" font-size="8" fill="#be123c">Serialized CUDA Core ALU Operations</text>')
w('<line x1="325" y1="27" x2="355" y2="27" stroke="#e11d48" stroke-width="1.5" marker-end="url(#arrR)"/>')
w('<rect x="355" y="0" width="103" height="54" rx="5" fill="#ffffff" stroke="#f43f5e" stroke-width="1"/>')
w('<text x="406" y="18" text-anchor="middle" font-size="10" font-weight="700" fill="#881337">Tensor Cores</text><text x="406" y="32" text-anchor="middle" font-size="9" fill="#9f1239">FP16 MMA</text><text x="406" y="45" text-anchor="middle" font-size="8.5" font-family="monospace" fill="#881337">q*K^T, P*V</text>')
w('</g>')
w('<rect x="16" y="110" width="458" height="50" rx="5" fill="#ffffff" stroke="#fca5a5"/>')
w('<text x="245" y="130" text-anchor="middle" font-size="10.5" font-weight="700" fill="#b91c1c">Critical Hardware Bottleneck in Autoregressive Serving:</text>')
w('<text x="245" y="147" text-anchor="middle" font-size="9" fill="#991b1b">Holding scales + zero-points per element pushes register file to &gt;160 regs/thread &#x2192; Warp Stalls</text></g>')

# Bottom: PageGauge Factorized
w('<g transform="translate(490, 260)"><rect width="490" height="532" rx="8" fill="url(#pgGrad)" stroke="#7c3aed" stroke-width="1.5"/>')
w('<text x="16" y="24" font-size="12.5" font-weight="700" fill="#5b21b6">PageGauge Factorized Execution (Zero In-Register Dequant)</text>')
w('<rect x="370" y="10" width="105" height="20" rx="4" fill="#7c3aed"/><text x="422" y="24" text-anchor="middle" font-size="10" font-weight="700" fill="#ffffff">Pure Tensor Cores</text>')

# 1. Direct Load
w('<g transform="translate(16, 42)"><rect width="458" height="46" rx="6" fill="#ffffff" stroke="#8b5cf6" stroke-width="1"/>')
w('<text x="16" y="18" font-size="11" font-weight="700" fill="#4c1d95">1. Direct Load INT8 Codes: Z_p^K, Z_p^V into Register Fragments</text>')
w('<text x="16" y="34" font-size="9.5" fill="#6d28d9">Halves memory traffic across GDDR7 bus; zero per-element offset metadata loaded</text>')
w('<rect x="385" y="11" width="60" height="24" rx="4" fill="#ede9fe"/><text x="415" y="27" text-anchor="middle" font-size="9.5" font-weight="700" fill="#6d28d9">64 regs</text></g>')
w('<line x1="245" y1="350" x2="245" y2="368" stroke="#7c3aed" stroke-width="2" marker-end="url(#arrP)"/>')

# 2. Shift Invariance
w('<g transform="translate(16, 110)"><rect width="458" height="116" rx="6" fill="url(#goldGrad)" stroke="#d97706" stroke-width="1.2"/>')
w('<text x="16" y="20" font-size="11.5" font-weight="700" fill="#92400e">Property 1: Key Center Softmax Shift-Invariance Cancellation</text>')
w('<rect x="16" y="30" width="426" height="52" rx="4" fill="#ffffff" stroke="#fde047"/>')
w('<text x="229" y="49" text-anchor="middle" font-size="8.5" font-family="monospace" font-weight="600" fill="#1e293b">alpha * q^T * K_p^T  =  alpha * s_p^K * q^T * (Z_p^K)^T  +  delta * 1^T</text>')
w('<text x="229" y="69" text-anchor="middle" font-size="9" font-family="monospace" font-weight="700" fill="#7c3aed">softmax(z + delta * 1^T) == softmax(z)   [Identical Distribution]</text>')
w('<text x="229" y="102" text-anchor="middle" font-size="10" font-weight="700" fill="#b45309">&#x2714; Constant row offset delta cancels in softmax &#x2192; ZERO ALU subtractions in registers!</text></g>')
w('<line x1="245" y1="488" x2="245" y2="505" stroke="#7c3aed" stroke-width="2" marker-end="url(#arrP)"/>')

# 3. Tensor Core MMA & Hoisted Scaling
w('<g transform="translate(16, 248)"><rect width="458" height="136" rx="6" fill="#ffffff" stroke="#8b5cf6" stroke-width="1.2"/>')
w('<text x="16" y="20" font-size="11.5" font-weight="700" fill="#4c1d95">2. Direct Tensor Core MMA &amp; Hoisted Fragment Scaling</text>')
w('<g transform="translate(16, 32)"><rect x="0" y="0" width="200" height="56" rx="5" fill="url(#tcGrad)" stroke="#1d4ed8" stroke-width="1"/>')
w('<text x="100" y="18" text-anchor="middle" font-size="10.5" font-weight="700" fill="#ffffff">NVIDIA Tensor Cores</text>')
w('<text x="100" y="33" text-anchor="middle" font-size="10.5" font-family="monospace" font-weight="700" fill="#ffffff">q * (Z_p^K)^T</text>')
w('<text x="100" y="47" text-anchor="middle" font-size="8" fill="#bfdbfe">Raw INT8 Matrix Contraction</text>')
w('<line x1="200" y1="28" x2="226" y2="28" stroke="#2563eb" stroke-width="1.5" marker-end="url(#arrB)"/>')
w('<rect x="226" y="0" width="200" height="56" rx="5" fill="#f0fdf4" stroke="#10b981" stroke-width="1.2"/>')
w('<text x="326" y="18" text-anchor="middle" font-size="10.5" font-weight="700" fill="#065f46">Hoisted Fragment Scale</text>')
w('<text x="326" y="33" text-anchor="middle" font-size="10.5" font-family="monospace" font-weight="700" fill="#047857">a_p = s_p^K * (q * (Z_p^K)^T)</text>')
w('<text x="326" y="47" text-anchor="middle" font-size="8" fill="#059669">Multiplied ONCE to 16-elem fragment</text></g>')
w('<g transform="translate(16, 96)"><rect width="426" height="30" rx="4" fill="#faf5ff" stroke="#a78bfa"/>')
w('<text x="213" y="19" text-anchor="middle" font-size="9.5" font-weight="600" fill="#5b21b6">Value Contraction: u_p = s_p^V * (P_p * Z_p^V) via Tensor Cores (MMA commutes with s_p^V)</text></g></g>')
w('<line x1="245" y1="646" x2="245" y2="663" stroke="#7c3aed" stroke-width="2" marker-end="url(#arrP)"/>')

# 4. Occupancy Badge
w('<g transform="translate(16, 406)"><rect width="458" height="110" rx="6" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.2"/>')
w('<text x="229" y="24" text-anchor="middle" font-size="12" font-weight="700" fill="#4c1d95">Hardware SM Efficiency on Blackwell (SM120)</text>')
w('<g transform="translate(14, 38)">')
badges = [("&lt;= 64 Regs/Thread", "Max SM Occupancy", "Zero Register Spills"), ("Zero ALU Dequant", "Shift Cancels in Exp", "Zero ALU Stalls"), ("Pure Tensor Cores", "Hardware MMA Units", "Max Compute FLOPS")]
for idx, (b1, b2, b3) in enumerate(badges):
    w(f'<rect x="{idx*148}" y="0" width="134" height="54" rx="4" fill="#ffffff" stroke="#8b5cf6"/>')
    w(f'<text x="{idx*148+67}" y="18" text-anchor="middle" font-size="9.5" font-weight="700" fill="#5b21b6">{b1}</text>')
    w(f'<text x="{idx*148+67}" y="34" text-anchor="middle" font-size="8" fill="#6d28d9">{b2}</text>')
    w(f'<text x="{idx*148+67}" y="46" text-anchor="middle" font-size="8" fill="#059669">{b3}</text>')
w('</g></g></g>')

# PANEL C: REDUCTION & EPILOGUE
w('<g transform="translate(1025, 70)">')
w('<rect x="0" y="0" width="188" height="120" rx="6" fill="url(#quantGrad)" stroke="#2563eb" stroke-width="1.2"/>')
w('<text x="94" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#1e40af">Historical INT8 State</text>')
w('<rect x="36" y="28" width="116" height="18" rx="3" fill="#2563eb"/><text x="94" y="41" text-anchor="middle" font-size="9.5" font-weight="700" fill="#ffffff">Pages Q in HBM</text>')
w('<text x="14" y="66" font-size="9.5" font-family="monospace" fill="#1e3a8a">m_int8 = max_p(a_p)</text>')
w('<text x="14" y="84" font-size="9.5" font-family="monospace" fill="#1e3a8a">l_int8 = sum(exp(a_p-m))</text>')
w('<text x="14" y="102" font-size="9.5" font-family="monospace" fill="#1e3a8a">u_int8 = sum(s_p^V*exp*Z)</text>')

w('<rect x="198" y="0" width="188" height="120" rx="6" fill="url(#tailGrad)" stroke="#0d9488" stroke-width="1.2"/>')
w('<text x="292" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#0f766e">Exact Tail &amp; Sinks</text>')
w('<rect x="234" y="28" width="116" height="18" rx="3" fill="#0d9488"/><text x="292" y="41" text-anchor="middle" font-size="9.5" font-weight="700" fill="#ffffff">S4 + T768 in HBM</text>')
w('<text x="212" y="66" font-size="9.5" font-family="monospace" fill="#115e59">m_exact = max_E(a_E)</text>')
w('<text x="212" y="84" font-size="9.5" font-family="monospace" fill="#115e59">l_exact = sum(exp(a_E-m))</text>')
w('<text x="212" y="102" font-size="9.5" font-family="monospace" fill="#115e59">u_exact = sum(exp*V\'_E)</text>')
w('</g>')

w('<line x1="1119" y1="190" x2="1119" y2="218" stroke="#2563eb" stroke-width="2" marker-end="url(#arrB)"/>')
w('<line x1="1317" y1="190" x2="1317" y2="218" stroke="#0d9488" stroke-width="2" marker-end="url(#arrT)"/>')

# SMEM Reducer
w('<g transform="translate(1025, 220)"><rect width="386" height="185" rx="8" fill="#ffffff" stroke="#64748b" stroke-width="1.5"/>')
w('<text x="16" y="24" font-size="12" font-weight="700" fill="#334155">Associative Online Softmax Merger (SMEM)</text>')
w('<rect x="275" y="10" width="95" height="20" rx="4" fill="#64748b"/><text x="322" y="24" text-anchor="middle" font-size="9.5" font-weight="700" fill="#ffffff">Numerically Exact</text>')
w('<g transform="translate(16, 40)"><rect width="354" height="102" rx="5" fill="#f8fafc" stroke="#cbd5e1"/>')
w('<text x="16" y="26" font-size="10" font-family="monospace" font-weight="700" fill="#0f172a">m = max(m_int8, m_exact)</text>')
w('<text x="16" y="54" font-size="9.5" font-family="monospace" font-weight="700" fill="#0f172a">l = exp(m_int8-m)*l_int8 + exp(m_exact-m)*l_exact</text>')
w('<text x="16" y="82" font-size="9.5" font-family="monospace" font-weight="700" fill="#0f172a">u = exp(m_int8-m)*u_int8 + exp(m_exact-m)*u_exact</text></g>')
w('<text x="193" y="166" text-anchor="middle" font-size="9.5" fill="#64748b">FlashAttention-style associative state reduction across warps</text></g>')

w('<line x1="1218" y1="405" x2="1218" y2="435" stroke="#d97706" stroke-width="2" marker-end="url(#arrG)"/>')

# Global Value Center Restoration Epilogue
w('<g transform="translate(1025, 438)"><rect width="386" height="160" rx="8" fill="url(#goldGrad)" stroke="#d97706" stroke-width="1.5"/>')
w('<text x="16" y="24" font-size="12" font-weight="700" fill="#92400e">Property 2: Single Global Value Center Epilogue</text>')
w('<g transform="translate(16, 36)"><rect width="354" height="66" rx="5" fill="#ffffff" stroke="#fde047"/>')
w('<text x="177" y="28" text-anchor="middle" font-size="13.5" font-family="monospace" font-weight="700" fill="#b45309">o = (u / l)  +  c_V^T</text>')
w('<text x="177" y="50" text-anchor="middle" font-size="10.5" font-weight="600" fill="#78350f">&#x21D2; Exactly holds because sum_i P_i == 1</text></g>')
w('<text x="193" y="124" text-anchor="middle" font-size="9.5" font-weight="700" fill="#92400e">Restores channel center c_V ONCE GLOBALLY in kernel epilogue;</text>')
w('<text x="193" y="142" text-anchor="middle" font-size="9" fill="#78350f">completely eliminates N page-level bias additions from inner loop!</text></g>')

w('<line x1="1218" y1="598" x2="1218" y2="628" stroke="#10b981" stroke-width="2" marker-end="url(#arrT)"/>')

# Output Vector
w('<g transform="translate(1025, 630)"><rect width="386" height="64" rx="8" fill="#ecfdf5" stroke="#10b981" stroke-width="1.5"/>')
w('<text x="193" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#065f46">Final Attention Output Vector: o in R^d</text>')
w('<text x="193" y="48" text-anchor="middle" font-size="10" font-weight="600" fill="#047857">Streamed directly to Out-Projection GEMM &amp; Feed-Forward MLP</text></g>')

# Speedup Badge
w('<g transform="translate(1025, 706)"><rect width="386" height="84" rx="8" fill="#f0fdfa" stroke="#0d9488" stroke-width="1.5"/>')
w('<text x="193" y="26" text-anchor="middle" font-size="14.5" font-weight="800" fill="#0f766e">1.67x Attention Kernel Speedup</text>')
w('<text x="193" y="48" text-anchor="middle" font-size="11" font-weight="600" fill="#134e4a">132.20 us (PageGauge) vs. 220.75 us (FlashInfer FP16)</text>')
w('<text x="193" y="68" text-anchor="middle" font-size="9.5" font-weight="600" fill="#059669">Evaluated on NVIDIA RTX 5090 (Blackwell SM120) with 256 MiB scrubs</text></g>')

# Cross-Panel Arrows
w('<path d="M 445 325 L 485 325" fill="none" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arrB)"/>')
w('<path d="M 965 325 L 1020 130" fill="none" stroke="#2563eb" stroke-width="2" marker-end="url(#arrB)"/>')

w('</svg>')

with open(svg_path, "w", encoding="utf-8") as f:
    f.write("\n".join(p))
print("Done writing refined SVG!")
