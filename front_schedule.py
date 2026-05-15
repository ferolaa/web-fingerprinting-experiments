"""
Task 1: FRONT Defence Schedule Generator
Implements the FRONT schedule from Appendix A of Smith et al. (USENIX Security 2022).
FRONT adds nc (client) and ns (server) chaff packets with timestamps drawn from
a Rayleigh distribution: f(t; w) = (t/w^2) * exp(-t^2 / 2w^2) for t >= 0.
Parameters match paper: Nc=Ns=1000, Wmin=0.5s, Wmax=7s, packet_size=1200 bytes.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)

# --- Defence parameters (from paper Section 4.2) ---
Nc, Ns   = 1000, 1000   # max chaff packets per direction
Wmin, Wmax = 0.5, 7.0  # uniform range for Rayleigh scale (seconds)
PKT_SIZE = 1200         # bytes per chaff packet

def rayleigh_timestamps(n, w):
    """Sample n timestamps from Rayleigh(w) distribution."""
    # Rayleigh CDF: F(t) = 1 - exp(-t^2 / 2w^2)
    # Inverse CDF: t = w * sqrt(-2 * ln(1 - u))
    u = np.random.uniform(0, 1, n)
    return w * np.sqrt(-2 * np.log(1 - u + 1e-12))

def generate_front_schedule():
    """Generate one FRONT chaff schedule."""
    # Sample parameters
    wc = np.random.uniform(Wmin, Wmax)
    ws = np.random.uniform(Wmin, Wmax)
    nc = np.random.randint(1, Nc + 1)
    ns_ = np.random.randint(1, Ns + 1)

    t_out = np.sort(rayleigh_timestamps(nc, wc))
    t_in  = np.sort(rayleigh_timestamps(ns_, ws))
    return t_out, t_in, wc, ws, nc, ns_

# --- Generate 5 independent schedules ---
fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=False)
fig.patch.set_facecolor('#0D1117')

colors = {'out': '#58A6FF', 'in': '#3FB950'}

for i, ax in enumerate(axes):
    t_out, t_in, wc, ws, nc, ns_ = generate_front_schedule()
    ax.set_facecolor('#161B22')

    # Plot outgoing (positive) and incoming (negative)
    ax.vlines(t_out, 0,  PKT_SIZE, color=colors['out'], alpha=0.6, linewidth=0.8)
    ax.vlines(t_in,  0, -PKT_SIZE, color=colors['in'],  alpha=0.6, linewidth=0.8)
    ax.axhline(0, color='#30363D', linewidth=0.8)

    ax.set_ylabel('Bytes', color='#8B949E', fontsize=8)
    ax.set_ylim(-1500, 1500)
    ax.tick_params(colors='#8B949E', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363D')

    ax.set_title(
        f'Schedule {i+1}:  nc={nc}, wc={wc:.2f}s  |  ns={ns_}, ws={ws:.2f}s',
        color='#F0F6FC', fontsize=8, loc='left', pad=4
    )
    ax.set_xlim(0, max(t_out.max() if len(t_out) else 1,
                       t_in.max()  if len(t_in)  else 1) * 1.05)
    ax.set_xlabel('Time (s)', color='#8B949E', fontsize=7)

out_patch = mpatches.Patch(color=colors['out'], label='Outgoing chaff (client → server)')
in_patch  = mpatches.Patch(color=colors['in'],  label='Incoming chaff (server → client)')
fig.legend(handles=[out_patch, in_patch], loc='upper right',
           facecolor='#161B22', edgecolor='#30363D',
           labelcolor='#F0F6FC', fontsize=9)

fig.suptitle(
    'FRONT Defence — Chaff Schedule Generator\n'
    r'$f(t;w)=\frac{t}{w^2}e^{-t^2/2w^2}$, '
    r'$w_c, w_s \sim \mathcal{U}(0.5, 7.0)$, '
    r'$n_c, n_s \sim \mathcal{U}\{1, 1000\}$',
    color='#F0F6FC', fontsize=10, y=1.01
)
plt.tight_layout()
plt.savefig('/home/claude/front_schedule.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("FRONT figure saved.")

# --- Also save a summary of parameters used ---
schedules_summary = []
for i in range(5):
    t_out, t_in, wc, ws, nc, ns_ = generate_front_schedule()
    bw_chaff = (nc + ns_) * PKT_SIZE  # bytes of chaff
    schedules_summary.append((nc, wc, ns_, ws, bw_chaff))
    print(f"Schedule {i+1}: nc={nc}, wc={wc:.2f}s, ns={ns_}, ws={ws:.2f}s, "
          f"chaff_bytes={bw_chaff/1024:.1f} KB")
