"""
k-FP reproduction with realistic noise, feature overlap, and open-world confusion.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, classification_report)
import warnings; warnings.filterwarnings('ignore')

np.random.seed(7)

N_MON   = 15     # monitored sites
SAMPLES = 60     # traces per site
N_UNDEF = 400    # unmonitored

def make_site_params(site_id):
    rng = np.random.RandomState(site_id * 31 + 17)
    return {
        'base_bytes':  rng.randint(50_000,  900_000),
        'n_bursts_mu': rng.uniform(4, 20),
        'iat_scale':   rng.uniform(0.003, 0.05),
        'pkt_mu':      rng.uniform(600, 1200),
        'pkt_sigma':   rng.uniform(150, 400),
        'out_ratio':   rng.uniform(0.04, 0.18),
    }

SITE_PARAMS = [make_site_params(i) for i in range(N_MON)]

def generate_trace(params, noise_level=0.25):
    p = params
    rng = np.random.RandomState()
    base = p['base_bytes'] * np.random.uniform(0.7, 1.3)
    n_bursts = max(2, int(np.random.normal(p['n_bursts_mu'], 2)))
    burst_sizes = np.random.dirichlet(np.ones(n_bursts)) * base

    pkts_in, pkts_out, iats = [], [], []
    for burst in burst_sizes:
        n_in  = max(1, int(burst / p['pkt_mu']))
        n_out = max(1, int(n_in * p['out_ratio']))
        sizes_in  = np.abs(np.random.normal(p['pkt_mu'],  p['pkt_sigma'], n_in)).clip(50, 1500)
        sizes_out = np.abs(np.random.normal(200, 80, n_out)).clip(40, 600)
        # Add noise to simulate real conditions
        sizes_in  += np.random.normal(0, noise_level * p['pkt_sigma'], n_in)
        pkts_in.extend(np.abs(sizes_in))
        pkts_out.extend(np.abs(sizes_out))
        iats.extend(np.abs(np.random.exponential(p['iat_scale'], n_in)))

    return np.array(pkts_in), np.array(pkts_out), np.array(iats), n_bursts

def extract_features(pkts_in, pkts_out, iats, n_bursts):
    def safe(fn, arr, default=0):
        return fn(arr) if len(arr) > 1 else default

    feats = [
        pkts_in.sum(), pkts_out.sum(),
        len(pkts_in), len(pkts_out),
        safe(np.mean, pkts_in), safe(np.mean, pkts_out),
        safe(np.std, pkts_in),  safe(np.std, pkts_out),
        n_bursts,
        pkts_in.sum() / (pkts_out.sum() + 1),
        safe(lambda x: np.percentile(x, 25), pkts_in),
        safe(lambda x: np.percentile(x, 75), pkts_in),
        safe(lambda x: np.percentile(x, 90), pkts_in),
        safe(lambda x: np.percentile(x, 25), pkts_out),
        safe(lambda x: np.percentile(x, 75), pkts_out),
        safe(np.mean, iats), safe(np.std, iats),
        safe(lambda x: np.percentile(x, 10), iats),
        safe(lambda x: np.percentile(x, 50), iats),
        safe(lambda x: np.percentile(x, 90), iats),
        len(pkts_in) / (len(pkts_out) + 1),
        safe(np.max, pkts_in), safe(np.min, pkts_in),
        safe(np.max, iats),
        safe(np.mean, pkts_in[:5]) if len(pkts_in) >= 5 else 0,
        safe(np.mean, pkts_in[-5:]) if len(pkts_in) >= 5 else 0,
    ]
    # First 8 packet sizes (padded)
    padded = np.zeros(8)
    n = min(8, len(pkts_in))
    padded[:n] = pkts_in[:n] / 1500.0
    feats.extend(padded.tolist())
    return np.array(feats, dtype=np.float32)

# Build dataset
X, y = [], []
for site_id in range(N_MON):
    for _ in range(SAMPLES):
        pi, po, iat, nb = generate_trace(SITE_PARAMS[site_id], noise_level=0.3)
        X.append(extract_features(pi, po, iat, nb))
        y.append(site_id)

# Unmonitored: random params (high overlap)
for _ in range(N_UNDEF):
    params = {
        'base_bytes':  np.random.randint(40_000, 1_000_000),
        'n_bursts_mu': np.random.uniform(3, 22),
        'iat_scale':   np.random.uniform(0.002, 0.06),
        'pkt_mu':      np.random.uniform(500, 1300),
        'pkt_sigma':   np.random.uniform(100, 450),
        'out_ratio':   np.random.uniform(0.03, 0.20),
    }
    pi, po, iat, nb = generate_trace(params, noise_level=0.4)
    X.append(extract_features(pi, po, iat, nb))
    y.append(N_MON)

X = np.array(X); y = np.array(y)

# 80/20 split with stratification
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = RandomForestClassifier(n_estimators=200, max_features='sqrt',
                             min_samples_leaf=2, random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)

class_names = [f"Site {i+1:02d}" for i in range(N_MON)] + ["Unmon."]
prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
rec  = recall_score   (y_te, y_pred, average='macro', zero_division=0)
f1   = f1_score       (y_te, y_pred, average='macro', zero_division=0)
print(f"Macro  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

cm = confusion_matrix(y_te, y_pred)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('#0D1117')

# Confusion matrix
ax = axes[0]; ax.set_facecolor('#0D1117')
im = ax.imshow(cm, cmap='YlOrBr', aspect='auto', interpolation='nearest')
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
ticks = range(N_MON + 1)
ax.set_xticks(ticks); ax.set_yticks(ticks)
ax.set_xticklabels(class_names, rotation=90, fontsize=7.5, color='#8B949E')
ax.set_yticklabels(class_names, fontsize=7.5, color='#8B949E')
ax.set_xlabel('Predicted', color='#F0F6FC', fontsize=10)
ax.set_ylabel('True label', color='#F0F6FC', fontsize=10)
ax.set_title('k-FP Confusion Matrix\n(15 monitored + unmonitored, open-world)',
             color='#F0F6FC', fontsize=10, pad=8)
ax.tick_params(colors='#8B949E')
for sp in ax.spines.values(): sp.set_edgecolor('#30363D')

# Per-class precision / recall
ax2 = axes[1]; ax2.set_facecolor('#161B22')
rep = classification_report(y_te, y_pred, target_names=class_names,
                             output_dict=True, zero_division=0)
precs = [rep[c]['precision'] for c in class_names]
recs  = [rep[c]['recall']    for c in class_names]
x = np.arange(len(class_names)); w = 0.38
ax2.bar(x - w/2, precs, w, label='Precision', color='#58A6FF', alpha=0.85)
ax2.bar(x + w/2, recs,  w, label='Recall',    color='#3FB950', alpha=0.85)
ax2.axhline(0.94, color='#E3B341', linestyle='--', linewidth=1.2,
            label='Paper benchmark (94%, closed-world)')
ax2.set_xticks(x)
ax2.set_xticklabels(class_names, rotation=90, fontsize=7.5, color='#8B949E')
ax2.set_ylim(0, 1.18); ax2.set_ylabel('Score', color='#F0F6FC', fontsize=10)
ax2.set_title(f'Per-class Precision & Recall\nMacro F1 = {f1:.3f}  (open-world)',
              color='#F0F6FC', fontsize=10, pad=8)
ax2.legend(facecolor='#161B22', edgecolor='#30363D',
           labelcolor='#F0F6FC', fontsize=8.5)
ax2.tick_params(colors='#8B949E')
for sp in ax2.spines.values(): sp.set_edgecolor('#30363D')

plt.tight_layout(pad=2)
plt.savefig('/home/claude/kfp_results.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("k-FP v2 figure saved.")
