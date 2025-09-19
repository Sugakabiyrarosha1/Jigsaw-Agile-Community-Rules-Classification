# ============================================================
# Jigsaw — SCRATCH Hyperparameter Tuning (Grid/Random, Resumable)
# - Expand PARAM_SPACE -> combos
# - Skip already-seen combos via trial_results_scratch.csv
# - Train with best-epoch weight capture, optional early stopping & TensorBoard
# - (Optional) save per-trial checkpoints and write a submission for the best AUC
# ============================================================

import os, re, json, time, random, hashlib, platform
from datetime import datetime, timezone
from itertools import product
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# ------------------- Paths & switches -------------------
TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"
SUB_PATH   = "sample_submission.csv"

RESULTS_CSV = "trial_results_scratch.csv"    # resumable log (append)
CHECKPOINT_DIR = "checkpoints_scratch"       # per-trial .pt files (optional)
SAVE_CHECKPOINTS = True

ENABLE_TENSORBOARD = False
EARLY_STOP_PATIENCE = 3       # 0 to disable
MAX_TRIALS_PER_RUN  = 10      # safety cap for a single session
SAVE_BEST_SESSION_SUBMISSION = True
SUBMISSION_CSV  = "submission_scratch.csv"
SUBMISSION_XLSX = "submission_scratch.xlsx"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ------------------- Load data -------------------
assert os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH) and os.path.exists(SUB_PATH), \
    "Place train.csv, test.csv, sample_submission.csv in the working directory."

TEXT_COLS = ['body','rule','subreddit','positive_example_1','positive_example_2','negative_example_1','negative_example_2']
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

for df in [train_df, test_df]:
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()

def build_input_template(row):
    return " [SEP] ".join([
        f"[COMMENT] {row['body']}",
        f"[RULE] {row['rule']}",
        f"[POS_EX_1] {row['positive_example_1']}",
        f"[POS_EX_2] {row['positive_example_2']}",
        f"[NEG_EX_1] {row['negative_example_1']}",
        f"[NEG_EX_2] {row['negative_example_2']}",
        f"[SUBREDDIT] r/{row['subreddit']}"
    ])

if "input_text" not in train_df.columns:
    train_df["input_text"] = train_df.apply(build_input_template, axis=1)
    test_df["input_text"]  = test_df.apply(build_input_template, axis=1)

# ------------------- Utils -------------------
def set_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def now_iso(): return datetime.now(timezone.utc).isoformat()

def combo_key(params:Dict[str,Any])->str:
    s = json.dumps({k:params[k] for k in sorted(params)}, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def load_done_keys(path:str)->set:
    if not os.path.exists(path): return set()
    try:
        df = pd.read_csv(path)
        return set(df["key"].astype(str).tolist()) if "key" in df.columns else set()
    except Exception:
        return set()

def append_result_row(row:Dict[str,Any], path=RESULTS_CSV):
    df = pd.DataFrame([row], columns=list(row.keys()))
    if os.path.exists(path): df.to_csv(path, mode="a", header=False, index=False)
    else:                    df.to_csv(path, index=False)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------- Tokenizer/Vocab -------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
def tokenize(s): return TOKEN_RE.findall((s or "").lower())

VOCAB_CACHE: Dict[int, Dict[str,int]] = {}
def build_vocab(df:pd.DataFrame, vocab_size:int=30000)->Dict[str,int]:
    if vocab_size in VOCAB_CACHE: return VOCAB_CACHE[vocab_size]
    from collections import Counter
    cnt = Counter()
    for col in ["body","rule"]:
        for txt in df[col].tolist():
            cnt.update(tokenize(txt))
    vocab = {"<pad>":0, "<unk>":1}
    for i,(tok,_) in enumerate(cnt.most_common(vocab_size-2), start=2):
        vocab[tok] = i
    VOCAB_CACHE[vocab_size] = vocab
    return vocab

def encode_text(s, vocab, max_len):
    ids = [vocab.get(t,1) for t in tokenize(s)][:max_len]
    if len(ids) < max_len: ids += [0]*(max_len-len(ids))
    return np.array(ids, dtype=np.int64)

class ScratchDataset(Dataset):
    def __init__(self, df, vocab, seq_len, with_labels=True):
        self.df=df.reset_index(drop=True); self.vocab=vocab; self.seq_len=seq_len; self.with_labels=with_labels
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.loc[i]
        half = self.seq_len//2
        x = np.concatenate([encode_text(r["body"], self.vocab, half),
                            encode_text(r["rule"], self.vocab, half)])
        if self.with_labels:
            y = int(r["rule_violation"])
            return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float32)
        return torch.tensor(x, dtype=torch.long)

# ------------------- Model -------------------
def parse_kernel_sizes(spec:str):
    ks = []
    for k in str(spec).split("-"):
        k = k.strip()
        if k.isdigit(): ks.append(int(k))
    return ks or [3,5]

def channel_schedule(start:int, blocks:int, growth:str):
    chs = [start]
    for _ in range(1, blocks):
        if growth == "x1.5": chs.append(int(round(chs[-1]*1.5)))
        elif growth == "x2": chs.append(chs[-1]*2)
        else:                chs.append(chs[-1])
    return chs

class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, conv_blocks, channels_start,
                 channel_growth, kernel_sizes_spec, use_batchnorm=True,
                 pooling="max", dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        ks = parse_kernel_sizes(kernel_sizes_spec)
        chs = channel_schedule(channels_start, conv_blocks, channel_growth)
        self.blocks = nn.ModuleList()
        in_ch = emb_dim
        for bi in range(conv_blocks):
            k = ks[min(bi, len(ks)-1)]
            out_ch = chs[bi]
            conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2)
            bn   = nn.BatchNorm1d(out_ch) if use_batchnorm else nn.Identity()
            self.blocks.append(nn.Sequential(conv, bn, nn.ReLU()))
            in_ch = out_ch
        self.pooling = pooling
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x):
        e = self.emb(x).transpose(1,2)   # [B,E,L]
        h = e
        for blk in self.blocks: h = blk(h)
        if self.pooling == "avg": h = F.adaptive_avg_pool1d(h,1).squeeze(-1)
        else:                     h = F.adaptive_max_pool1d(h,1).squeeze(-1)
        h = self.drop(h)
        return self.fc(h).squeeze(-1)

# ------------------- Loss/Optim/Val -------------------
class BCEWithLS(nn.Module):
    def __init__(self, smoothing=0.0): super().__init__(); self.s=smoothing
    def forward(self, logits, targets):
        if self.s>0: targets = targets*(1-self.s)+0.5*self.s
        return F.binary_cross_entropy_with_logits(logits, targets)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, smoothing=0.0): super().__init__(); self.g=gamma; self.s=smoothing
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        if self.s>0: targets = targets*(1-self.s)+0.5*self.s
        loss_pos = -targets * ((1-p)**self.g) * torch.log(torch.clamp(p, 1e-8, 1.0))
        loss_neg = -(1-targets) * (p**self.g) * torch.log(torch.clamp(1-p, 1.0-1e-8))
        return (loss_pos+loss_neg).mean()

def get_loss(name, smoothing):
    return FocalLoss(2.0, smoothing) if name=="focal" else BCEWithLS(smoothing)

def make_optimizer(model, name, lr, weight_decay):
    if name == "adamw": return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd": return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else: raise ValueError(f"Unknown optimizer: {name}")

def _epoch_validate(model, dl, device="cpu"):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb,yb in dl:
            xb,yb = xb.to(device), yb.to(device)
            p = torch.sigmoid(model(xb)).detach().cpu().numpy()
            preds.append(p); ys.append(yb.detach().cpu().numpy())
    preds = np.concatenate(preds); ys = np.concatenate(ys)
    auc = roc_auc_score(ys, preds)
    acc = accuracy_score(ys.astype(int), (preds >= 0.5).astype(int))
    return auc, acc, preds, ys

# ------------------- Train one combo -------------------
def train_eval_once_with_best(params:dict, enable_tb:bool=False):
    set_seed(int(params["seed"]))
    vocab = build_vocab(train_df, int(params["vocab_size"]))
    seq_len = int(params["seq_len"])
    tr, va = train_test_split(train_df, test_size=0.2, random_state=int(params["seed"]),
                              stratify=train_df["rule_violation"])
    ds_tr = ScratchDataset(tr, vocab, seq_len, True)
    ds_va = ScratchDataset(va, vocab, seq_len, True)
    dl_tr = DataLoader(ds_tr, batch_size=int(params["batch_size"]), shuffle=True, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=int(params["batch_size"]), shuffle=False, num_workers=2)

    model = TextCNN(
        vocab_size=len(vocab),
        emb_dim=int(params["emb_dim"]),
        conv_blocks=int(params["conv_blocks"]),
        channels_start=int(params["channels_start"]),
        channel_growth=str(params["channel_growth"]),
        kernel_sizes_spec=str(params["kernel_sizes"]),
        use_batchnorm=bool(params["use_batchnorm"]),
        pooling=str(params["pooling"]),
        dropout=float(params["dropout"])
    ).to(DEVICE)

    opt = make_optimizer(model, str(params["optimizer"]), float(params["learning_rate"]), float(params["weight_decay"]))
    loss_fn = get_loss(str(params["loss_fn"]), float(params["label_smoothing"]))
    grad_clip = float(params["grad_clip"])
    epochs = int(params["epochs"])

    pos_weight = None
    if str(params["class_weighting"])=="balanced":
        pos_weight = torch.tensor([(len(tr)-tr["rule_violation"].sum())/(tr["rule_violation"].sum()+1e-6)], device=DEVICE)

    tb = None
    if enable_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            run_name = f"tb_scratch_tune/emb{params['emb_dim']}_cb{params['conv_blocks']}_ch{params['channels_start']}_lr{params['learning_rate']}_bs{params['batch_size']}_{datetime.now(timezone.utc).isoformat()}"
            tb = SummaryWriter(log_dir=run_name)
        except Exception as e:
            print("TensorBoard unavailable:", e)

    best_auc, best_acc, best_state = -1.0, 0.0, None
    global_step = 0
    no_improve = 0

    for ep in range(epochs):
        model.train()
        for xb,yb in dl_tr:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            if pos_weight is not None:
                loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            else:
                loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if tb: tb.add_scalar("train/loss", float(loss.item()), global_step); global_step += 1

        auc, acc, _, _ = _epoch_validate(model, dl_va, device=DEVICE)
        improved = auc > best_auc + 1e-5
        if improved:
            best_auc, best_acc = auc, acc
            best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        msg = f"[SCRATCH] Epoch {ep+1}/{epochs} AUC={auc:.5f} ACC={acc:.4f} (best {best_auc:.5f}, patience {no_improve}/{EARLY_STOP_PATIENCE})"
        print(msg)
        if tb:
            tb.add_scalar("val/auc", float(auc), ep)
            tb.add_scalar("val/accuracy", float(acc), ep)

        if EARLY_STOP_PATIENCE and no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping: no improvement.")
            break

    if tb: tb.close()
    return best_auc, best_acc, best_state, vocab

# ------------------- Predict test with a state -------------------
def predict_test_with_state(best_state, params, vocab, out_csv="submission_scratch.csv"):
    seq_len = int(params["seq_len"])
    class TestDS(Dataset):
        def __init__(self, df, vocab, seq_len):
            self.df=df.reset_index(drop=True); self.vocab=vocab; self.seq_len=seq_len
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            r = self.df.loc[i]
            half = self.seq_len//2
            x = np.concatenate([encode_text(r["body"], self.vocab, half),
                                encode_text(r["rule"], self.vocab, half)])
            return torch.tensor(x, dtype=torch.long)

    test_ds = TestDS(test_df, vocab, seq_len)
    test_dl = DataLoader(test_ds, batch_size=int(params["batch_size"]), shuffle=False, num_workers=2)

    model = TextCNN(
        vocab_size=len(vocab),
        emb_dim=int(params["emb_dim"]),
        conv_blocks=int(params["conv_blocks"]),
        channels_start=int(params["channels_start"]),
        channel_growth=str(params["channel_growth"]),
        kernel_sizes_spec=str(params["kernel_sizes"]),
        use_batchnorm=bool(params["use_batchnorm"]),
        pooling=str(params["pooling"]),
        dropout=float(params["dropout"])
    ).to(DEVICE)
    model.load_state_dict({k: v.to(DEVICE) for k,v in best_state.items()})
    model.eval()

    preds = []
    with torch.no_grad():
        for xb in test_dl:
            xb = xb.to(DEVICE)
            p = torch.sigmoid(model(xb)).detach().cpu().numpy()
            preds.append(p)
    preds = np.concatenate(preds).reshape(-1)

    sub = pd.read_csv(SUB_PATH).copy()
    if "id" not in sub.columns: sub["id"] = np.arange(len(preds))
    sub["rule_violation"] = np.clip(preds, 0, 1)
    sub.to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv} (rows={len(sub)})")
    return out_csv

# ------------------- Param space handling -------------------
CONSTANTS_DEFAULT = {
    "vocab_size": 30000,
    "use_batchnorm": True,
    "pooling": "max",
    "optimizer": "adamw",
    "grad_clip": 1.0,
    "scheduler": "none",        # catalog only
    "class_weighting": "none",
    "seed": 42
}
REQ = ['seq_len','emb_dim','conv_blocks','channels_start','channel_growth','kernel_sizes',
       'dropout','weight_decay','label_smoothing','learning_rate','batch_size','epochs','loss_fn']
INTS   = ["seq_len","emb_dim","conv_blocks","channels_start","batch_size","epochs","seed"]
FLOATS = ["dropout","weight_decay","label_smoothing","learning_rate","grad_clip"]
STRS   = ["channel_growth","kernel_sizes","pooling","optimizer","scheduler","loss_fn","class_weighting"]
BOOLS  = ["use_batchnorm"]

def coerce_one(p:Dict[str,Any])->Dict[str,Any]:
    x = {**CONSTANTS_DEFAULT, **p}
    missing = [k for k in REQ if k not in x]
    if missing: raise KeyError(f"Missing required param(s): {missing}")
    for k in INTS:   x[k] = int(x[k])
    for k in FLOATS: x[k] = float(x[k])
    for k in STRS:   x[k] = str(x[k])
    for k in BOOLS:
        v = x[k]; x[k] = (v.strip().lower() in ("true","1","yes","y")) if isinstance(v,str) else bool(v)
    return x

def expand_grid(space:Dict[str,List[Any]], shuffle=True, seed=42)->List[Dict[str,Any]]:
    keys = list(space.keys())
    vals = [space[k] if isinstance(space[k], (list, tuple)) else [space[k]] for k in keys]
    combos = []
    for tup in product(*vals):
        combos.append({k:v for k,v in zip(keys, tup)})
    if shuffle:
        rnd = random.Random(seed); rnd.shuffle(combos)
    return combos

# ------------------- Tuner (grid/random + resume) -------------------
def run_param_space(space:Dict[str,List[Any]],
                    constants:Dict[str,Any]=None,
                    mode:str="grid",        # "grid" or "random"
                    n_samples:int=None,     # used when mode="random"
                    max_trials:int=10,
                    enable_tb:bool=False,
                    save_best_submission:bool=True):
    constants = constants or {}
    full_space = {**space, **constants} if constants else space
    grid = expand_grid(space, shuffle=True, seed=int(constants.get("seed", 42)))
    if mode == "random" and n_samples is not None:
        grid = grid[:n_samples]  # because we shuffled already

    done = load_done_keys(RESULTS_CSV)
    print(f"Total combos: {len(grid)} | Completed in CSV: {len(done)}")

    best_auc = -1.0
    best_payload = None
    ran = 0
    t0 = time.time()

    for idx, raw in enumerate(grid):
        params = coerce_one({**raw, **constants})
        key = combo_key(params)
        if key in done:
            continue

        print(f"\n=== Trial {ran+1}/{max_trials} | idx={idx} ===")
        print({k: params[k] for k in REQ})

        t1 = time.time()
        try:
            auc, acc, state, vocab = train_eval_once_with_best(params, enable_tb)
            status = "ok"
            # Optional: save per-trial checkpoint
            if SAVE_CHECKPOINTS and state is not None:
                torch.save({"state_dict": state, "params": params},
                           os.path.join(CHECKPOINT_DIR, f"{key}.pt"))
        except Exception as e:
            auc, acc = float("nan"), float("nan")
            state, vocab = None, None
            status = f"error: {e}"
            print("❌", e)
        dur = time.time() - t1

        row_out = {
            "timestamp": now_iso(),
            "key": key,
            "mode": "scratch",
            "device": DEVICE,
            "python": platform.python_version(),
            "grid_idx": idx,
            "val_auc": auc,
            "val_acc": acc,
            "runtime_sec": round(dur,2),
            "status": status,
            **{f"hp/{k}": params[k] for k in sorted(params)}
        }
        append_result_row(row_out, RESULTS_CSV)
        ran += 1

        if status == "ok" and auc > best_auc:
            best_auc = auc
            best_payload = (state, params, vocab)

        if ran >= max_trials:
            break

    print(f"\nSession done. Ran {ran} trial(s) in {round(time.time()-t0,2)}s.")
    if best_payload and save_best_submission:
        state, params, vocab = best_payload
        predict_test_with_state(state, params, vocab, out_csv=SUBMISSION_CSV)
        try:
            sub_df = pd.read_csv(SUBMISSION_CSV)
            with pd.ExcelWriter(SUBMISSION_XLSX, engine="xlsxwriter") as w:
                sub_df.to_excel(w, sheet_name="submission", index=False)
            print(f"✅ Wrote {SUBMISSION_XLSX}")
        except Exception as e:
            print("Note: could not write XLSX submission:", e)
    else:
        print("No submission written this session.")

# ============================================================
# DEFINE YOUR PARAM SPACE HERE
# Keep it sane for a laptop; resume lets you add more later.
# ============================================================
PARAM_SPACE = dict(
    # Capacity/structure
    seq_len=[200, 224, 256],
    emb_dim=[128, 160, 192],
    conv_blocks=[1, 2],
    channels_start=[128, 160],
    channel_growth=["x1.5"],
    kernel_sizes=["3-5-7"],        # keep as TEXT (avoid Excel auto-dates)

    # Optimization/regularization
    optimizer=["adamw"],
    learning_rate=[8e-4, 1e-3, 1.2e-3],
    batch_size=[64, 128],
    epochs=[8],
    dropout=[0.2, 0.25],
    weight_decay=[1e-4, 2e-4],
    label_smoothing=[0.0, 0.03],
    loss_fn=["bce_logits"],

    # Fixed via constants below but left here if you want to vary
    # vocab_size=[30000],
    # use_batchnorm=[True],
    # pooling=["max"],
    # grad_clip=[1.0],
    # scheduler=["none"],
    # class_weighting=["none"],
    # seed=[42],
)

# Constants override (applied to every combo; can change)
CONSTANTS = dict(
    vocab_size=30000,
    use_batchnorm=True,
    pooling="max",
    grad_clip=1.0,
    scheduler="none",
    class_weighting="none",
    seed=42,
)

# ============================================================
# GO: run grid (or random sample) with resume
# ============================================================
if __name__ == "__main__":
    # mode="grid" runs the full Cartesian product (skips done via CSV)
    # Set MAX_TRIALS_PER_RUN for this session size; you can rerun later to continue.
    run_param_space(
        PARAM_SPACE,
        constants=CONSTANTS,
        mode="grid",             # or "random"
        n_samples=None,          # only used for mode="random"
        max_trials=MAX_TRIALS_PER_RUN,
        enable_tb=ENABLE_TENSORBOARD,
        save_best_submission=SAVE_BEST_SESSION_SUBMISSION
    )
