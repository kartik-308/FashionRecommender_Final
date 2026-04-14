"""
Core recommendation logic — all datasets merged into one unified FAISS index
Phase 4: CLIP-based explainability
Phase 5: Accept feedback, BLIP-2 captioning, preference drift detection
"""
import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import CLIPProcessor, CLIPModel
import warnings
warnings.filterwarnings("ignore")

_BASE = r"C:\Users\Kartikeya Singh\OneDrive - LNMIIT\Desktop\archive"

DATASET_CONFIGS = {
    "df2_test": {
        "label":     "DeepFashion2 Test",
        "image_dir": os.path.join(_BASE, "DeepFashion2", "deepfashion2_original_images", "test", "test", "image"),
        "annos_dir": None,
        "csv":       None,
        "source":    "DeepFashion2",
        "cache_dir": os.path.join(_BASE, "DeepFashion2", "deepfashion2_original_images", "test"),
    },
    "df2_train": {
        "label":     "DeepFashion2 Train",
        "image_dir": os.path.join(_BASE, "DeepFashion2", "deepfashion2_original_images", "train", "image"),
        "annos_dir": os.path.join(_BASE, "DeepFashion2", "deepfashion2_original_images", "train", "annos"),
        "csv":       None,
        "source":    "DeepFashion2",
        "cache_dir": os.path.join(_BASE, "DeepFashion2", "deepfashion2_original_images", "train"),
    },
    "df2_val": {
        "label":     "DeepFashion2 Validation",
        "image_dir": os.path.join(_BASE, "DeepFashion2", "deepfashion2_original_images", "validation", "image"),
        "annos_dir": None,
        "csv":       None,
        "source":    "DeepFashion2",
        "cache_dir": os.path.join(_BASE, "DeepFashion2", "deepfashion2_original_images", "validation"),
    },
    "ebay_small": {
        "label":     "eBay Fashion",
        "image_dir": os.path.join(_BASE, "ebayDataset", "ebay_fashion_dataset", "images"),
        "annos_dir": None,
        "csv":       os.path.join(_BASE, "ebayDataset", "ebay_fashion_dataset", "dataset_index.csv"),
        "source":    "eBay",
        "cache_dir": os.path.join(_BASE, "ebayDataset", "ebay_fashion_dataset"),
    },
    "ebay_large": {
        "label":     "Amazon Fashion",
        "image_dir": os.path.join(_BASE, "ebay_fashion_dataset", "images"),
        "annos_dir": None,
        "csv":       os.path.join(_BASE, "ebay_fashion_dataset", "dataset_index.csv"),
        "source":    "Amazon",
        "cache_dir": os.path.join(_BASE, "ebay_fashion_dataset"),
    },
}

BATCH_SIZE  = 64
TOP_K       = 5
ALPHA       = 0.7
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
W1_QUERY    = 0.50
W2_PREF     = 0.30
W3_REDUND   = 0.15
W4_CONSTR   = 0.20
W5_ACCEPT   = 0.25   # bonus for items similar to user-accepted items
BETA        = 0.40
CANDIDATE_K = 200    # candidates pulled per dataset before reranking

# ── Preference drift ─────────────────────────────────────────────────────────
DRIFT_WINDOW = 10    # rolling window of recent accepts to track
DRIFT_THRESH = 0.6   # cosine similarity below which drift is flagged
ALPHA_FAST   = 0.4   # faster adaptation rate when drift detected

# ── Phase 4 attribute probes ──────────────────────────────────────────────────
ATTRIBUTE_GROUPS = {
    "pattern": ["floral pattern","striped pattern","checkered pattern","solid colour",
                "geometric pattern","animal print","polka dot pattern","tie-dye pattern"],
    "fabric":  ["light fabric","heavy fabric","denim fabric","knit fabric",
                "silk fabric","cotton fabric","linen fabric","leather material"],
    "sleeve":  ["short sleeves","long sleeves","sleeveless","three-quarter sleeves",
                "cap sleeves","off-shoulder sleeves","puff sleeves"],
    "fit":     ["slim fit","oversized fit","regular fit","loose fit","bodycon fit"],
    "length":  ["cropped length","full length","midi length","mini length","knee length"],
    "neckline":["v-neck","crew neck","turtleneck","collared neckline","scoop neck"],
    "style":   ["casual style","formal style","sporty style","bohemian style",
                "minimalist style","vintage style","streetwear style"],
    "colour":  ["white clothing","black clothing","blue clothing","red clothing",
                "green clothing","pink clothing","grey clothing","beige clothing"],
}
_attr_text_vecs: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def to_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    for attr in ["pooler_output", "last_hidden_state"]:
        if hasattr(out, attr):
            val = getattr(out, attr)
            return val[:, 0, :] if attr == "last_hidden_state" else val
    for val in vars(out).values():
        if isinstance(val, torch.Tensor):
            return val
    raise ValueError(f"Cannot extract tensor from {type(out)}")


def _cache_paths(key):
    d = DATASET_CONFIGS[key]["cache_dir"]
    return (
        os.path.join(d, f"cache_{key}_embeddings.npy"),
        os.path.join(d, f"cache_{key}_metadata.parquet"),
        os.path.join(d, f"cache_{key}_metadata.csv"),
    )


def _read_meta(pq_path, csv_path):
    """Read metadata preferring parquet, migrating CSV→parquet on first load."""
    if os.path.exists(pq_path):
        return pd.read_parquet(pq_path)
    try:
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8").fillna("")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, dtype=str, encoding="latin-1").fillna("")
    try:
        df.to_parquet(pq_path, index=False)
    except Exception:
        pass
    return df


# ── Per-dataset loader ────────────────────────────────────────────────────────
def _load_one_dataset(key, model, processor):
    cfg     = DATASET_CONFIGS[key]
    img_dir = cfg["image_dir"]
    emb_path, pq_path, csv_path = _cache_paths(key)

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image dir not found: {img_dir}")

    files = sorted([f for f in os.listdir(img_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    # ── Try loading from cache ────────────────────────────────────────────
    if os.path.exists(emb_path) and (os.path.exists(pq_path) or os.path.exists(csv_path)):
        embeddings = np.load(emb_path)
        df_cached  = _read_meta(pq_path, csv_path)
        if len(embeddings) == len(df_cached):
            if "popularity" in df_cached.columns:
                df_cached["popularity"] = pd.to_numeric(
                    df_cached["popularity"], errors="coerce").fillna(0.0)
            df_cached["dataset"]   = key
            df_cached["source"]    = cfg["source"]
            df_cached["full_path"] = df_cached["filename"].apply(
                lambda f: os.path.join(img_dir, f))
            print(f"  [{key}] {len(embeddings):,} loaded")
            return embeddings, df_cached

    # ── Build fresh DataFrame ─────────────────────────────────────────────
    df = pd.DataFrame({
        "filename":  files,
        "full_path": [os.path.join(img_dir, f) for f in files],
        "dataset":   key,
        "source":    cfg["source"],
    })

    annos_dir = cfg.get("annos_dir")
    csv_src   = cfg.get("csv")

    if annos_dir and os.path.exists(annos_dir):
        cats = {}
        for fname in df["filename"]:
            ann_file = os.path.join(annos_dir, os.path.splitext(fname)[0] + ".json")
            if os.path.exists(ann_file):
                try:
                    with open(ann_file) as f:
                        ann = json.load(f)
                    for v in ann.values():
                        if isinstance(v, dict) and "category_name" in v:
                            cats[fname] = v["category_name"]; break
                except Exception:
                    pass
        df["category"] = df["filename"].map(cats).fillna("unknown")
        df["title"] = ""; df["price"] = ""
    elif csv_src and os.path.exists(csv_src):
        meta = pd.read_csv(csv_src, dtype=str).fillna("")
        if "id" in meta.columns:
            meta["filename"] = meta["id"].str.zfill(6) + ".jpg"
        df = df.merge(meta, on="filename", how="left")
        cat_col = next((c for c in ["category","category_name"] if c in df.columns), None)
        df["category"] = df[cat_col].fillna("unknown") if cat_col else "unknown"
        df["title"]    = df["title"].fillna("") if "title" in df.columns else ""
        df["price"]    = df["price"].fillna("") if "price" in df.columns else ""
        if "source_x" in df.columns:
            df["source"] = cfg["source"]
            df.drop(columns=["source_x","source_y"], errors="ignore", inplace=True)
    else:
        df["category"] = "unknown"; df["title"] = ""; df["price"] = ""

    freq = df["category"].value_counts(normalize=True)
    df["popularity"] = df["category"].map(freq).fillna(0.0)

    # ── Encode ────────────────────────────────────────────────────────────
    print(f"  [{key}] encoding {len(df):,} images ...")
    all_embeds = []
    for i in range(0, len(df), BATCH_SIZE):
        batch = df["full_path"].iloc[i:i+BATCH_SIZE].tolist()
        imgs  = []
        for p in batch:
            try:    imgs.append(Image.open(p).convert("RGB"))
            except: imgs.append(Image.new("RGB", (224,224), (180,180,180)))
        inputs = processor(images=imgs, return_tensors="pt", padding=True)
        pv = inputs["pixel_values"].to(DEVICE)
        with torch.no_grad():
            out   = model.get_image_features(pixel_values=pv)
            feats = to_tensor(out)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeds.append(feats.cpu().numpy())

    embeddings = np.vstack(all_embeds).astype("float32")
    np.save(emb_path, embeddings)
    df.to_parquet(pq_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  [{key}] cached {embeddings.shape}")
    return embeddings, df


# ── Unified loader (parallel) ─────────────────────────────────────────────────
def load_all_datasets(model, processor):
    results = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(_load_one_dataset, k, model, processor): k
                   for k in DATASET_CONFIGS}
        for fut in as_completed(futures):
            k = futures[fut]
            try:
                results[k] = fut.result()
            except Exception as e:
                print(f"  [{k}] skipped: {e}")

    if not results:
        raise RuntimeError("No datasets loaded.")

    all_dfs, all_embs = [], []
    for k in DATASET_CONFIGS:          # preserve order
        if k in results:
            emb, df = results[k]
            all_dfs.append(df)
            all_embs.append(emb)

    merged = pd.concat(all_dfs, ignore_index=True)
    for col in ["title","price","source","dataset"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna("").astype(str)
    if "category" in merged.columns:
        merged["category"] = merged["category"].fillna("unknown").astype(str)
        merged["category"] = merged["category"].replace("","unknown")
    if "popularity" in merged.columns:
        merged["popularity"] = pd.to_numeric(
            merged["popularity"], errors="coerce").fillna(0.0)

    emb_matrix = np.vstack(all_embs).astype("float32")

    # Ensure all embeddings are L2-normalized (they should already be from encoding)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8
    emb_matrix = (emb_matrix / norms).astype("float32")

    print(f"\n✅ Unified index: {len(merged):,} images")
    print(merged["dataset"].value_counts().to_string())
    return emb_matrix, merged


# ── CLIP ──────────────────────────────────────────────────────────────────────
def load_clip():
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


def build_faiss_index(embeddings):
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def build_per_source_indexes(embeddings, df):
    """Build a separate FAISS sub-index for each source (DeepFashion2, eBay, Amazon).
    Returns {source_name: (faiss_index, global_indices_array)}."""
    indexes = {}
    for src in df["source"].unique():
        mask  = (df["source"] == src).values
        idxs  = np.where(mask)[0]
        sub   = embeddings[mask].astype("float32")
        dim   = sub.shape[1]
        ix    = faiss.IndexFlatIP(dim)
        ix.add(sub)
        indexes[src] = (ix, idxs)
        print(f"  Sub-index [{src}]: {len(idxs):,} vectors")
    return indexes


# ── Phase 4 explainability (CLIP attribute probes) ───────────────────────────
def precompute_attribute_vectors(model, processor):
    global _attr_text_vecs
    labels = [l for g in ATTRIBUTE_GROUPS.values() for l in g]
    print(f"  Encoding {len(labels)} attribute probes...")
    inputs = processor(text=labels, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out   = model.get_text_features(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=inputs["attention_mask"].to(DEVICE))
        feats = to_tensor(out)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    vecs = feats.cpu().numpy().astype("float32")
    _attr_text_vecs = {l: vecs[i] for i, l in enumerate(labels)}
    print("  Attribute probes ready.")


def explain_result(image_embedding, query_text=None):
    if not _attr_text_vecs:
        return ""
    img = image_embedding / (np.linalg.norm(image_embedding) + 1e-8)
    winners = {}
    for group, labels in ATTRIBUTE_GROUPS.items():
        best, best_s = None, -1.0
        for l in labels:
            if l not in _attr_text_vecs:
                continue
            s = float(np.dot(img, _attr_text_vecs[l]))
            if s > best_s:
                best_s = s; best = l
        if best:
            winners[group] = (best, best_s)
    top = sorted(winners.values(), key=lambda x: -x[1])[:3]
    if not top:
        return ""
    return "Recommended because: " + ", ".join(a[0] for a in top)


# ── Phase 5 explainability (BLIP image captioning) ───────────────────────────
_blip2_model     = None
_blip2_processor = None


def load_blip2():
    """Load BLIP-base (~250M params) for natural-language image captioning.
    Much faster than BLIP-2 OPT-2.7B on CPU (~2-3s vs ~30s per image).
    Falls back gracefully if unavailable."""
    global _blip2_model, _blip2_processor
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        model_id = "Salesforce/blip-image-captioning-base"
        print(f"  Loading BLIP ({model_id})...")
        _blip2_processor = BlipProcessor.from_pretrained(model_id)
        _blip2_model = BlipForConditionalGeneration.from_pretrained(
            model_id
        ).to(DEVICE)
        _blip2_model.eval()
        print("  BLIP captioning ready.")
        return True
    except Exception as e:
        print(f"  BLIP load failed (will use CLIP-only explanations): {e}")
        _blip2_model = None
        _blip2_processor = None
        return False


def blip2_caption(image_path):
    """Generate a natural-language caption for an image using BLIP.
    Returns empty string if BLIP is not loaded or fails."""
    if _blip2_model is None or _blip2_processor is None:
        return ""
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = _blip2_processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ids = _blip2_model.generate(**inputs, max_new_tokens=30)
        caption = _blip2_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return caption
    except Exception:
        return ""


# ── Query encoding ────────────────────────────────────────────────────────────
def encode_query(model, processor, text=None, image_path=None,
                 text_weight=0.5, image_weight=0.5):
    tv = iv = None
    if text:
        inp = processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model.get_text_features(
                input_ids=inp["input_ids"].to(DEVICE),
                attention_mask=inp["attention_mask"].to(DEVICE))
            f = to_tensor(out); f = f / f.norm(dim=-1, keepdim=True)
        tv = f.cpu().numpy()[0].astype("float32")
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        inp = processor(images=[img], return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model.get_image_features(pixel_values=inp["pixel_values"].to(DEVICE))
            f = to_tensor(out); f = f / f.norm(dim=-1, keepdim=True)
        iv = f.cpu().numpy()[0].astype("float32")
    if tv is not None and iv is not None:
        c = text_weight*tv + image_weight*iv
        return c/np.linalg.norm(c), f"text+image ({int(text_weight*100)}/{int(image_weight*100)})"
    if tv is not None: return tv, "text only"
    if iv is not None: return iv, "image only"
    raise ValueError("Provide text or image_path")


# ── Preference tracker with accept feedback & drift detection ─────────────────
class PreferenceTracker:
    def __init__(self, alpha=ALPHA):
        self.alpha       = alpha
        self.base_alpha  = alpha         # original alpha, restored after drift fades
        self.pref_vec    = None
        self.rejected    = set()
        self.accepted    = set()         # filenames the user explicitly liked
        self.shown       = []
        self.accept_vecs = []            # rolling window of accepted embeddings
        self.drift_detected = False
        self.drift_score    = 1.0        # 1.0 = no drift, lower = more drift

    # ── Core update (called on every search with query vec) ──────────────
    def update(self, vec):
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        if self.pref_vec is None:
            self.pref_vec = vec.copy()
        else:
            self.pref_vec = self.alpha * self.pref_vec + (1 - self.alpha) * vec
            self.pref_vec /= (np.linalg.norm(self.pref_vec) + 1e-8)

    # ── Accept: strong positive signal ───────────────────────────────────
    def accept(self, vec, filename):
        """User explicitly liked this item — boost preference toward it."""
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        self.accepted.add(filename)

        # Stronger update than passive shown: weight = 0.5 (vs 1-alpha ~ 0.3)
        if self.pref_vec is None:
            self.pref_vec = vec.copy()
        else:
            self.pref_vec = 0.5 * self.pref_vec + 0.5 * vec
            self.pref_vec /= (np.linalg.norm(self.pref_vec) + 1e-8)

        # Track for drift detection
        self.accept_vecs.append(vec)
        if len(self.accept_vecs) > DRIFT_WINDOW:
            self.accept_vecs = self.accept_vecs[-DRIFT_WINDOW:]

        # Check for preference drift
        self._check_drift()

    # ── Drift detection ──────────────────────────────────────────────────
    def _check_drift(self):
        """Compare the current preference vector with recent accepts.
        If they diverge, the user's taste is shifting — increase learning rate."""
        if len(self.accept_vecs) < 3 or self.pref_vec is None:
            self.drift_detected = False
            self.drift_score = 1.0
            return

        # Mean of recent accepts
        recent_mean = np.mean(self.accept_vecs[-DRIFT_WINDOW:], axis=0)
        recent_mean = recent_mean / (np.linalg.norm(recent_mean) + 1e-8)

        # Cosine similarity between preference vector and recent accepts
        sim = float(np.dot(self.pref_vec, recent_mean))
        self.drift_score = round(sim, 4)

        if sim < DRIFT_THRESH:
            self.drift_detected = True
            self.alpha = ALPHA_FAST   # speed up adaptation
        else:
            self.drift_detected = False
            self.alpha = self.base_alpha  # restore normal rate

    def add_shown(self, emb):
        self.shown.append(emb / (np.linalg.norm(emb) + 1e-8))

    def reject(self, filename):
        self.rejected.add(filename)

    def get(self):
        return self.pref_vec

    def redundancy_score(self, v):
        if not self.shown:
            return 0.0
        v = v / (np.linalg.norm(v) + 1e-8)
        return max(float(np.dot(v, s)) for s in self.shown)

    def rejection_penalty(self, f):
        return 1.0 if f in self.rejected else 0.0

    def accept_boost(self, v):
        """Compute how similar a candidate is to user-accepted items.
        Returns 0.0 if no accepts yet."""
        if not self.accept_vecs:
            return 0.0
        v = v / (np.linalg.norm(v) + 1e-8)
        return max(float(np.dot(v, a)) for a in self.accept_vecs)

    # ── Serialization ────────────────────────────────────────────────────
    def to_dict(self):
        return {
            "pref_vec":       self.pref_vec.tolist() if self.pref_vec is not None else None,
            "rejected":       list(self.rejected),
            "accepted":       list(self.accepted),
            "shown":          [s.tolist() for s in self.shown],
            "accept_vecs":    [a.tolist() for a in self.accept_vecs],
            "drift_detected": self.drift_detected,
            "drift_score":    self.drift_score,
            "alpha":          self.alpha,
        }

    @classmethod
    def from_dict(cls, d, alpha=ALPHA):
        saved_alpha = d.get("alpha", alpha)
        t = cls(alpha=saved_alpha)
        t.base_alpha = alpha
        t.pref_vec = np.array(d["pref_vec"], dtype="float32") if d.get("pref_vec") else None
        t.rejected = set(d.get("rejected", []))
        t.accepted = set(d.get("accepted", []))
        t.shown    = [np.array(s, dtype="float32") for s in d.get("shown", [])]
        t.accept_vecs = [np.array(a, dtype="float32") for a in d.get("accept_vecs", [])]
        t.drift_detected = d.get("drift_detected", False)
        t.drift_score    = d.get("drift_score", 1.0)
        return t


# ── Reranker with per-source normalization, accept boost & diversity ──────────
def rerank(candidates_df, candidate_indices, all_embeddings, query_vec, tracker):
    """Score candidates, normalize scores per-source so no dataset dominates
    by raw cosine scale, then select with proportional diversity."""
    pref_vec = tracker.get()
    rows_out = []
    for idx, row in zip(candidate_indices, candidates_df.itertuples()):
        iv  = all_embeddings[idx]; iv = iv / (np.linalg.norm(iv) + 1e-8)
        sq  = float(np.dot(query_vec, iv))
        sp  = float(np.dot(pref_vec, iv)) if pref_vec is not None else 0.0
        rd  = tracker.redundancy_score(iv)
        rp  = tracker.rejection_penalty(row.filename)
        ab  = tracker.accept_boost(iv)
        pop = float(getattr(row, "popularity", 0.5))
        rows_out.append({
            "filename":     row.filename,
            "full_path":    row.full_path,
            "category":     str(getattr(row, "category", "---")),
            "title":        str(getattr(row, "title", "")),
            "price":        str(getattr(row, "price", "")),
            "source":       str(getattr(row, "source", "")),
            "dataset":      str(getattr(row, "dataset", "")),
            "popularity":   round(pop, 4),
            "sim_query":    round(sq,  4),
            "sim_pref":     round(sp,  4),
            "redundancy":   round(rd,  4),
            "rej_penalty":  round(rp,  4),
            "accept_boost": round(ab,  4),
        })

    res = pd.DataFrame(rows_out)
    res = res[~res["filename"].isin(tracker.rejected)].copy()
    if res.empty:
        return res.reset_index(drop=True)

    # ── Per-source score normalization ────────────────────────────────────
    # Normalize sim_query to [0, 1] within each source so that each source
    # competes on relative ranking rather than raw cosine magnitude.
    res["norm_sim_query"] = 0.0
    res["norm_sim_pref"]  = 0.0
    for src in res["source"].unique():
        mask = res["source"] == src
        # Normalize sim_query
        sq_vals = res.loc[mask, "sim_query"]
        sq_min, sq_max = sq_vals.min(), sq_vals.max()
        rng = sq_max - sq_min
        if rng > 1e-8:
            res.loc[mask, "norm_sim_query"] = (sq_vals - sq_min) / rng
        else:
            res.loc[mask, "norm_sim_query"] = 1.0  # all identical → top rank
        # Normalize sim_pref
        sp_vals = res.loc[mask, "sim_pref"]
        sp_min, sp_max = sp_vals.min(), sp_vals.max()
        rng_p = sp_max - sp_min
        if rng_p > 1e-8:
            res.loc[mask, "norm_sim_pref"] = (sp_vals - sp_min) / rng_p
        else:
            res.loc[mask, "norm_sim_pref"] = 0.5

    # Compute final_score using normalized similarities + accept boost
    res["final_score"] = (
        W1_QUERY  * res["norm_sim_query"]
        + W2_PREF * res["norm_sim_pref"]
        - W3_REDUND * res["redundancy"]
        - BETA * res["rej_penalty"]
        + W4_CONSTR * res["popularity"]
        + W5_ACCEPT * res["accept_boost"]
    ).round(4)

    res = res.sort_values("final_score", ascending=False)

    # ── Proportional diversity selection ──────────────────────────────────
    # Each source gets at least floor(TOP_K / num_sources) slots;
    # remaining slots go to the best-scoring items across all sources.
    sources = res["source"].unique().tolist()
    n_src   = len(sources)
    if n_src == 0:
        return res.head(TOP_K).reset_index(drop=True)

    base_per_src = max(1, TOP_K // n_src)
    selected, used = [], set()

    # Phase 1: guaranteed allocation per source
    for src in sources:
        src_rows = res[res["source"] == src]
        count = 0
        for idx2, row in src_rows.iterrows():
            if count >= base_per_src:
                break
            if row["filename"] not in used:
                selected.append(idx2)
                used.add(row["filename"])
                count += 1

    # Phase 2: fill remaining slots by best normalized score (no cap)
    for idx2, row in res.iterrows():
        if len(selected) >= TOP_K:
            break
        if row["filename"] not in used:
            selected.append(idx2)
            used.add(row["filename"])

    out = res.loc[selected].sort_values("final_score", ascending=False).reset_index(drop=True)
    # Drop internal columns before returning
    out.drop(columns=["norm_sim_query", "norm_sim_pref", "rej_penalty", "accept_boost"],
             inplace=True, errors="ignore")
    return out


def retrieve_and_rerank(query_vec, index, df, all_embeddings, tracker,
                        per_source_indexes=None):
    """Stratified retrieval: query each per-source sub-index separately so
    every dataset contributes candidates regardless of size imbalance.
    Falls back to unified index if per_source_indexes is not provided."""
    tracker.update(query_vec)
    pref  = tracker.get()
    fused = 0.6*query_vec + 0.4*pref if pref is not None else query_vec
    fused = (fused / np.linalg.norm(fused)).reshape(1,-1).astype("float32")

    if per_source_indexes:
        # ── Stratified retrieval: per-source sub-index search ─────────────
        all_cand_idxs = []
        for src, (sub_ix, global_idxs) in per_source_indexes.items():
            k = min(CANDIDATE_K, sub_ix.ntotal)
            if k == 0:
                continue
            _, local_idxs = sub_ix.search(fused, k)
            # Map local sub-index positions back to global indices
            mapped = global_idxs[local_idxs[0]]
            all_cand_idxs.append(mapped)
        cand_idxs = np.unique(np.concatenate(all_cand_idxs)) if all_cand_idxs else np.array([], dtype=int)
    else:
        # ── Fallback: unified index ───────────────────────────────────────
        _, idxs   = index.search(fused, CANDIDATE_K)
        cand_idxs = idxs[0]

    if len(cand_idxs) == 0:
        return pd.DataFrame()

    candidates_df = df.iloc[cand_idxs].reset_index(drop=True)
    results       = rerank(candidates_df, cand_idxs, all_embeddings, query_vec, tracker)

    for _, row in results.iterrows():
        m = df[df["filename"] == row["filename"]].index
        if len(m): tracker.add_shown(all_embeddings[m[0]])

    return results
