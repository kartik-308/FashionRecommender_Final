"""
Flask backend — multi-dataset fashion recommender with auth
"""
import os
import uuid
import base64
from io import BytesIO
from datetime import datetime
from functools import wraps
from flask import (Flask, request, jsonify, session,
                   redirect, url_for, render_template)
from PIL import Image
import recommender as rec
import auth

# ── App Initialization ────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static", template_folder="templates")

# Secret key signs and verifies session cookies — MUST be changed in production
app.secret_key = "fashion-finder-secret-key-change-in-prod"

# Reject any uploaded file larger than 16 MB before it even hits the route handler
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Directory for temporarily storing user-uploaded query images
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # create if it doesn't exist


# ── Global Model State ────────────────────────────────────────────────────────
# These globals are loaded once at startup and shared across all requests.
# This avoids reloading heavy ML models on every API call.

_index      = None   # FAISS vector index for fast similarity search
_df         = None   # Pandas DataFrame with image metadata (filename, category, price...)
_embeddings = None   # NumPy array of image embedding vectors (one per image)
_model      = None   # CLIP vision-language model
_processor  = None   # CLIP input preprocessor (tokenizer + image transforms)
_ready      = False  # True only after all models and indexes are fully loaded
_error      = None   # Stores any error message from failed initialization
_total_images = 0    # Total number of images across all datasets
_per_source_indexes = None  # Per-source FAISS sub-indexes for stratified retrieval
_blip2_ready = False  # True if BLIP-2 loaded successfully


def init_models():
    """
    Load all ML models and build the search index.
    Called once at startup (before the server begins accepting requests).
    Sets _ready=True on success, or stores the error message on failure.
    """
    global _model, _processor, _index, _df, _embeddings, _ready, _error, _total_images, _per_source_indexes, _blip2_ready
    try:
        print("Loading CLIP...")
        _model, _processor = rec.load_clip()  # downloads/loads CLIP weights
        print(f"  CLIP ready on {rec.DEVICE.upper()}")

        # Load every configured dataset and compute (or load cached) embeddings
        _embeddings, _df = rec.load_all_datasets(_model, _processor)

        # Build a FAISS index over all embeddings for sub-millisecond nearest-neighbor search
        _index = rec.build_faiss_index(_embeddings)
        _per_source_indexes = rec.build_per_source_indexes(_embeddings, _df)
        _total_images = len(_df)

        # Pre-encode attribute descriptor words (e.g. "floral", "casual", "slim-fit")
        # so they don't need to be re-encoded on every search request
        rec.precompute_attribute_vectors(_model, _processor)

        # Load BLIP-2 for natural-language image captioning (non-blocking)
        _blip2_ready = rec.load_blip2()

        _ready = True
        print(f"Unified FAISS index: {_total_images:,} images")
        if _blip2_ready:
            print("BLIP-2 captioning: enabled")
        else:
            print("BLIP-2 captioning: disabled (CLIP-only explanations)")
    except Exception as e:
        _error = str(e)
        print(f"Init failed: {e}")


# ── Auth Decorators ───────────────────────────────────────────────────────────

def login_required(f):
    """
    Route decorator: redirects to the login page if the user has no active session.
    Usage: @login_required above any route that needs authentication.
    """
    @wraps(f)  # preserves the wrapped function's __name__, required by Flask's routing
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """
    Route decorator: redirects to home if the current user is not an admin.
    Must be used AFTER @login_required so session is guaranteed to exist.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "admin":
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated


# ── Utility Functions ─────────────────────────────────────────────────────────

def img_to_b64(path):
    """
    Load an image from disk, resize it to at most 400×400 (preserving aspect ratio),
    and return a base64-encoded JPEG data URI suitable for embedding in JSON responses.
    Returns None if the image can't be read (e.g. file missing or corrupt).
    """
    try:
        img = Image.open(path).convert("RGB")   # ensure 3-channel RGB
        img.thumbnail((400, 400))               # in-place resize, keeps aspect ratio
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)  # quality=85 balances size vs clarity
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def get_tracker():
    """
    Load the current user's preference tracker from the database.
    If no tracker has been saved yet, returns a fresh (empty) one.
    The tracker remembers liked/disliked items to personalize result ranking.
    """
    username = session.get("username")
    if username:
        d = auth.load_tracker(username, "unified")
        if d:
            return rec.PreferenceTracker.from_dict(d)  # deserialize from saved dict
    return rec.PreferenceTracker()  # default empty tracker for guests or first-timers


def save_tracker_for_user(tracker):
    """
    Persist the in-memory tracker back to the database for the current user.
    Called after every search and every rejection to keep preferences up to date.
    """
    username = session.get("username")
    if username:
        auth.save_tracker(username, "unified", tracker.to_dict())


def _build_ds_info():
    """
    Build a list of dataset summaries [{key, label, count}, …] for the admin dashboard.
    - If models are loaded: counts are read from the live DataFrame (fast, accurate).
    - If models aren't ready yet: counts are estimated by scanning the image directories.
    """
    result = []
    for key, cfg in rec.DATASET_CONFIGS.items():
        if _ready and _df is not None and "dataset" in _df.columns:
            # Fast path: count rows in the loaded DataFrame
            count = int((_df["dataset"] == key).sum())
        else:
            # Fallback: count image files on disk
            try:
                img_dir = cfg.get("image_dir", "")
                count = len([f for f in os.listdir(img_dir)
                             if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            except Exception:
                count = 0
        result.append({"key": key, "label": cfg["label"], "count": count})
    return result


# ── Auth Routes ───────────────────────────────────────────────────────────────

@app.route("/login")
def login_page():
    """
    Landing login page. Redirects already-authenticated users directly to their
    appropriate dashboard (admin → admin dashboard, user → home).
    """
    if "username" in session:
        return redirect(url_for("home") if session.get("role") != "admin"
                        else url_for("admin_dashboard"))
    return render_template("login.html")


@app.route("/login/user", methods=["GET", "POST"])
def login_user():
    """
    Handles both login and registration for regular users.
    A hidden 'action' form field determines which path to take: 'login' or 'register'.
    On success, sets the session and redirects to home.
    """
    if "username" in session:
        return redirect(url_for("home"))

    error = None
    mode  = "login"  # controls which form view the template renders

    if request.method == "POST":
        action   = request.form.get("action", "login")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        mode     = action  # pass back to template so the right tab stays active

        if action == "login":
            user = auth.login(username, password)
            # Extra role check: this endpoint is only for regular users, not admins
            if user and user["role"] == "user":
                session["username"] = user["username"]
                session["role"]     = "user"
                return redirect(url_for("home"))
            error = "Invalid credentials or not a user account."

        elif action == "register":
            ok, err = auth.register(username, password, role="user")
            if ok:
                # Auto-login after successful registration
                session["username"] = username
                session["role"]     = "user"
                return redirect(url_for("home"))
            error = err

    return render_template("login_user.html", error=error, mode=mode)


@app.route("/login/admin", methods=["GET", "POST"])
def login_admin():
    """
    Admin-only login. No registration path — admin accounts are created
    manually or via the admin dashboard.
    """
    if "username" in session and session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user     = auth.login(username, password)
        # Verify the role is specifically 'admin', not just any valid login
        if user and user["role"] == "admin":
            session["username"] = user["username"]
            session["role"]     = "admin"
            return redirect(url_for("admin_dashboard"))
        error = "Invalid admin credentials."

    return render_template("login_admin.html", error=error)


@app.route("/logout")
def logout():
    """
    Clears the session and redirects to the appropriate login page.
    Admins go back to the admin login; users go to the user login.
    """
    role = session.get("role")
    session.clear()
    return redirect(url_for("login_admin") if role == "admin" else url_for("login_user"))


# ── Admin Routes ──────────────────────────────────────────────────────────────

@app.route("/admin")
@login_required
@admin_required
def admin_dashboard():
    """
    Main admin dashboard. Shows all users, aggregated search stats,
    dataset image counts, and a form to create new users.
    """
    users          = auth.all_users()
    total_searches = sum(u["searches"] for u in users)  # aggregate across all users
    ds_info        = _build_ds_info()
    return render_template("admin.html",
        username=session["username"],
        users=users,
        total_searches=total_searches,
        total_images=_total_images,
        datasets=ds_info,
        create_error=None,
        create_ok=None,
    )


@app.route("/admin/user/<username>")
@login_required
@admin_required
def admin_user_detail(username):
    """
    Detail view for a single user: shows their profile info and full search history.
    Redirects back to the dashboard if the user doesn't exist.
    """
    user = auth.get_user(username)
    if not user:
        return redirect(url_for("admin_dashboard"))
    history = auth.get_history(username)
    return render_template("admin_user.html",
        profile={
            "username":    username,
            "role":        user["role"],
            "created":     user["created"],
            "preferences": user.get("preferences", {}),
        },
        history=history,
    )


@app.route("/admin/delete/<username>", methods=["POST"])
@login_required
@admin_required
def admin_delete_user(username):
    """
    Delete a user account. Uses POST (not DELETE) because HTML forms only
    support GET and POST. The auth module protects admin accounts from deletion.
    """
    auth.delete_user(username)
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/create", methods=["POST"])
@login_required
@admin_required
def admin_create_user():
    """
    Create a new user (of any role) from the admin dashboard.
    Re-renders the dashboard with a success or error banner instead of
    redirecting, so the admin can see the result immediately.
    """
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    role     = request.form.get("role", "user")
    ok, err  = auth.register(username, password, role)

    # Re-fetch fresh data so the new user appears in the table immediately
    users          = auth.all_users()
    total_searches = sum(u["searches"] for u in users)
    ds_info        = _build_ds_info()
    return render_template("admin.html",
        username=session["username"],
        users=users,
        total_searches=total_searches,
        total_images=_total_images,
        datasets=ds_info,
        create_error=err if not ok else None,
        create_ok=f"User '{username}' created." if ok else None,
    )


# ── Main App Routes ───────────────────────────────────────────────────────────

@app.route("/")
@login_required
def home():
    """Serve the main search UI. Passes username and role for personalisation."""
    return render_template("index.html",
        username=session.get("username"),
        role=session.get("role"),
    )


@app.route("/api/status")
@login_required
def status():
    """
    Health-check endpoint polled by the frontend while models are loading.
    Returns readiness flag, total image count, and a per-source breakdown.
    """
    sources = {}
    if _ready and _df is not None and "source" in _df.columns:
        sources = _df["source"].value_counts().to_dict()
    return jsonify({
        "ready":        _ready,
        "error":        _error,
        "total_images": _total_images,
        "sources":      sources,
    })


@app.route("/api/search", methods=["POST"])
@login_required
def search():
    """
    Core search endpoint. Accepts a multipart form with:
      - text        : optional text query
      - image       : optional uploaded image file
      - text_weight : float 0–1 controlling the text vs. image blend (default 0.5)

    Pipeline:
      1. Save and validate the uploaded image (if any)
      2. Encode the query into a CLIP embedding vector
      3. Retrieve nearest neighbours from FAISS and re-rank with user preferences
      4. Persist the updated preference tracker and log the search to history
      5. Return results as JSON with inline base64 images and per-item explanations
    """
    if not _ready:
        return jsonify({"error": "Model not ready yet."}), 503

    text         = request.form.get("text", "").strip() or None
    image_file   = request.files.get("image")
    text_weight  = float(request.form.get("text_weight", 0.5))
    image_weight = 1.0 - text_weight  # weights must sum to 1.0

    # ── Handle image upload ───────────────────────────────────────────────────
    image_path = None
    if image_file and image_file.filename and image_file.content_length != 0:
        ext        = os.path.splitext(image_file.filename)[1].lower() or ".jpg"
        fname      = f"{uuid.uuid4().hex}{ext}"  # random filename avoids collisions
        image_path = os.path.join(UPLOAD_FOLDER, fname)
        image_file.save(image_path)
        try:
            # verify() checks the file header — catches corrupt or fake images
            Image.open(image_path).verify()
        except Exception:
            os.remove(image_path)  # discard the bad file
            image_path = None

    # Must have at least one input modality
    if not text and not image_path:
        return jsonify({"error": "Provide a text query or upload an image."}), 400

    # ── Encode the query into a vector ────────────────────────────────────────
    try:
        query_vec, mode = rec.encode_query(
            _model, _processor,
            text=text, image_path=image_path,
            text_weight=text_weight, image_weight=image_weight,
        )
        # mode is one of: "text", "image", or "combined"
    except Exception as e:
        return jsonify({"error": f"Encoding failed: {e}"}), 500

    # ── Retrieve, re-rank, and persist ────────────────────────────────────────
    tracker = get_tracker()
    results = rec.retrieve_and_rerank(query_vec, _index, _df, _embeddings, tracker,
                                      per_source_indexes=_per_source_indexes)
    save_tracker_for_user(tracker)  # persist updated tracker after re-ranking

    # Log this search to the user's history
    auth.append_history(session["username"], {
        "time":       datetime.now().isoformat(),
        "dataset":    "unified",
        "query":      text or "",
        "mode":       mode,
        "top_result": results.iloc[0]["filename"] if len(results) else "",
    })

    # ── Build the response payload ────────────────────────────────────────────
    items = []
    for _, row in results.iterrows():
        # Look up the embedding for this result to generate a natural-language explanation
        match = _df[_df["filename"] == row["filename"]].index
        explanation = ""
        caption     = ""
        if len(match):
            img_emb     = _embeddings[match[0]]
            explanation = rec.explain_result(img_emb, text)
        # BLIP-2 caption (only runs on final TOP_K results for performance)
        if _blip2_ready:
            caption = rec.blip2_caption(row["full_path"])

        items.append({
            "filename":      row["filename"],
            "image_b64":     img_to_b64(row["full_path"]),
            "category":      row.get("category", ""),
            "title":         row.get("title", ""),
            "price":         row.get("price", ""),
            "source":        row.get("source", ""),
            "dataset":       row.get("dataset", ""),
            "final_score":   row["final_score"],
            "sim_query":     row["sim_query"],
            "sim_pref":      row["sim_pref"],
            "redundancy":    row["redundancy"],
            "popularity":    row["popularity"],
            "explanation":   explanation,
            "blip2_caption": caption,
        })

    return jsonify({
        "mode":           mode,
        "results":        items,
        "query_img_b64":  img_to_b64(image_path) if image_path else None,
        "drift_detected": tracker.drift_detected,
        "drift_score":    tracker.drift_score,
    })


@app.route("/api/accept", methods=["POST"])
@login_required
def accept_item():
    """
    Mark an item as liked. Boosts the user's preference vector toward this item.
    Also returns drift detection status.
    Expects JSON body: { "filename": "some_image.jpg" }
    """
    filename = request.json.get("filename")
    if not filename:
        return jsonify({"error": "filename required"}), 400
    if _embeddings is None or _df is None:
        return jsonify({"error": "Model not ready"}), 503

    # Look up the embedding for this item
    match = _df[_df["filename"] == filename].index
    if len(match) == 0:
        return jsonify({"error": "Item not found"}), 404

    emb = _embeddings[match[0]]
    tracker = get_tracker()
    tracker.accept(emb, filename)
    save_tracker_for_user(tracker)

    return jsonify({
        "ok":             True,
        "drift_detected": tracker.drift_detected,
        "drift_score":    tracker.drift_score,
    })


@app.route("/api/reject", methods=["POST"])
@login_required
def reject():
    """
    Mark an item as disliked. Updates the tracker so this item (and visually
    similar ones) are ranked lower in future searches for this user.
    Expects JSON body: { "filename": "some_image.jpg" }
    """
    filename = request.json.get("filename")
    if not filename:
        return jsonify({"error": "filename required"}), 400
    tracker = get_tracker()
    tracker.reject(filename)
    save_tracker_for_user(tracker)
    return jsonify({
        "ok":             True,
        "drift_detected": tracker.drift_detected,
        "drift_score":    tracker.drift_score,
    })


@app.route("/api/reset", methods=["POST"])
@login_required
def reset():
    """
    Wipe the current user's preference tracker entirely.
    Useful when a user wants to start fresh without any personalisation history.
    """
    auth.clear_tracker(session.get("username"), "unified")
    return jsonify({"ok": True})


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_models()               # load ML models before accepting any traffic
    app.run(debug=False, port=5000)  # debug=False is important for production-like behaviour