"""
Simple file-based auth — users stored in users.json
"""
import os
import json
import hashlib
import uuid
from datetime import datetime

# Absolute path to users.json, always resolved relative to this script's location.
# This ensures the file is found correctly regardless of where the app is launched from.
USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")


def _hash(password: str) -> str:
    """One-way SHA-256 hash of a password. Passwords are never stored in plain text."""
    return hashlib.sha256(password.encode()).hexdigest()


def _load() -> dict:
    """
    Load all user data from users.json.
    If the file doesn't exist yet, create it and seed a default admin account.
    """
    if not os.path.exists(USERS_FILE):
        # First run: bootstrap the database with a single admin user
        data = {
            "admin": {
                "password": _hash("admin123"),  # hashed immediately — never plain text
                "role":     "admin",
                "created":  datetime.now().isoformat(),
                "preferences": {},   # dataset_key → tracker dict (saved UI/filter state)
                "history":   [],     # chronological list of search records
            }
        }
        _save(data)
        return data

    # Normal run: read and return the existing JSON file
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict):
    """Overwrite users.json with the current in-memory user dictionary."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)  # indent=2 keeps the file human-readable


# ── Public API ────────────────────────────────────────────────────────────────

def login(username: str, password: str):
    """
    Verify credentials.
    Returns a lightweight {username, role} dict on success, or None on failure.
    The full user record (including the hashed password) is intentionally NOT returned.
    """
    data = _load()
    user = data.get(username)                          # None if username not found
    if user and user["password"] == _hash(password):  # compare hashes, never plain text
        return {"username": username, "role": user["role"]}
    return None  # authentication failed


def register(username: str, password: str, role: str = "user"):
    """
    Create a new user account.
    Returns (True, None) on success, or (False, error_message) on failure.
    Default role is 'user'; pass role='admin' to create an admin (use carefully).
    """
    # Basic input validation
    if not username or not password:
        return False, "Username and password required."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    data = _load()

    # Prevent duplicate usernames
    if username in data:
        return False, "Username already exists."

    # Add the new user — password is hashed before storage
    data[username] = {
        "password":    _hash(password),
        "role":        role,
        "created":     datetime.now().isoformat(),
        "preferences": {},  # empty until the user saves any settings
        "history":     [],  # empty until the user performs searches
    }
    _save(data)
    return True, None


def get_user(username: str) -> dict | None:
    """Return the full user record for the given username, or None if not found."""
    return _load().get(username)


def all_users() -> list:
    """
    Return a sanitized summary list of all users.
    Passwords are intentionally excluded for security.
    """
    data = _load()
    return [
        {
            "username": u,
            "role":     d["role"],
            "created":  d["created"],
            "searches": len(d.get("history", [])),  # total searches performed so far
        }
        for u, d in data.items()
    ]


def delete_user(username: str) -> bool:
    """
    Delete a user by username.
    Returns True on success, False if the user doesn't exist or is an admin.
    Admin accounts are protected from deletion to prevent accidental lockout.
    """
    data = _load()
    # Guard: can't delete a non-existent user or any admin account
    if username not in data or data[username]["role"] == "admin":
        return False
    del data[username]
    _save(data)
    return True


# ── Preference persistence ────────────────────────────────────────────────────

def load_tracker(username: str, dataset_key: str) -> dict | None:
    """
    Retrieve a user's saved tracker (filter/UI state) for a specific dataset.
    Returns None if no tracker has been saved yet for that dataset.
    """
    data = _load()
    user = data.get(username, {})
    return user.get("preferences", {}).get(dataset_key)  # nested safe-get


def save_tracker(username: str, dataset_key: str, tracker_dict: dict):
    """
    Persist (insert or overwrite) a tracker dict for the given user and dataset.
    Silently does nothing if the username doesn't exist.
    """
    data = _load()
    if username not in data:
        return
    # setdefault ensures 'preferences' key exists before writing into it
    data[username].setdefault("preferences", {})[dataset_key] = tracker_dict
    _save(data)


def clear_tracker(username: str, dataset_key: str | None = None):
    """
    Clear saved tracker(s) for a user.
    - If dataset_key is provided: removes only that specific tracker.
    - If dataset_key is None: wipes ALL trackers for the user.
    """
    data = _load()
    if username not in data:
        return
    if dataset_key:
        data[username].get("preferences", {}).pop(dataset_key, None)  # pop silently ignores missing keys
    else:
        data[username]["preferences"] = {}  # reset entirely
    _save(data)


# ── Search history ────────────────────────────────────────────────────────────

def append_history(username: str, record: dict):
    """
    Append a search record to the user's history.
    Enforces a rolling cap of 100 entries — oldest records are dropped when exceeded.
    Silently does nothing if the username doesn't exist.
    """
    data = _load()
    if username not in data:
        return
    data[username].setdefault("history", []).append(record)

    # Trim to the last 100 entries to prevent unbounded file growth
    data[username]["history"] = data[username]["history"][-100:]
    _save(data)


def get_history(username: str) -> list:
    """
    Return the full search history list for a user.
    Returns an empty list if the user doesn't exist or has no history.
    """
    data = _load()
    return data.get(username, {}).get("history", [])