"""Simple file-based user auth for Hand Gesture System."""
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import USERS_FILE
from werkzeug.security import generate_password_hash, check_password_hash


def _load_users():
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_users(users):
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def register_user(username: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    users = _load_users()
    if username in users:
        return False, "Username already exists."
    if not username or not password:
        return False, "Username and password required."
    users[username] = {"password": generate_password_hash(password)}
    _save_users(users)
    return True, "Registered successfully."


def check_user(username: str, password: str) -> bool:
    """Verify credentials. Returns True if valid."""
    users = _load_users()
    if username not in users:
        return False
    return check_password_hash(users[username]["password"], password)
