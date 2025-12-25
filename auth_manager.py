import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional


class AuthManager:
    """SQLite-backed authentication manager for Streamlit."""

    def __init__(self, storage_dir: str = "./.sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.db_path = self.storage_dir / "users.db"
        self._initialize_db()

    def _initialize_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
            if "is_admin" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")

    def _hash_password(self, password: str, salt: bytes) -> str:
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            200_000
        )
        return digest.hex()

    def register_user(self, username: str, password: str) -> bool:
        username = username.strip().lower()
        if not username or not password:
            return False

        salt = secrets.token_bytes(16)
        password_hash = self._hash_password(password, salt)
        created_at = datetime.utcnow().isoformat() + "Z"
        with sqlite3.connect(self.db_path) as conn:
            total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        is_admin = 1 if total_users == 0 else 0

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO users (username, salt, password_hash, created_at, is_admin)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (username, salt.hex(), password_hash, created_at, is_admin)
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def verify_user(self, username: str, password: str) -> bool:
        username = username.strip().lower()
        if not username or not password:
            return False

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT salt, password_hash FROM users WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()

        if not row:
            return False

        salt_hex, stored_hash = row
        salt = bytes.fromhex(salt_hex)
        password_hash = self._hash_password(password, salt)
        return secrets.compare_digest(password_hash, stored_hash)

    def get_user_profile(self, username: str) -> Optional[Dict]:
        username = username.strip().lower()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT username, created_at, is_admin FROM users WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()

        if not row:
            return None

        return {"username": row[0], "created_at": row[1], "is_admin": bool(row[2])}

    def is_admin(self, username: str) -> bool:
        profile = self.get_user_profile(username)
        if not profile:
            return False
        return bool(profile.get("is_admin"))
    
    def get_all_users(self) -> Dict[str, Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT username, created_at, is_admin FROM users ORDER BY created_at DESC"
            )
            rows = cursor.fetchall()

        return {
            row[0]: {
                "username": row[0],
                "created_at": row[1],
                "is_admin": bool(row[2]),
            }
            for row in rows
        }
    
    def set_admin_status(self, username: str, is_admin: bool = True) -> bool:
        username = username.strip().lower()
        if not username:
            return False
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE users SET is_admin = ? WHERE username = ?",
                (1 if is_admin else 0, username),
            )
            return cursor.rowcount > 0
        
    def create_session_token(self, username: str, ttl_hours: int = 24) -> str:
        username = username.strip().lower()
        if not username:
            raise ValueError("Username is required to create a session token.")
        token = secrets.token_urlsafe(32)
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(hours=ttl_hours)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sessions (token, username, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    token,
                    username,
                    created_at.isoformat() + "Z",
                    expires_at.isoformat() + "Z",
                ),
            )
        return token

    def verify_session_token(self, token: str) -> Optional[str]:
        if not token:
            return None
        now = datetime.utcnow()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT username, expires_at FROM sessions WHERE token = ?",
                (token,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            username, expires_at = row
            try:
                expiry = datetime.fromisoformat(expires_at.replace("Z", ""))
            except ValueError:
                conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
                return None
            if expiry < now:
                conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
                return None
            return username

    def revoke_session_token(self, token: str) -> None:
        if not token:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))