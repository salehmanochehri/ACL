import hashlib
import secrets
import sqlite3
from datetime import datetime
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
                    created_at TEXT NOT NULL
                )
                """
            )

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

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO users (username, salt, password_hash, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (username, salt.hex(), password_hash, created_at)
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
                "SELECT username, created_at FROM users WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()

        if not row:
            return None

        return {"username": row[0], "created_at": row[1]}
