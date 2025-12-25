import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import hashlib
from pathlib import Path


class SessionManager:
    """Manages persistent storage of user sessions"""

    def __init__(self, storage_dir: str = "./.sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.storage_dir / "chats").mkdir(exist_ok=True)
        (self.storage_dir / "configs").mkdir(exist_ok=True)
        (self.storage_dir / "dynamics").mkdir(exist_ok=True)

    def _get_user_id(self, user_identifier: Optional[str] = None) -> str:
        """
        Generate a unique user ID. In production, this would come from auth system.
        For now, use a session-based or cookie-based identifier.
        """
        if user_identifier:
            return hashlib.md5(user_identifier.encode()).hexdigest()[:16]
        return "default_user"

    def _get_user_dir(self, user_id: str) -> Path:
        """Get user-specific directory"""
        user_dir = self.storage_dir / user_id
        user_dir.mkdir(exist_ok=True)
        (user_dir / "chats").mkdir(exist_ok=True)
        (user_dir / "configs").mkdir(exist_ok=True)
        (user_dir / "dynamics").mkdir(exist_ok=True)
        return user_dir

    def save_monitor_state(self, user_id: str, session_id: str, monitor_state: dict):
        """Save monitor state as JSON file"""
        user_dir = self._get_user_dir(user_id)
        monitor_file = user_dir / "chats" / f"{session_id}_monitor.json"
        with open(monitor_file, 'w') as f:
            json.dump(monitor_state, f, indent=2)

    def load_monitor_state(self, user_id: str, session_id: str) -> Optional[dict]:
        """Load monitor state from JSON file"""
        user_dir = self._get_user_dir(user_id)
        monitor_file = user_dir / "chats" / f"{session_id}_monitor.json"
        if monitor_file.exists():
            with open(monitor_file, 'r') as f:
                return json.load(f)
        return None

    def create_session(self, user_id: str, title: str = None) -> str:
        """Create a new session and return session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp}"

        if not title:
            title = f"Design Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        session_data = {
            "session_id": session_id,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
            "chat_history": [],
            "config": {},
            "control_objective": "",
            "design_results": [],
            "custom_dynamics_path": None,
            "survey": None
        }

        self._save_session(user_id, session_id, session_data)
        return session_id

    def _save_session(self, user_id: str, session_id: str, session_data: Dict):
        """Save session data to file"""
        user_dir = self._get_user_dir(user_id)
        session_file = user_dir / "chats" / f"{session_id}.json"

        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

    def load_session(self, user_id: str, session_id: str) -> Optional[Dict]:
        """Load session data from file"""
        user_dir = self._get_user_dir(user_id)
        session_file = user_dir / "chats" / f"{session_id}.json"

        if not session_file.exists():
            return None

        with open(session_file, 'r') as f:
            session_data = json.load(f)

        # Load monitor_state
        monitor_state = self.load_monitor_state(user_id, session_id)
        if monitor_state:
            session_data['monitor_state'] = monitor_state

        return session_data

    def update_session(self, user_id: str, session_id: str, updates: Dict):
        """Update session data"""
        session_data = self.load_session(user_id, session_id)
        if not session_data:
            return False

        # Handle monitor_state separately
        monitor_state = updates.pop('monitor_state', None)
        if monitor_state:
            self.save_monitor_state(user_id, session_id, monitor_state)

        session_data.update(updates)
        session_data["updated_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._save_session(user_id, session_id, session_data)
        return True

    def get_all_sessions(self, user_id: str) -> List[Dict]:
        """Get all sessions for a user, sorted by update time"""
        user_dir = self._get_user_dir(user_id)
        chat_dir = user_dir / "chats"

        sessions = []
        for session_file in chat_dir.glob("session_*.json"):
            if "_monitor" in session_file.name:
                continue
            with open(session_file, 'r') as f:
                session_data = json.load(f)
                sessions.append({
                    "session_id": session_data["session_id"],
                    "title": session_data["title"],
                    "created_at": session_data["created_at"],
                    "updated_at": session_data["updated_at"]
                })

        # Sort by updated_at, most recent first
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        return sessions

    def get_all_surveys(self) -> List[Dict]:
        """Collect survey responses across all users for admin reporting."""
        surveys = []
        for user_dir in self.storage_dir.glob("*"):
            if not user_dir.is_dir():
                continue
            chats_dir = user_dir / "chats"
            if not chats_dir.exists():
                continue
            for session_file in chats_dir.glob("session_*.json"):
                if "_monitor" in session_file.name:
                    continue
                try:
                    with open(session_file, "r", encoding="utf-8") as handle:
                        session_data = json.load(handle)
                except json.JSONDecodeError:
                    continue
                survey = session_data.get("survey")
                if not survey:
                    continue
                surveys.append(
                    {
                        "user_id": user_dir.name,
                        "session_id": session_data.get("session_id"),
                        "title": session_data.get("title"),
                        "created_at": session_data.get("created_at"),
                        "updated_at": session_data.get("updated_at"),
                        "survey": survey,
                    }
                )
        return surveys

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """Delete a session"""
        user_dir = self._get_user_dir(user_id)
        session_file = user_dir / "chats" / f"{session_id}.json"

        if session_file.exists():
            session_file.unlink()
            return True
        return False

    def rename_session(self, user_id: str, session_id: str, new_title: str) -> bool:
        """Rename a session"""
        session_data = self.load_session(user_id, session_id)
        if not session_data:
            return False

        session_data["title"] = new_title
        session_data["updated_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._save_session(user_id, session_id, session_data)
        return True

    def save_custom_dynamics(self, user_id: str, session_id: str,
                             file_content: bytes, filename: str) -> str:
        """Save uploaded custom dynamics file"""
        user_dir = self._get_user_dir(user_id)
        dynamics_dir = user_dir / "dynamics"

        # Create session-specific dynamics file
        dynamics_path = dynamics_dir / f"{session_id}_{filename}"

        with open(dynamics_path, 'wb') as f:
            f.write(file_content)

        return str(dynamics_path)

    def load_custom_dynamics(self, user_id: str, session_id: str) -> Optional[str]:
        """Get path to custom dynamics file for a session"""
        user_dir = self._get_user_dir(user_id)
        dynamics_dir = user_dir / "dynamics"

        # Find dynamics file for this session
        for dynamics_file in dynamics_dir.glob(f"{session_id}_*"):
            return str(dynamics_file)

        return None

    def export_session(self, user_id: str, session_id: str) -> Optional[Dict]:
        """Export complete session data for backup/sharing"""
        session_data = self.load_session(user_id, session_id)
        if not session_data:
            return None

        # Add dynamics file content if exists
        dynamics_path = self.load_custom_dynamics(user_id, session_id)
        if dynamics_path and os.path.exists(dynamics_path):
            with open(dynamics_path, 'rb') as f:
                session_data["dynamics_file_content"] = f.read().hex()
                session_data["dynamics_filename"] = os.path.basename(dynamics_path)

        # Monitor state is already JSON-serializable
        return session_data

    def import_session(self, user_id: str, session_data: Dict) -> str:
        """Import a session from exported data"""
        # Generate new session ID to avoid conflicts
        new_session_id = self.create_session(user_id, session_data.get("title", "Imported Session"))

        # Update with imported data, including monitor_state if present
        import_updates = {
            "chat_history": session_data.get("chat_history", []),
            "config": session_data.get("config", {}),
            "control_objective": session_data.get("control_objective", ""),
            "design_results": session_data.get("design_results", [])
        }
        if "monitor_state" in session_data:
            import_updates["monitor_state"] = session_data["monitor_state"]
        self.update_session(user_id, new_session_id, import_updates)

        # Restore dynamics file if present
        if "dynamics_file_content" in session_data:
            file_content = bytes.fromhex(session_data["dynamics_file_content"])
            filename = session_data.get("dynamics_filename", "dynamics.py")
            self.save_custom_dynamics(user_id, new_session_id, file_content, filename)

        return new_session_id
