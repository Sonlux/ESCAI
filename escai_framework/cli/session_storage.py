"""
Enhanced session storage system with SQLite backend for ESCAI CLI.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict

from .utils.console import get_console

console = get_console()


@dataclass
class SessionMetadata:
    """Session metadata structure."""
    session_id: str
    agent_id: str
    framework: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    tags: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class CommandEntry:
    """Command history entry."""
    command_id: str
    session_id: str
    timestamp: datetime
    command: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
    execution_time: Optional[float] = None
    success: bool = True


class SessionStorage:
    """
    Enhanced session storage with SQLite backend.
    
    Provides:
    - Persistent session storage with metadata
    - Command history tracking and replay
    - Session search and filtering
    - Session comparison capabilities
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path.home() / ".escai" / "sessions.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    tags TEXT,
                    description TEXT,
                    config TEXT,
                    statistics TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Command history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS command_history (
                    command_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    command TEXT NOT NULL,
                    arguments TEXT NOT NULL,
                    result TEXT,
                    execution_time REAL,
                    success BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            # Session events table for detailed monitoring data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON sessions(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_framework ON sessions(framework)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_command_history_session_id ON command_history(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_command_history_timestamp ON command_history(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_events_session_id ON session_events(session_id)")
            
            conn.commit()
    
    def create_session(self, agent_id: str, framework: str, description: str = "", 
                      tags: List[str] = None, config: Dict[str, Any] = None) -> str:
        """Create a new monitoring session."""
        session_id = str(uuid.uuid4())
        
        session = SessionMetadata(
            session_id=session_id,
            agent_id=agent_id,
            framework=framework,
            status="active",
            start_time=datetime.now(),
            tags=tags or [],
            description=description
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions 
                (session_id, agent_id, framework, status, start_time, tags, description, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.agent_id,
                session.framework,
                session.status,
                session.start_time.isoformat(),
                json.dumps(session.tags),
                session.description,
                json.dumps(config or {})
            ))
            conn.commit()
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if row:
                session = dict(row)
                session['tags'] = json.loads(session['tags'] or '[]')
                session['config'] = json.loads(session['config'] or '{}')
                session['statistics'] = json.loads(session['statistics'] or '{}')
                return session
        
        return None
    
    def list_sessions(self, agent_id: Optional[str] = None, framework: Optional[str] = None,
                     status: Optional[str] = None, tags: Optional[List[str]] = None,
                     limit: int = 50) -> List[Dict[str, Any]]:
        """List sessions with optional filtering."""
        query = "SELECT * FROM sessions WHERE 1=1"
        params: List[Union[str, int]] = []
        
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        
        if framework:
            query += " AND framework = ?"
            params.append(framework)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if tags:
            # Simple tag filtering - could be enhanced with proper JSON queries
            for tag in tags:
                query += " AND tags LIKE ?"
                params.append(f'%"{tag}"%')
        
        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            sessions = []
            for row in cursor.fetchall():
                session = dict(row)
                session['tags'] = json.loads(session['tags'] or '[]')
                session['config'] = json.loads(session['config'] or '{}')
                session['statistics'] = json.loads(session['statistics'] or '{}')
                sessions.append(session)
            
            return sessions
    
    def update_session(self, session_id: str, **updates):
        """Update session fields."""
        if not updates:
            return
        
        # Handle special fields
        if 'tags' in updates:
            updates['tags'] = json.dumps(updates['tags'])
        if 'config' in updates:
            updates['config'] = json.dumps(updates['config'])
        if 'statistics' in updates:
            updates['statistics'] = json.dumps(updates['statistics'])
        
        # Add updated timestamp
        updates['updated_at'] = datetime.now().isoformat()
        
        # Build dynamic update query
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values()) + [session_id]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                UPDATE sessions SET {set_clause} WHERE session_id = ?
            """, values)
            conn.commit()
    
    def end_session(self, session_id: str):
        """End a monitoring session."""
        self.update_session(
            session_id,
            status="completed",
            end_time=datetime.now().isoformat()
        )
    
    def log_command(self, session_id: str, command: str, arguments: Dict[str, Any],
                   result: Optional[str] = None, execution_time: Optional[float] = None,
                   success: bool = True) -> str:
        """Log a command execution."""
        command_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO command_history 
                (command_id, session_id, timestamp, command, arguments, result, execution_time, success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                command_id,
                session_id,
                datetime.now().isoformat(),
                command,
                json.dumps(arguments),
                result,
                execution_time,
                success
            ))
            conn.commit()
        
        return command_id
    
    def get_command_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get command history for a session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM command_history 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            """, (session_id,))
            
            commands = []
            for row in cursor.fetchall():
                command = dict(row)
                command['arguments'] = json.loads(command['arguments'])
                commands.append(command)
            
            return commands
    
    def search_sessions(self, query: str, search_fields: List[str] = None) -> List[Dict[str, Any]]:
        """Search sessions by text query."""
        if search_fields is None:
            search_fields = ['agent_id', 'framework', 'description']
        
        # Build search query
        where_clauses = []
        params = []
        
        for field in search_fields:
            where_clauses.append(f"{field} LIKE ?")
            params.append(f"%{query}%")
        
        sql_query = f"""
            SELECT * FROM sessions 
            WHERE {' OR '.join(where_clauses)}
            ORDER BY start_time DESC
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql_query, params)
            
            sessions = []
            for row in cursor.fetchall():
                session = dict(row)
                session['tags'] = json.loads(session['tags'] or '[]')
                session['config'] = json.loads(session['config'] or '{}')
                session['statistics'] = json.loads(session['statistics'] or '{}')
                sessions.append(session)
            
            return sessions
    
    def delete_session(self, session_id: str):
        """Delete a session and all related data."""
        with sqlite3.connect(self.db_path) as conn:
            # Delete in order due to foreign key constraints
            conn.execute("DELETE FROM session_events WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM command_history WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get overall session statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_sessions,
                    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_sessions,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
                    COUNT(DISTINCT agent_id) as unique_agents,
                    COUNT(DISTINCT framework) as frameworks_used
                FROM sessions
            """)
            
            row = cursor.fetchone()
            stats = {
                'total_sessions': row[0],
                'active_sessions': row[1], 
                'completed_sessions': row[2],
                'unique_agents': row[3],
                'frameworks_used': row[4]
            }
            
            # Get command statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_commands,
                    COUNT(CASE WHEN success = 1 THEN 1 END) as successful_commands,
                    AVG(execution_time) as avg_execution_time
                FROM command_history
            """)
            
            cmd_row = cursor.fetchone()
            stats.update({
                'total_commands': cmd_row[0],
                'successful_commands': cmd_row[1],
                'avg_execution_time': cmd_row[2]
            })
            
            return stats
    
    def cleanup_old_sessions(self, days: int = 30, status: Optional[str] = None) -> int:
        """Clean up old sessions."""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        cutoff_iso = datetime.fromtimestamp(cutoff_date).isoformat()
        
        query = "DELETE FROM sessions WHERE start_time < ?"
        params = [cutoff_iso]
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            deleted_count = cursor.rowcount
            conn.commit()
            
            return deleted_count