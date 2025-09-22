"""
Log file management system with rotation, archival, and cleanup.

This module provides comprehensive log file management including
automatic rotation, compression, archival, and cleanup operations.
"""

import os
import gzip
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading
import time
import json
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID

from .logging_system import get_logger


@dataclass
class LogFileInfo:
    """Information about a log file."""
    path: Path
    size_bytes: int
    created: datetime
    modified: datetime
    compressed: bool
    archived: bool


@dataclass
class LogManagementConfig:
    """Configuration for log management."""
    max_file_size_mb: int = 10
    max_files_per_type: int = 5
    archive_after_days: int = 7
    delete_after_days: int = 30
    compress_archives: bool = True
    auto_cleanup_enabled: bool = True
    cleanup_interval_hours: int = 24


class LogFileManager:
    """Manages log file rotation, archival, and cleanup."""
    
    def __init__(self, log_dir: Path, config: Optional[LogManagementConfig] = None):
        self.log_dir = log_dir
        self.config = config or LogManagementConfig()
        self.console = Console()
        self.logger = get_logger("log_manager")
        
        # Ensure directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.log_dir / "archives"
        self.archive_dir.mkdir(exist_ok=True)
        
        # Background cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        if self.config.auto_cleanup_enabled:
            self.start_auto_cleanup()
    
    def start_auto_cleanup(self):
        """Start automatic cleanup background thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        
        self._stop_cleanup.clear()
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        self.logger.info("Started automatic log cleanup")
    
    def stop_auto_cleanup(self):
        """Stop automatic cleanup background thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5)
            self.logger.info("Stopped automatic log cleanup")
    
    def _cleanup_worker(self):
        """Background worker for automatic cleanup."""
        while not self._stop_cleanup.is_set():
            try:
                self.perform_maintenance()
            except Exception as e:
                self.logger.error(f"Error in automatic cleanup: {e}")
            
            # Wait for next cleanup cycle
            self._stop_cleanup.wait(self.config.cleanup_interval_hours * 3600)
    
    def get_log_files(self) -> List[LogFileInfo]:
        """Get information about all log files."""
        log_files = []
        
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.is_file():
                stat = log_file.stat()
                
                log_files.append(LogFileInfo(
                    path=log_file,
                    size_bytes=stat.st_size,
                    created=datetime.fromtimestamp(stat.st_ctime),
                    modified=datetime.fromtimestamp(stat.st_mtime),
                    compressed=log_file.suffix in ['.gz', '.bz2'],
                    archived=False
                ))
        
        # Check archived files
        for archive_file in self.archive_dir.glob("*.tar.gz"):
            if archive_file.is_file():
                stat = archive_file.stat()
                
                log_files.append(LogFileInfo(
                    path=archive_file,
                    size_bytes=stat.st_size,
                    created=datetime.fromtimestamp(stat.st_ctime),
                    modified=datetime.fromtimestamp(stat.st_mtime),
                    compressed=True,
                    archived=True
                ))
        
        return sorted(log_files, key=lambda f: f.modified, reverse=True)
    
    def rotate_log_file(self, log_file: Path) -> Optional[Path]:
        """Rotate a log file if it exceeds size limit."""
        if not log_file.exists():
            return None
        
        file_size_mb = log_file.stat().st_size / (1024 * 1024)
        
        if file_size_mb < self.config.max_file_size_mb:
            return None
        
        # Generate rotated filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_name = f"{log_file.stem}_{timestamp}.log"
        rotated_path = log_file.parent / rotated_name
        
        try:
            # Move current log to rotated name
            shutil.move(str(log_file), str(rotated_path))
            
            # Create new empty log file
            log_file.touch()
            
            self.logger.info(f"Rotated log file: {log_file.name} -> {rotated_name}")
            return rotated_path
            
        except Exception as e:
            self.logger.error(f"Failed to rotate log file {log_file}: {e}")
            return None
    
    def compress_log_file(self, log_file: Path) -> Optional[Path]:
        """Compress a log file using gzip."""
        if not log_file.exists() or log_file.suffix == '.gz':
            return None
        
        compressed_path = log_file.with_suffix(log_file.suffix + '.gz')
        
        try:
            with open(log_file, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file
            log_file.unlink()
            
            self.logger.info(f"Compressed log file: {log_file.name} -> {compressed_path.name}")
            return compressed_path
            
        except Exception as e:
            self.logger.error(f"Failed to compress log file {log_file}: {e}")
            return None
    
    def archive_old_logs(self, days_old: Optional[int] = None) -> List[Path]:
        """Archive log files older than specified days."""
        days_old = days_old or self.config.archive_after_days
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        archived_files = []
        log_files = self.get_log_files()
        
        # Group files by date for archiving
        files_by_date = {}
        for log_file in log_files:
            if (not log_file.archived and 
                log_file.modified < cutoff_date and
                log_file.path.suffix in ['.log', '.gz']):
                
                date_key = log_file.modified.strftime("%Y%m%d")
                if date_key not in files_by_date:
                    files_by_date[date_key] = []
                files_by_date[date_key].append(log_file.path)
        
        # Create archives for each date
        for date_key, files in files_by_date.items():
            if files:
                archive_path = self._create_archive(date_key, files)
                if archive_path:
                    archived_files.append(archive_path)
                    
                    # Remove original files after successful archiving
                    for file_path in files:
                        try:
                            file_path.unlink()
                            self.logger.debug(f"Removed archived file: {file_path.name}")
                        except Exception as e:
                            self.logger.warning(f"Could not remove {file_path}: {e}")
        
        return archived_files
    
    def _create_archive(self, date_key: str, files: List[Path]) -> Optional[Path]:
        """Create a tar.gz archive for the given files."""
        archive_name = f"escai_logs_{date_key}.tar.gz"
        archive_path = self.archive_dir / archive_name
        
        try:
            with tarfile.open(archive_path, 'w:gz') as tar:
                for file_path in files:
                    if file_path.exists():
                        tar.add(file_path, arcname=file_path.name)
            
            self.logger.info(f"Created archive: {archive_name} with {len(files)} files")
            return archive_path
            
        except Exception as e:
            self.logger.error(f"Failed to create archive {archive_name}: {e}")
            return None
    
    def cleanup_old_files(self, days_old: Optional[int] = None) -> List[Path]:
        """Delete files older than specified days."""
        days_old = days_old or self.config.delete_after_days
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        deleted_files = []
        
        # Clean up old archives
        for archive_file in self.archive_dir.glob("*.tar.gz"):
            if archive_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    archive_file.unlink()
                    deleted_files.append(archive_file)
                    self.logger.info(f"Deleted old archive: {archive_file.name}")
                except Exception as e:
                    self.logger.error(f"Could not delete {archive_file}: {e}")
        
        # Clean up any remaining old log files
        for log_file in self.log_dir.glob("*.log*"):
            if (log_file.is_file() and 
                log_file.stat().st_mtime < cutoff_date.timestamp()):
                try:
                    log_file.unlink()
                    deleted_files.append(log_file)
                    self.logger.info(f"Deleted old log file: {log_file.name}")
                except Exception as e:
                    self.logger.error(f"Could not delete {log_file}: {e}")
        
        return deleted_files
    
    def perform_maintenance(self) -> Dict[str, Any]:
        """Perform comprehensive log maintenance."""
        maintenance_start = time.time()
        results = {
            'rotated_files': [],
            'compressed_files': [],
            'archived_files': [],
            'deleted_files': [],
            'errors': []
        }
        
        try:
            # 1. Rotate large log files
            for log_file in self.log_dir.glob("*.log"):
                if log_file.is_file():
                    rotated = self.rotate_log_file(log_file)
                    if rotated:
                        results['rotated_files'].append(str(rotated))
            
            # 2. Compress old log files
            for log_file in self.log_dir.glob("*.log"):
                if (log_file.is_file() and 
                    log_file.name != "escai_cli.log"):  # Don't compress current log
                    
                    # Check if file is old enough to compress
                    file_age = datetime.now() - datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_age.days >= 1:
                        compressed = self.compress_log_file(log_file)
                        if compressed:
                            results['compressed_files'].append(str(compressed))
            
            # 3. Archive old files
            archived = self.archive_old_logs()
            results['archived_files'] = [str(f) for f in archived]
            
            # 4. Clean up very old files
            deleted = self.cleanup_old_files()
            results['deleted_files'] = [str(f) for f in deleted]
            
        except Exception as e:
            error_msg = f"Maintenance error: {e}"
            results['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        maintenance_time = time.time() - maintenance_start
        results['maintenance_time'] = maintenance_time
        
        self.logger.info(
            f"Log maintenance completed in {maintenance_time:.2f}s: "
            f"{len(results['rotated_files'])} rotated, "
            f"{len(results['compressed_files'])} compressed, "
            f"{len(results['archived_files'])} archived, "
            f"{len(results['deleted_files'])} deleted"
        )
        
        return results
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for log files."""
        log_files = self.get_log_files()
        
        stats = {
            'total_files': len(log_files),
            'total_size_mb': 0,
            'active_files': 0,
            'active_size_mb': 0,
            'archived_files': 0,
            'archived_size_mb': 0,
            'compressed_files': 0,
            'oldest_file': None,
            'newest_file': None,
            'by_type': {}
        }
        
        if not log_files:
            return stats
        
        # Calculate statistics
        for log_file in log_files:
            size_mb = log_file.size_bytes / (1024 * 1024)
            stats['total_size_mb'] += size_mb
            
            if log_file.archived:
                stats['archived_files'] += 1
                stats['archived_size_mb'] += size_mb
            else:
                stats['active_files'] += 1
                stats['active_size_mb'] += size_mb
            
            if log_file.compressed:
                stats['compressed_files'] += 1
            
            # Track by file type
            file_type = log_file.path.suffix
            if file_type not in stats['by_type']:
                stats['by_type'][file_type] = {'count': 0, 'size_mb': 0}
            stats['by_type'][file_type]['count'] += 1
            stats['by_type'][file_type]['size_mb'] += size_mb
        
        # Find oldest and newest files
        stats['oldest_file'] = min(log_files, key=lambda f: f.created).created.isoformat()
        stats['newest_file'] = max(log_files, key=lambda f: f.modified).modified.isoformat()
        
        # Round sizes
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        stats['active_size_mb'] = round(stats['active_size_mb'], 2)
        stats['archived_size_mb'] = round(stats['archived_size_mb'], 2)
        
        return stats
    
    def generate_storage_report(self) -> None:
        """Generate and display storage report."""
        stats = self.get_storage_stats()
        
        # Main statistics table
        main_table = Table(title="Log Storage Statistics")
        main_table.add_column("Metric", style="cyan")
        main_table.add_column("Value", style="yellow")
        
        main_table.add_row("Total Files", str(stats['total_files']))
        main_table.add_row("Total Size", f"{stats['total_size_mb']} MB")
        main_table.add_row("Active Files", str(stats['active_files']))
        main_table.add_row("Active Size", f"{stats['active_size_mb']} MB")
        main_table.add_row("Archived Files", str(stats['archived_files']))
        main_table.add_row("Archived Size", f"{stats['archived_size_mb']} MB")
        main_table.add_row("Compressed Files", str(stats['compressed_files']))
        
        self.console.print(Panel(main_table, title="[bold blue]Storage Overview[/bold blue]"))
        
        # File type breakdown
        if stats['by_type']:
            type_table = Table(title="Files by Type")
            type_table.add_column("Type", style="cyan")
            type_table.add_column("Count", style="yellow")
            type_table.add_column("Size (MB)", style="green")
            
            for file_type, type_stats in stats['by_type'].items():
                type_table.add_row(
                    file_type or "no extension",
                    str(type_stats['count']),
                    f"{type_stats['size_mb']:.2f}"
                )
            
            self.console.print(Panel(type_table, title="[bold green]File Types[/bold green]"))


# Global log manager instance
_log_manager: Optional[LogFileManager] = None


def initialize_log_management(log_dir: Path, config: Optional[LogManagementConfig] = None) -> LogFileManager:
    """Initialize global log management."""
    global _log_manager
    _log_manager = LogFileManager(log_dir, config)
    return _log_manager


def get_log_manager() -> Optional[LogFileManager]:
    """Get the global log manager instance."""
    return _log_manager


def perform_log_maintenance() -> Dict[str, Any]:
    """Perform log maintenance using global manager."""
    if _log_manager is None:
        raise RuntimeError("Log management not initialized")
    return _log_manager.perform_maintenance()


def get_log_storage_stats() -> Dict[str, Any]:
    """Get log storage statistics from global manager."""
    if _log_manager is None:
        raise RuntimeError("Log management not initialized")
    return _log_manager.get_storage_stats()


def generate_log_storage_report() -> None:
    """Generate log storage report from global manager."""
    if _log_manager is None:
        raise RuntimeError("Log management not initialized")
    _log_manager.generate_storage_report()