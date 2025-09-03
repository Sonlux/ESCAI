"""
Configuration versioning module for the ESCAI framework.

This module provides version control capabilities for configuration management,
including version history, rollback, and change tracking.
"""

import json
import hashlib
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class ConfigVersioning:
    """Configuration versioning manager."""
    
    def __init__(self, versions_dir: str = "config/versions"):
        """
        Initialize configuration versioning.
        
        Args:
            versions_dir: Directory to store configuration versions
        """
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        # Version metadata file
        self.metadata_file = self.versions_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load version metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load version metadata: {e}")
        
        return {
            'versions': [],
            'current_version': None,
            'created_at': datetime.now().isoformat()
        }
    
    def _save_metadata(self) -> None:
        """Save version metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save version metadata: {e}")
    
    def _generate_version_id(self, config_data: Dict[str, Any]) -> str:
        """Generate unique version ID based on configuration content."""
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def save_version(self, config_data: Dict[str, Any], 
                    description: str = "", tags: Optional[List[str]] = None) -> str:
        """
        Save a new configuration version.
        
        Args:
            config_data: Configuration dictionary to version
            description: Description of the changes
            tags: Optional tags for the version
            
        Returns:
            Version ID of the saved version
        """
        try:
            version_id = self._generate_version_id(config_data)
            timestamp = datetime.now().isoformat()
            
            # Check if version already exists
            if any(v['id'] == version_id for v in self.metadata['versions']):
                logger.debug(f"Version {version_id} already exists")
                return version_id
            
            # Save version data
            version_file = self.versions_dir / f"{version_id}.json"
            version_data = {
                'id': version_id,
                'timestamp': timestamp,
                'description': description,
                'tags': tags or [],
                'config': config_data,
                'size': len(json.dumps(config_data))
            }
            
            with open(version_file, 'w') as f:
                json.dump(version_data, f, indent=2)
            
            # Update metadata
            self.metadata['versions'].append({
                'id': version_id,
                'timestamp': timestamp,
                'description': description,
                'tags': tags or [],
                'size': version_data['size']
            })
            
            # Keep only last 50 versions
            if len(self.metadata['versions']) > 50:
                old_versions = self.metadata['versions'][:-50]
                self.metadata['versions'] = self.metadata['versions'][-50:]
                
                # Clean up old version files
                for old_version in old_versions:
                    old_file = self.versions_dir / f"{old_version['id']}.json"
                    if old_file.exists():
                        old_file.unlink()
            
            self.metadata['current_version'] = version_id
            self._save_metadata()
            
            logger.info(f"Saved configuration version: {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to save configuration version: {e}")
            raise
    
    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration data for a specific version.
        
        Args:
            version_id: Version ID to retrieve
            
        Returns:
            Version data dictionary or None if not found
        """
        try:
            version_file = self.versions_dir / f"{version_id}.json"
            if not version_file.exists():
                return None
            
            with open(version_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to get version {version_id}: {e}")
            return None
    
    def get_version_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get version history metadata.
        
        Args:
            limit: Maximum number of versions to return
            
        Returns:
            List of version metadata dictionaries
        """
        versions = self.metadata['versions']
        # Sort by timestamp (newest first)
        versions.sort(key=lambda x: x['timestamp'], reverse=True)
        return versions[:limit]
    
    def get_current_version(self) -> Optional[str]:
        """Get current version ID."""
        return self.metadata.get('current_version')
    
    def compare_versions(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """
        Compare two configuration versions.
        
        Args:
            version1_id: First version ID
            version2_id: Second version ID
            
        Returns:
            Comparison result dictionary
        """
        try:
            version1 = self.get_version(version1_id)
            version2 = self.get_version(version2_id)
            
            if not version1 or not version2:
                raise ValueError("One or both versions not found")
            
            config1 = version1['config']
            config2 = version2['config']
            
            differences = self._find_differences(config1, config2)
            
            return {
                'version1': {
                    'id': version1_id,
                    'timestamp': version1['timestamp'],
                    'description': version1['description']
                },
                'version2': {
                    'id': version2_id,
                    'timestamp': version2['timestamp'],
                    'description': version2['description']
                },
                'differences': differences,
                'total_changes': len(differences)
            }
            
        except Exception as e:
            logger.error(f"Failed to compare versions: {e}")
            raise
    
    def _find_differences(self, config1: Dict[str, Any], 
                         config2: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
        """Find differences between two configuration dictionaries."""
        differences = []
        
        # Check all keys in config1
        for key, value1 in config1.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in config2:
                differences.append({
                    'key': full_key,
                    'type': 'removed',
                    'old_value': value1,
                    'new_value': None
                })
            elif isinstance(value1, dict) and isinstance(config2[key], dict):
                # Recursively check nested dictionaries
                nested_diffs = self._find_differences(value1, config2[key], full_key)
                differences.extend(nested_diffs)
            elif value1 != config2[key]:
                differences.append({
                    'key': full_key,
                    'type': 'modified',
                    'old_value': value1,
                    'new_value': config2[key]
                })
        
        # Check for new keys in config2
        for key, value2 in config2.items():
            if key not in config1:
                full_key = f"{prefix}.{key}" if prefix else key
                differences.append({
                    'key': full_key,
                    'type': 'added',
                    'old_value': None,
                    'new_value': value2
                })
        
        return differences
    
    def delete_version(self, version_id: str) -> bool:
        """
        Delete a configuration version.
        
        Args:
            version_id: Version ID to delete
            
        Returns:
            True if version was deleted successfully
        """
        try:
            # Remove from metadata
            self.metadata['versions'] = [
                v for v in self.metadata['versions'] if v['id'] != version_id
            ]
            
            # Update current version if it was deleted
            if self.metadata.get('current_version') == version_id:
                if self.metadata['versions']:
                    self.metadata['current_version'] = self.metadata['versions'][-1]['id']
                else:
                    self.metadata['current_version'] = None
            
            # Delete version file
            version_file = self.versions_dir / f"{version_id}.json"
            if version_file.exists():
                version_file.unlink()
            
            self._save_metadata()
            
            logger.info(f"Deleted configuration version: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {e}")
            return False
    
    def tag_version(self, version_id: str, tags: List[str]) -> bool:
        """
        Add tags to a configuration version.
        
        Args:
            version_id: Version ID to tag
            tags: List of tags to add
            
        Returns:
            True if tags were added successfully
        """
        try:
            # Update metadata
            for version in self.metadata['versions']:
                if version['id'] == version_id:
                    existing_tags = set(version.get('tags', []))
                    existing_tags.update(tags)
                    version['tags'] = list(existing_tags)
                    break
            else:
                return False
            
            # Update version file
            version_data = self.get_version(version_id)
            if version_data:
                existing_tags = set(version_data.get('tags', []))
                existing_tags.update(tags)
                version_data['tags'] = list(existing_tags)
                
                version_file = self.versions_dir / f"{version_id}.json"
                with open(version_file, 'w') as f:
                    json.dump(version_data, f, indent=2)
            
            self._save_metadata()
            
            logger.info(f"Added tags to version {version_id}: {tags}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to tag version {version_id}: {e}")
            return False
    
    def find_versions_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Find versions by tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of version metadata dictionaries
        """
        return [
            version for version in self.metadata['versions']
            if tag in version.get('tags', [])
        ]
    
    def cleanup_old_versions(self, keep_count: int = 20) -> int:
        """
        Clean up old configuration versions.
        
        Args:
            keep_count: Number of recent versions to keep
            
        Returns:
            Number of versions deleted
        """
        try:
            versions = self.metadata['versions']
            versions.sort(key=lambda x: x['timestamp'], reverse=True)
            
            if len(versions) <= keep_count:
                return 0
            
            versions_to_delete = versions[keep_count:]
            deleted_count = 0
            
            for version in versions_to_delete:
                if self.delete_version(version['id']):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old configuration versions")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old versions: {e}")
            return 0
    
    def export_version_history(self, export_path: str) -> None:
        """
        Export version history to file.
        
        Args:
            export_path: Path to export version history
        """
        try:
            export_data = {
                'metadata': self.metadata,
                'export_timestamp': datetime.now().isoformat(),
                'total_versions': len(self.metadata['versions'])
            }
            
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported version history to: {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export version history: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get versioning statistics.
        
        Returns:
            Statistics dictionary
        """
        versions = self.metadata['versions']
        
        if not versions:
            return {
                'total_versions': 0,
                'oldest_version': None,
                'newest_version': None,
                'total_size': 0,
                'average_size': 0
            }
        
        total_size = sum(v.get('size', 0) for v in versions)
        
        return {
            'total_versions': len(versions),
            'oldest_version': min(versions, key=lambda x: x['timestamp']),
            'newest_version': max(versions, key=lambda x: x['timestamp']),
            'total_size': total_size,
            'average_size': total_size // len(versions) if versions else 0,
            'versions_by_month': self._get_versions_by_month(versions)
        }
    
    def _get_versions_by_month(self, versions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get version count by month."""
        by_month: Dict[str, int] = {}
        
        for version in versions:
            try:
                timestamp = datetime.fromisoformat(version['timestamp'])
                month_key = timestamp.strftime('%Y-%m')
                by_month[month_key] = by_month.get(month_key, 0) + 1
            except Exception:
                logger.warning(f"Could not parse timestamp for version {version.get('id')}", exc_info=True)
                continue
        
        return by_month