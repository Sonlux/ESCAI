#!/usr/bin/env python3
"""
CI/CD Workflow Cleanup Tool

This module provides functionality to detect, validate, and manage GitHub Actions
workflow files for the ESCAI framework project.
"""

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WorkflowFileDetector:
    """Handles detection and validation of GitHub Actions workflow files."""
    
    def __init__(self, workflows_dir: str = ".github/workflows"):
        """
        Initialize the workflow file detector.
        
        Args:
            workflows_dir: Path to the GitHub workflows directory
        """
        self.workflows_dir = Path(workflows_dir)
        self.supported_extensions = ['.yml', '.yaml']
    
    def scan_workflow_directory(self) -> List[Path]:
        """
        Scan the .github/workflows directory for .yml and .yaml files.
        
        Returns:
            List of Path objects for workflow files found
            
        Raises:
            FileNotFoundError: If the workflows directory doesn't exist
            PermissionError: If the directory is not accessible
        """
        logger.info(f"Scanning workflow directory: {self.workflows_dir}")
        
        # Validate directory exists and is accessible
        if not self._validate_directory_access():
            return []
        
        workflow_files = []
        
        try:
            # Scan for workflow files with supported extensions
            for file_path in self.workflows_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                    workflow_files.append(file_path)
                    logger.info(f"Found workflow file: {file_path}")
            
            logger.info(f"Total workflow files found: {len(workflow_files)}")
            return workflow_files
            
        except PermissionError as e:
            logger.error(f"Permission denied accessing workflows directory: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error scanning workflows directory: {e}")
            raise
    
    def validate_workflow_files_exist(self, file_paths: Optional[List[Path]] = None) -> Tuple[List[Path], List[Path]]:
        """
        Validate that workflow files exist before processing.
        
        Args:
            file_paths: Optional list of specific files to validate. 
                       If None, scans the directory for all workflow files.
        
        Returns:
            Tuple of (existing_files, missing_files)
        """
        if file_paths is None:
            try:
                file_paths = self.scan_workflow_directory()
            except (FileNotFoundError, PermissionError):
                return [], []
        
        existing_files = []
        missing_files = []
        
        for file_path in file_paths:
            if file_path.exists() and file_path.is_file():
                existing_files.append(file_path)
                logger.debug(f"Validated existing file: {file_path}")
            else:
                missing_files.append(file_path)
                logger.warning(f"Missing workflow file: {file_path}")
        
        logger.info(f"Validation complete - Existing: {len(existing_files)}, Missing: {len(missing_files)}")
        return existing_files, missing_files
    
    def _validate_directory_access(self) -> bool:
        """
        Validate that the workflows directory exists and is accessible.
        
        Returns:
            True if directory is accessible, False otherwise
        """
        if not self.workflows_dir.exists():
            logger.error(f"Workflows directory does not exist: {self.workflows_dir}")
            return False
        
        if not self.workflows_dir.is_dir():
            logger.error(f"Workflows path is not a directory: {self.workflows_dir}")
            return False
        
        # Test read access
        try:
            list(self.workflows_dir.iterdir())
            logger.debug(f"Directory access validated: {self.workflows_dir}")
            return True
        except PermissionError:
            logger.error(f"Permission denied accessing directory: {self.workflows_dir}")
            return False
        except Exception as e:
            logger.error(f"Error accessing directory {self.workflows_dir}: {e}")
            return False


class WorkflowArchiver:
    """Handles archiving of GitHub Actions workflow files."""
    
    def __init__(self, archive_dir: str = "archived-workflows"):
        """
        Initialize the workflow archiver.
        
        Args:
            archive_dir: Path to the archive directory
        """
        self.archive_dir = Path(archive_dir)
    
    def archive_workflow_file(self, workflow_file: Path, reason: str = "Workflow cleanup") -> Optional[Path]:
        """
        Move a workflow file from .github/workflows to archived-workflows with .md extension.
        
        Args:
            workflow_file: Path to the workflow file to archive
            reason: Reason for archiving the workflow
            
        Returns:
            Path to the archived file if successful, None otherwise
        """
        if not workflow_file.exists():
            logger.error(f"Workflow file does not exist: {workflow_file}")
            return None
        
        if not workflow_file.is_file():
            logger.error(f"Path is not a file: {workflow_file}")
            return None
        
        # Ensure archive directory exists
        if not self._ensure_archive_directory():
            return None
        
        # Generate archive file path with .md extension
        archive_filename = workflow_file.stem + ".md"
        archive_path = self.archive_dir / archive_filename
        
        # Handle filename conflicts
        archive_path = self._resolve_filename_conflict(archive_path)
        
        try:
            # Read original content with UTF-8 encoding
            original_content = workflow_file.read_text(encoding='utf-8')
            
            # Create archived content with metadata
            archived_content = self._create_archived_content(
                original_content, 
                workflow_file, 
                reason
            )
            
            # Write to archive location with UTF-8 encoding
            archive_path.write_text(archived_content, encoding='utf-8')
            logger.info(f"Archived workflow content to: {archive_path}")
            
            # Remove original file
            workflow_file.unlink()
            logger.info(f"Removed original workflow file: {workflow_file}")
            
            return archive_path
            
        except Exception as e:
            logger.error(f"Error archiving workflow file {workflow_file}: {e}")
            return None
    
    def archive_multiple_workflows(self, workflow_files: List[Path], reason: str = "Workflow cleanup") -> List[Tuple[Path, Optional[Path]]]:
        """
        Archive multiple workflow files.
        
        Args:
            workflow_files: List of workflow files to archive
            reason: Reason for archiving the workflows
            
        Returns:
            List of tuples (original_path, archived_path) where archived_path is None if archiving failed
        """
        results = []
        
        for workflow_file in workflow_files:
            logger.info(f"Archiving workflow file: {workflow_file}")
            archived_path = self.archive_workflow_file(workflow_file, reason)
            results.append((workflow_file, archived_path))
        
        successful_archives = sum(1 for _, archived_path in results if archived_path is not None)
        logger.info(f"Successfully archived {successful_archives} out of {len(workflow_files)} workflow files")
        
        return results
    
    def _ensure_archive_directory(self) -> bool:
        """
        Ensure the archive directory exists.
        
        Returns:
            True if directory exists or was created successfully, False otherwise
        """
        try:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Archive directory ensured: {self.archive_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to create archive directory {self.archive_dir}: {e}")
            return False
    
    def _resolve_filename_conflict(self, archive_path: Path) -> Path:
        """
        Resolve filename conflicts by appending a counter.
        
        Args:
            archive_path: Proposed archive file path
            
        Returns:
            Path that doesn't conflict with existing files
        """
        if not archive_path.exists():
            return archive_path
        
        counter = 1
        base_name = archive_path.stem
        suffix = archive_path.suffix
        
        while True:
            new_name = f"{base_name}_{counter}{suffix}"
            new_path = archive_path.parent / new_name
            if not new_path.exists():
                logger.info(f"Resolved filename conflict: {archive_path} -> {new_path}")
                return new_path
            counter += 1
    
    def _create_archived_content(self, original_content: str, original_path: Path, reason: str) -> str:
        """
        Create archived content with metadata header.
        
        Args:
            original_content: Original workflow file content
            original_path: Path to the original workflow file
            reason: Reason for archiving
            
        Returns:
            Content with metadata header
        """
        timestamp = datetime.now().isoformat()
        
        metadata_header = f"""# Archived Workflow File

**Original File:** `{original_path}`  
**Archived Date:** {timestamp}  
**Reason:** {reason}  
**Original Content Preserved:** Yes  

---

## Original Workflow Content

```yaml
{original_content}
```
"""
        
        return metadata_header


class GitIgnoreManager:
    """Handles .gitignore file operations for workflow cleanup."""
    
    def __init__(self, gitignore_path: str = ".gitignore"):
        """
        Initialize the GitIgnore manager.
        
        Args:
            gitignore_path: Path to the .gitignore file
        """
        self.gitignore_path = Path(gitignore_path)
        self.archive_pattern = "archived-workflows/"
    
    def read_gitignore(self) -> Optional[str]:
        """
        Read the existing .gitignore file content.
        
        Returns:
            Content of .gitignore file as string, or None if file doesn't exist or can't be read
        """
        try:
            if self.gitignore_path.exists():
                content = self.gitignore_path.read_text(encoding='utf-8')
                logger.debug(f"Successfully read .gitignore file: {self.gitignore_path}")
                return content
            else:
                logger.info(f".gitignore file does not exist: {self.gitignore_path}")
                return None
        except Exception as e:
            logger.error(f"Error reading .gitignore file {self.gitignore_path}: {e}")
            return None
    
    def check_pattern_exists(self, content: str) -> bool:
        """
        Check if the archived-workflows pattern already exists in .gitignore content.
        
        Args:
            content: Content of the .gitignore file
            
        Returns:
            True if pattern exists, False otherwise
        """
        if not content:
            return False
        
        lines = content.splitlines()
        
        # Check for exact pattern match or variations
        patterns_to_check = [
            "archived-workflows/",
            "archived-workflows",
            "/archived-workflows/",
            "/archived-workflows"
        ]
        
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            
            if line in patterns_to_check:
                logger.info(f"Found existing pattern in .gitignore: {line}")
                return True
        
        logger.debug("Archive pattern not found in .gitignore")
        return False
    
    def add_ignore_pattern(self, preserve_existing: bool = True) -> bool:
        """
        Add the archived-workflows pattern to .gitignore if not present.
        
        Args:
            preserve_existing: Whether to preserve existing .gitignore content
            
        Returns:
            True if pattern was added or already exists, False on error
        """
        try:
            # Read existing content
            existing_content = self.read_gitignore() if preserve_existing else None
            
            # Check if pattern already exists
            if existing_content and self.check_pattern_exists(existing_content):
                logger.info("Archive pattern already exists in .gitignore, no changes needed")
                return True
            
            # Prepare new content
            if existing_content:
                # Add pattern to existing content
                new_content = self._add_pattern_to_existing_content(existing_content)
            else:
                # Create new .gitignore with just the pattern
                new_content = self._create_new_gitignore_content()
            
            # Write updated content
            return self._write_gitignore(new_content)
            
        except Exception as e:
            logger.error(f"Error adding ignore pattern to .gitignore: {e}")
            return False
    
    def _add_pattern_to_existing_content(self, existing_content: str) -> str:
        """
        Add the archive pattern to existing .gitignore content.
        
        Args:
            existing_content: Current content of .gitignore
            
        Returns:
            Updated content with archive pattern added
        """
        # Ensure content ends with newline
        if not existing_content.endswith('\n'):
            existing_content += '\n'
        
        # Add a section for archived workflows
        archive_section = f"""
# Archived GitHub Actions workflows
{self.archive_pattern}
"""
        
        new_content = existing_content + archive_section
        logger.debug("Added archive pattern to existing .gitignore content")
        return new_content
    
    def _create_new_gitignore_content(self) -> str:
        """
        Create new .gitignore content with just the archive pattern.
        
        Returns:
            New .gitignore content
        """
        content = f"""# Archived GitHub Actions workflows
{self.archive_pattern}
"""
        logger.debug("Created new .gitignore content with archive pattern")
        return content
    
    def _write_gitignore(self, content: str) -> bool:
        """
        Write content to .gitignore file.
        
        Args:
            content: Content to write to .gitignore
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.gitignore_path.write_text(content, encoding='utf-8')
            logger.info(f"Successfully updated .gitignore file: {self.gitignore_path}")
            return True
        except Exception as e:
            logger.error(f"Error writing to .gitignore file {self.gitignore_path}: {e}")
            return False
    
    def validate_gitignore_update(self) -> bool:
        """
        Validate that the .gitignore file contains the archive pattern.
        
        Returns:
            True if pattern exists in .gitignore, False otherwise
        """
        content = self.read_gitignore()
        if content is None:
            logger.error("Cannot validate .gitignore update - file not readable")
            return False
        
        pattern_exists = self.check_pattern_exists(content)
        if pattern_exists:
            logger.info("Validation successful: archive pattern found in .gitignore")
        else:
            logger.error("Validation failed: archive pattern not found in .gitignore")
        
        return pattern_exists


class WorkflowVerifier:
    """Handles verification and rollback operations for workflow cleanup."""
    
    def __init__(self, workflows_dir: str = ".github/workflows", archive_dir: str = "archived-workflows"):
        """
        Initialize the workflow verifier.
        
        Args:
            workflows_dir: Path to the GitHub workflows directory
            archive_dir: Path to the archive directory
        """
        self.workflows_dir = Path(workflows_dir)
        self.archive_dir = Path(archive_dir)
        self.supported_extensions = ['.yml', '.yaml']
        self.logger = logging.getLogger(__name__)
    
    def verify_no_active_workflows(self) -> Tuple[bool, List[Path]]:
        """
        Verify that no active workflow files remain in .github/workflows directory.
        
        Returns:
            Tuple of (is_clean, remaining_files) where is_clean is True if no workflow files remain
        """
        self.logger.info("Verifying no active workflow files remain")
        
        try:
            if not self.workflows_dir.exists():
                self.logger.info("Workflows directory does not exist - verification passed")
                return True, []
            
            remaining_files = []
            for file_path in self.workflows_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                    remaining_files.append(file_path)
                    self.logger.warning(f"Found remaining workflow file: {file_path}")
            
            is_clean = len(remaining_files) == 0
            if is_clean:
                self.logger.info("Verification passed - no active workflow files found")
            else:
                self.logger.error(f"Verification failed - found {len(remaining_files)} active workflow files")
            
            return is_clean, remaining_files
            
        except Exception as e:
            self.logger.error(f"Error during workflow verification: {e}")
            return False, []
    
    def list_archived_workflows(self) -> List[Dict[str, str]]:
        """
        List all archived workflows with their metadata.
        
        Returns:
            List of dictionaries containing metadata for each archived workflow
        """
        self.logger.info("Listing archived workflows with metadata")
        
        archived_workflows = []
        
        try:
            if not self.archive_dir.exists():
                self.logger.info("Archive directory does not exist - no archived workflows found")
                return archived_workflows
            
            for archive_file in self.archive_dir.glob("*.md"):
                metadata = self._extract_metadata_from_archive(archive_file)
                if metadata:
                    archived_workflows.append(metadata)
                    self.logger.debug(f"Found archived workflow: {metadata['filename']}")
            
            self.logger.info(f"Found {len(archived_workflows)} archived workflows")
            return archived_workflows
            
        except Exception as e:
            self.logger.error(f"Error listing archived workflows: {e}")
            return []
    
    def rollback_workflow(self, archive_filename: str) -> Optional[Path]:
        """
        Restore a single archived workflow back to .github/workflows directory.
        
        Args:
            archive_filename: Name of the archived file (e.g., "ci-cd.md")
            
        Returns:
            Path to the restored workflow file if successful, None otherwise
        """
        self.logger.info(f"Rolling back archived workflow: {archive_filename}")
        
        archive_path = self.archive_dir / archive_filename
        
        if not archive_path.exists():
            self.logger.error(f"Archive file does not exist: {archive_path}")
            return None
        
        try:
            # Extract metadata and original content
            metadata = self._extract_metadata_from_archive(archive_path)
            if not metadata:
                self.logger.error(f"Could not extract metadata from archive: {archive_path}")
                return None
            
            original_content = self._extract_original_content_from_archive(archive_path)
            if not original_content:
                self.logger.error(f"Could not extract original content from archive: {archive_path}")
                return None
            
            # Determine original filename and extension
            original_filename = metadata.get('original_filename', archive_filename.replace('.md', '.yml'))
            
            # Ensure workflows directory exists
            self.workflows_dir.mkdir(parents=True, exist_ok=True)
            
            # Create restored workflow path
            restored_path = self.workflows_dir / original_filename
            
            # Handle filename conflicts
            restored_path = self._resolve_restore_filename_conflict(restored_path)
            
            # Write original content back
            restored_path.write_text(original_content, encoding='utf-8')
            self.logger.info(f"Restored workflow content to: {restored_path}")
            
            # Optionally remove the archive file (commented out for safety)
            # archive_path.unlink()
            # self.logger.info(f"Removed archive file: {archive_path}")
            
            return restored_path
            
        except Exception as e:
            self.logger.error(f"Error rolling back workflow {archive_filename}: {e}")
            return None
    
    def rollback_all_workflows(self) -> List[Tuple[str, Optional[Path]]]:
        """
        Restore all archived workflows back to .github/workflows directory.
        
        Returns:
            List of tuples (archive_filename, restored_path) where restored_path is None if rollback failed
        """
        self.logger.info("Rolling back all archived workflows")
        
        archived_workflows = self.list_archived_workflows()
        results = []
        
        for workflow_metadata in archived_workflows:
            archive_filename = workflow_metadata['filename']
            restored_path = self.rollback_workflow(archive_filename)
            results.append((archive_filename, restored_path))
        
        successful_rollbacks = sum(1 for _, restored_path in results if restored_path is not None)
        self.logger.info(f"Successfully rolled back {successful_rollbacks} out of {len(archived_workflows)} workflows")
        
        return results
    
    def _extract_metadata_from_archive(self, archive_path: Path) -> Optional[Dict[str, str]]:
        """
        Extract metadata from an archived workflow file.
        
        Args:
            archive_path: Path to the archived workflow file
            
        Returns:
            Dictionary containing metadata, or None if extraction failed
        """
        try:
            content = archive_path.read_text(encoding='utf-8')
            
            # Parse metadata using regex patterns
            metadata = {
                'filename': archive_path.name,
                'archive_path': str(archive_path)
            }
            
            # Extract original file path
            original_file_match = re.search(r'\*\*Original File:\*\*\s*`([^`]+)`', content)
            if original_file_match:
                original_path = original_file_match.group(1)
                metadata['original_path'] = original_path
                metadata['original_filename'] = Path(original_path).name
            
            # Extract archived date
            archived_date_match = re.search(r'\*\*Archived Date:\*\*\s*([^\n]+)', content)
            if archived_date_match:
                metadata['archived_date'] = archived_date_match.group(1).strip()
            
            # Extract reason
            reason_match = re.search(r'\*\*Reason:\*\*\s*([^\n]+)', content)
            if reason_match:
                metadata['reason'] = reason_match.group(1).strip()
            
            # Extract content preservation status
            preserved_match = re.search(r'\*\*Original Content Preserved:\*\*\s*([^\n]+)', content)
            if preserved_match:
                metadata['content_preserved'] = preserved_match.group(1).strip()
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {archive_path}: {e}")
            return None
    
    def _extract_original_content_from_archive(self, archive_path: Path) -> Optional[str]:
        """
        Extract the original workflow content from an archived file.
        
        Args:
            archive_path: Path to the archived workflow file
            
        Returns:
            Original workflow content as string, or None if extraction failed
        """
        try:
            content = archive_path.read_text(encoding='utf-8')
            
            # Find the original workflow content section
            yaml_block_match = re.search(r'```yaml\n(.*?)\n```', content, re.DOTALL)
            if yaml_block_match:
                return yaml_block_match.group(1)
            
            # Fallback: look for content after "## Original Workflow Content"
            content_section_match = re.search(r'## Original Workflow Content\s*\n\n```yaml\n(.*?)\n```', content, re.DOTALL)
            if content_section_match:
                return content_section_match.group(1)
            
            self.logger.error(f"Could not find original content in archive: {archive_path}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting original content from {archive_path}: {e}")
            return None
    
    def _resolve_restore_filename_conflict(self, restore_path: Path) -> Path:
        """
        Resolve filename conflicts when restoring workflows.
        
        Args:
            restore_path: Proposed restore file path
            
        Returns:
            Path that doesn't conflict with existing files
        """
        if not restore_path.exists():
            return restore_path
        
        counter = 1
        base_name = restore_path.stem
        suffix = restore_path.suffix
        
        while True:
            new_name = f"{base_name}_restored_{counter}{suffix}"
            new_path = restore_path.parent / new_name
            if not new_path.exists():
                self.logger.info(f"Resolved restore filename conflict: {restore_path} -> {new_path}")
                return new_path
            counter += 1


class WorkflowCleanupExecutor:
    """Main executor for the CI/CD workflow cleanup process."""
    
    def __init__(self):
        """Initialize the cleanup executor with all required components."""
        self.detector = WorkflowFileDetector()
        self.archiver = WorkflowArchiver()
        self.gitignore_manager = GitIgnoreManager()
        self.verifier = WorkflowVerifier()
        self.logger = logging.getLogger(__name__)
    
    def execute_cleanup(self, dry_run: bool = False, auto_confirm: bool = False) -> bool:
        """
        Execute the complete workflow cleanup process.
        
        Args:
            dry_run: If True, show what would be done without making changes
            auto_confirm: If True, skip confirmation prompts
            
        Returns:
            True if cleanup completed successfully, False otherwise
        """
        self.logger.info("Starting CI/CD workflow cleanup process")
        
        try:
            # Phase 1: Detection and validation
            self.logger.info("Phase 1: Detecting and validating workflow files")
            workflow_files = self._detect_workflows()
            
            if not workflow_files:
                self.logger.info("No workflow files found - cleanup not needed")
                print("✓ No workflow files found. Cleanup not needed.")
                return True
            
            # Phase 2: Display findings and get confirmation
            if not self._display_findings_and_confirm(workflow_files, dry_run, auto_confirm):
                self.logger.info("Cleanup cancelled by user")
                print("Cleanup cancelled.")
                return False
            
            if dry_run:
                self.logger.info("Dry run completed - no changes made")
                print("✓ Dry run completed. No changes were made.")
                return True
            
            # Phase 3: Archive workflow files
            self.logger.info("Phase 2: Archiving workflow files")
            if not self._archive_workflows(workflow_files):
                return False
            
            # Phase 4: Update .gitignore
            self.logger.info("Phase 3: Updating .gitignore file")
            if not self._update_gitignore():
                return False
            
            # Phase 5: Verification
            self.logger.info("Phase 4: Verifying cleanup completion")
            if not self._verify_cleanup():
                return False
            
            self.logger.info("CI/CD workflow cleanup completed successfully")
            print("\n✓ Workflow cleanup completed successfully!")
            print("  - All workflow files have been archived")
            print("  - .gitignore has been updated")
            print("  - GitHub Actions will no longer execute these workflows")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Unexpected error during cleanup: {e}")
            print(f"✗ Error during cleanup: {e}")
            return False
    
    def _detect_workflows(self) -> List[Path]:
        """Detect and validate workflow files."""
        try:
            workflow_files = self.detector.scan_workflow_directory()
            existing_files, missing_files = self.detector.validate_workflow_files_exist(workflow_files)
            
            if missing_files:
                self.logger.warning(f"Found {len(missing_files)} missing workflow files")
                for missing_file in missing_files:
                    self.logger.warning(f"Missing file: {missing_file}")
            
            self.logger.info(f"Found {len(existing_files)} valid workflow files")
            return existing_files
            
        except FileNotFoundError:
            self.logger.error(".github/workflows directory not found")
            print("✗ Error: .github/workflows directory not found")
            return []
        except PermissionError:
            self.logger.error("Permission denied accessing workflows directory")
            print("✗ Error: Permission denied accessing workflows directory")
            return []
    
    def _display_findings_and_confirm(self, workflow_files: List[Path], dry_run: bool, auto_confirm: bool) -> bool:
        """Display findings and get user confirmation."""
        print(f"\n📋 Workflow Cleanup Summary:")
        print(f"   Found {len(workflow_files)} workflow file(s) to process:")
        
        for workflow_file in workflow_files:
            print(f"   - {workflow_file}")
        
        print(f"\n🔄 Planned Actions:")
        print(f"   1. Create 'archived-workflows' directory (if needed)")
        print(f"   2. Move workflow files to archive with .md extension")
        print(f"   3. Add metadata headers to archived files")
        print(f"   4. Update .gitignore to exclude archived workflows")
        print(f"   5. Verify no active workflow files remain")
        
        if dry_run:
            print(f"\n🔍 DRY RUN MODE: No changes will be made")
            return True
        
        if auto_confirm:
            print(f"\n✓ Auto-confirm enabled - proceeding with cleanup")
            return True
        
        print(f"\n⚠️  WARNING: This will disable all GitHub Actions workflows!")
        print(f"   Workflows will be preserved in archived-workflows/ directory")
        
        while True:
            response = input(f"\nProceed with cleanup? (yes/no): ").lower().strip()
            if response in ['yes', 'y']:
                self.logger.info("User confirmed cleanup execution")
                return True
            elif response in ['no', 'n']:
                self.logger.info("User cancelled cleanup execution")
                return False
            else:
                print("Please enter 'yes' or 'no'")
    
    def _archive_workflows(self, workflow_files: List[Path]) -> bool:
        """Archive workflow files."""
        print(f"\n📦 Archiving {len(workflow_files)} workflow file(s)...")
        
        results = self.archiver.archive_multiple_workflows(
            workflow_files,
            "Workflow not stopping - archived for troubleshooting"
        )
        
        successful_count = 0
        failed_count = 0
        
        for original_path, archived_path in results:
            if archived_path:
                print(f"   ✓ {original_path.name} -> {archived_path}")
                self.logger.info(f"Successfully archived: {original_path} -> {archived_path}")
                successful_count += 1
            else:
                print(f"   ✗ Failed to archive {original_path.name}")
                self.logger.error(f"Failed to archive: {original_path}")
                failed_count += 1
        
        if failed_count > 0:
            self.logger.error(f"Failed to archive {failed_count} workflow files")
            print(f"\n✗ Failed to archive {failed_count} workflow file(s)")
            return False
        
        print(f"   ✓ Successfully archived {successful_count} workflow file(s)")
        return True
    
    def _update_gitignore(self) -> bool:
        """Update .gitignore file."""
        print(f"\n📝 Updating .gitignore file...")
        
        # Check current status
        current_content = self.gitignore_manager.read_gitignore()
        if current_content:
            pattern_exists = self.gitignore_manager.check_pattern_exists(current_content)
            if pattern_exists:
                print(f"   ✓ Archive pattern already exists in .gitignore")
                self.logger.info("Archive pattern already exists in .gitignore")
                return True
        
        # Add the pattern
        success = self.gitignore_manager.add_ignore_pattern()
        if success:
            print(f"   ✓ Added archived-workflows/ to .gitignore")
            self.logger.info("Successfully updated .gitignore")
            
            # Validate the update
            if self.gitignore_manager.validate_gitignore_update():
                print(f"   ✓ .gitignore update verified")
                self.logger.info(".gitignore update validated")
                return True
            else:
                print(f"   ✗ .gitignore update verification failed")
                self.logger.error(".gitignore update validation failed")
                return False
        else:
            print(f"   ✗ Failed to update .gitignore")
            self.logger.error("Failed to update .gitignore")
            return False
    
    def _verify_cleanup(self) -> bool:
        """Verify that cleanup was completed successfully."""
        print(f"\n🔍 Verifying cleanup completion...")
        
        # Use the new verifier to check for remaining workflow files
        is_clean, remaining_files = self.verifier.verify_no_active_workflows()
        
        if not is_clean:
            print(f"   ✗ Found {len(remaining_files)} remaining workflow files:")
            for remaining_file in remaining_files:
                print(f"     - {remaining_file}")
            self.logger.error(f"Cleanup verification failed - {len(remaining_files)} files remain")
            return False
        else:
            print(f"   ✓ No workflow files remain in .github/workflows")
            self.logger.info("Cleanup verification successful - no workflow files remain")
        
        # List archived workflows with metadata
        archived_workflows = self.verifier.list_archived_workflows()
        if archived_workflows:
            print(f"   ✓ Archive directory contains {len(archived_workflows)} archived workflow(s)")
            for workflow in archived_workflows:
                print(f"     - {workflow['filename']} (archived: {workflow.get('archived_date', 'unknown')})")
            self.logger.info(f"Archive directory verified with {len(archived_workflows)} files")
        else:
            print(f"   ✗ Archive directory not found or contains no archived workflows")
            self.logger.error("Archive directory verification failed")
            return False
        
        return True
    
    def list_archived_workflows(self) -> bool:
        """List all archived workflows with their metadata."""
        print(f"\n📋 Archived Workflows:")
        
        archived_workflows = self.verifier.list_archived_workflows()
        
        if not archived_workflows:
            print(f"   No archived workflows found.")
            return True
        
        print(f"   Found {len(archived_workflows)} archived workflow(s):")
        print()
        
        for i, workflow in enumerate(archived_workflows, 1):
            print(f"   {i}. {workflow['filename']}")
            print(f"      Original: {workflow.get('original_path', 'unknown')}")
            print(f"      Archived: {workflow.get('archived_date', 'unknown')}")
            print(f"      Reason: {workflow.get('reason', 'unknown')}")
            print(f"      Content Preserved: {workflow.get('content_preserved', 'unknown')}")
            print()
        
        return True
    
    def rollback_workflows(self, specific_workflow: Optional[str] = None, auto_confirm: bool = False) -> bool:
        """
        Rollback archived workflows to restore them as active workflows.
        
        Args:
            specific_workflow: Name of specific workflow to rollback (e.g., "ci-cd.md")
            auto_confirm: If True, skip confirmation prompts
            
        Returns:
            True if rollback completed successfully, False otherwise
        """
        self.logger.info("Starting workflow rollback process")
        
        # List available archived workflows
        archived_workflows = self.verifier.list_archived_workflows()
        
        if not archived_workflows:
            print(f"\n📋 No archived workflows found to rollback.")
            return True
        
        print(f"\n📋 Available Archived Workflows:")
        for i, workflow in enumerate(archived_workflows, 1):
            print(f"   {i}. {workflow['filename']} (from {workflow.get('original_path', 'unknown')})")
        
        # Determine which workflows to rollback
        if specific_workflow:
            # Rollback specific workflow
            if not any(w['filename'] == specific_workflow for w in archived_workflows):
                print(f"\n✗ Archived workflow '{specific_workflow}' not found.")
                return False
            
            workflows_to_rollback = [specific_workflow]
            print(f"\n🔄 Rolling back specific workflow: {specific_workflow}")
        else:
            # Rollback all workflows
            workflows_to_rollback = [w['filename'] for w in archived_workflows]
            print(f"\n🔄 Rolling back all {len(workflows_to_rollback)} archived workflows")
        
        # Get confirmation
        if not auto_confirm:
            print(f"\n⚠️  WARNING: This will restore workflow files to .github/workflows/")
            print(f"   GitHub Actions will start executing these workflows again!")
            
            while True:
                response = input(f"\nProceed with rollback? (yes/no): ").lower().strip()
                if response in ['yes', 'y']:
                    self.logger.info("User confirmed rollback execution")
                    break
                elif response in ['no', 'n']:
                    self.logger.info("User cancelled rollback execution")
                    print("Rollback cancelled.")
                    return False
                else:
                    print("Please enter 'yes' or 'no'")
        
        # Perform rollback
        print(f"\n🔄 Rolling back workflows...")
        
        successful_count = 0
        failed_count = 0
        
        for workflow_filename in workflows_to_rollback:
            restored_path = self.verifier.rollback_workflow(workflow_filename)
            
            if restored_path:
                print(f"   ✓ {workflow_filename} -> {restored_path}")
                self.logger.info(f"Successfully rolled back: {workflow_filename} -> {restored_path}")
                successful_count += 1
            else:
                print(f"   ✗ Failed to rollback {workflow_filename}")
                self.logger.error(f"Failed to rollback: {workflow_filename}")
                failed_count += 1
        
        # Report results
        if failed_count > 0:
            print(f"\n⚠️  Rollback completed with {failed_count} failure(s)")
            print(f"   Successfully rolled back: {successful_count}")
            print(f"   Failed to rollback: {failed_count}")
            return False
        else:
            print(f"\n✓ Rollback completed successfully!")
            print(f"   Restored {successful_count} workflow file(s)")
            print(f"   GitHub Actions workflows are now active again")
            return True


def main():
    """Main function with command-line interface for workflow cleanup."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CI/CD Workflow Cleanup Tool - Safely disable GitHub Actions workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python workflow_cleanup.py                    # Interactive cleanup
  python workflow_cleanup.py --dry-run          # Show what would be done
  python workflow_cleanup.py --auto-confirm     # Skip confirmation prompts
  python workflow_cleanup.py --list             # List archived workflows
  python workflow_cleanup.py --verify           # Verify no active workflows remain
  python workflow_cleanup.py --rollback         # Rollback all archived workflows
  python workflow_cleanup.py --rollback ci-cd.md # Rollback specific workflow
  python workflow_cleanup.py --verbose          # Enable verbose logging
        """
    )
    
    # Action group - mutually exclusive actions
    action_group = parser.add_mutually_exclusive_group()
    
    action_group.add_argument(
        '--list',
        action='store_true',
        help='List all archived workflows with their metadata'
    )
    
    action_group.add_argument(
        '--verify',
        action='store_true',
        help='Verify that no active workflow files remain in .github/workflows'
    )
    
    action_group.add_argument(
        '--rollback',
        nargs='?',
        const='all',
        metavar='WORKFLOW',
        help='Rollback archived workflows (specify filename or use without argument for all)'
    )
    
    # Cleanup options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making any changes (cleanup only)'
    )
    
    parser.add_argument(
        '--auto-confirm',
        action='store_true',
        help='Skip confirmation prompts and proceed automatically'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    args = parser.parse_args()
    
    # Configure logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    # Initialize executor
    executor = WorkflowCleanupExecutor()
    
    # Handle different actions
    if args.list:
        # List archived workflows
        print("=" * 60)
        print("CI/CD Workflow Cleanup Tool - List Archived Workflows")
        print("=" * 60)
        
        success = executor.list_archived_workflows()
        exit_code = 0 if success else 1
        
    elif args.verify:
        # Verify no active workflows remain
        print("=" * 60)
        print("CI/CD Workflow Cleanup Tool - Verify Cleanup")
        print("=" * 60)
        
        is_clean, remaining_files = executor.verifier.verify_no_active_workflows()
        
        if is_clean:
            print("✓ Verification passed - no active workflow files found")
            archived_workflows = executor.verifier.list_archived_workflows()
            if archived_workflows:
                print(f"✓ Found {len(archived_workflows)} archived workflow(s)")
            exit_code = 0
        else:
            print(f"✗ Verification failed - found {len(remaining_files)} active workflow files:")
            for remaining_file in remaining_files:
                print(f"  - {remaining_file}")
            exit_code = 1
            
    elif args.rollback is not None:
        # Rollback workflows
        print("=" * 60)
        print("CI/CD Workflow Cleanup Tool - Rollback Workflows")
        print("=" * 60)
        
        specific_workflow = None if args.rollback == 'all' else args.rollback
        success = executor.rollback_workflows(
            specific_workflow=specific_workflow,
            auto_confirm=args.auto_confirm
        )
        exit_code = 0 if success else 1
        
    else:
        # Default cleanup action
        print("=" * 60)
        print("CI/CD Workflow Cleanup Tool")
        print("=" * 60)
        print("This tool will safely disable GitHub Actions workflows by:")
        print("• Moving workflow files to archived-workflows/ directory")
        print("• Renaming files with .md extension to prevent execution")
        print("• Adding metadata headers with archival information")
        print("• Updating .gitignore to exclude archived workflows")
        print("=" * 60)
        
        if args.dry_run:
            print("🔍 DRY RUN MODE: No changes will be made")
            print("=" * 60)
        
        # Execute cleanup
        success = executor.execute_cleanup(
            dry_run=args.dry_run,
            auto_confirm=args.auto_confirm
        )
        
        # Exit with appropriate code
        exit_code = 0 if success else 1
        
        if success:
            print(f"\n🎉 Cleanup completed successfully!")
            if not args.dry_run:
                print(f"   Your GitHub Actions workflows have been safely disabled.")
                print(f"   Archived workflows can be found in: archived-workflows/")
        else:
            print(f"\n❌ Cleanup failed!")
            print(f"   Please check the error messages above and try again.")
    
    return exit_code


if __name__ == "__main__":
    main()