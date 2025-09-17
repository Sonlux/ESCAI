"""
Prerequisite checker and requirement documentation.

This module validates system prerequisites and provides detailed
requirement documentation for CLI commands.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import subprocess
import sys
import importlib
import os
import platform


class PrerequisiteType(Enum):
    """Types of prerequisites that can be checked."""
    PYTHON_VERSION = "python_version"
    PYTHON_PACKAGE = "python_package"
    SYSTEM_COMMAND = "system_command"
    ENVIRONMENT_VARIABLE = "environment_variable"
    FILE_PATH = "file_path"
    DIRECTORY_PATH = "directory_path"
    NETWORK_ACCESS = "network_access"
    PERMISSIONS = "permissions"


class PrerequisiteStatus(Enum):
    """Status of prerequisite checks."""
    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"
    WARNING = "warning"
    UNKNOWN = "unknown"


@dataclass
class Prerequisite:
    """Represents a single prerequisite requirement."""
    name: str
    type: PrerequisiteType
    description: str
    requirement: str
    check_method: str
    install_instructions: str
    importance: str  # "required", "recommended", "optional"
    status: Optional[PrerequisiteStatus] = None
    details: Optional[str] = None


@dataclass
class PrerequisiteCheckResult:
    """Result of prerequisite checking."""
    command: str
    prerequisites: List[Prerequisite]
    all_satisfied: bool
    required_satisfied: bool
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]


class PrerequisiteChecker:
    """Validates system prerequisites and provides requirement documentation."""
    
    def __init__(self):
        self._command_prerequisites = self._initialize_command_prerequisites()
        self._system_info = self._gather_system_info()
    
    def check_command_prerequisites(self, command: str) -> PrerequisiteCheckResult:
        """
        Check all prerequisites for a specific command.
        
        Args:
            command: The command name to check prerequisites for
            
        Returns:
            Complete prerequisite check result
        """
        if command not in self._command_prerequisites:
            return PrerequisiteCheckResult(
                command=command,
                prerequisites=[],
                all_satisfied=False,
                required_satisfied=False,
                warnings=[f"Prerequisites not defined for command: {command}"],
                errors=[],
                recommendations=[]
            )
        
        prerequisites = self._command_prerequisites[command].copy()
        warnings = []
        errors = []
        recommendations = []
        
        # Check each prerequisite
        for prereq in prerequisites:
            status, details = self._check_prerequisite(prereq)
            prereq.status = status
            prereq.details = details
            
            if status == PrerequisiteStatus.NOT_SATISFIED:
                if prereq.importance == "required":
                    errors.append(f"Required: {prereq.name} - {details}")
                    recommendations.append(f"Install {prereq.name}: {prereq.install_instructions}")
                else:
                    warnings.append(f"Recommended: {prereq.name} - {details}")
                    recommendations.append(f"Consider installing {prereq.name}: {prereq.install_instructions}")
            elif status == PrerequisiteStatus.WARNING:
                warnings.append(f"Warning: {prereq.name} - {details}")
            elif status == PrerequisiteStatus.UNKNOWN:
                warnings.append(f"Could not verify: {prereq.name} - {details}")
        
        # Determine overall satisfaction
        required_satisfied = all(
            p.status == PrerequisiteStatus.SATISFIED 
            for p in prerequisites 
            if p.importance == "required"
        )
        
        all_satisfied = all(
            p.status == PrerequisiteStatus.SATISFIED 
            for p in prerequisites
        )
        
        return PrerequisiteCheckResult(
            command=command,
            prerequisites=prerequisites,
            all_satisfied=all_satisfied,
            required_satisfied=required_satisfied,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations
        )
    
    def generate_prerequisite_documentation(self, command: str) -> str:
        """
        Generate comprehensive prerequisite documentation for a command.
        
        Args:
            command: The command name to generate documentation for
            
        Returns:
            Formatted prerequisite documentation
        """
        if command not in self._command_prerequisites:
            return f"Prerequisite documentation not available for command: {command}"
        
        prerequisites = self._command_prerequisites[command]
        
        doc = f"""
# Prerequisites for: {command}

## System Requirements

### Required Prerequisites
These must be satisfied for the command to work properly:

"""
        
        required_prereqs = [p for p in prerequisites if p.importance == "required"]
        for prereq in required_prereqs:
            doc += f"""
#### {prereq.name}
- **Description**: {prereq.description}
- **Requirement**: {prereq.requirement}
- **Installation**: {prereq.install_instructions}

"""
        
        doc += """
### Recommended Prerequisites
These improve functionality and performance:

"""
        
        recommended_prereqs = [p for p in prerequisites if p.importance == "recommended"]
        for prereq in recommended_prereqs:
            doc += f"""
#### {prereq.name}
- **Description**: {prereq.description}
- **Requirement**: {prereq.requirement}
- **Installation**: {prereq.install_instructions}

"""
        
        doc += """
### Optional Prerequisites
These provide additional features:

"""
        
        optional_prereqs = [p for p in prerequisites if p.importance == "optional"]
        for prereq in optional_prereqs:
            doc += f"""
#### {prereq.name}
- **Description**: {prereq.description}
- **Requirement**: {prereq.requirement}
- **Installation**: {prereq.install_instructions}

"""
        
        doc += f"""
## Quick Setup Guide

### 1. Check Prerequisites
```bash
escai check-prerequisites {command}
```

### 2. Install Required Dependencies
```bash
# Python packages
pip install -r requirements.txt

# System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3-dev build-essential

# System dependencies (macOS)
brew install python3
```

### 3. Verify Installation
```bash
escai {command} --help
```

## Troubleshooting

### Common Issues

**Issue**: Command not found
- **Solution**: Ensure ESCAI is properly installed: `pip install escai-framework`

**Issue**: Permission denied
- **Solution**: Check file permissions and run with appropriate privileges

**Issue**: Import errors
- **Solution**: Verify all Python dependencies are installed: `pip check`

**Issue**: Framework not detected
- **Solution**: Ensure target framework (LangChain, AutoGen, etc.) is installed and configured

### Getting Help

If you encounter issues not covered here:
1. Run `escai check-prerequisites {command}` for detailed diagnostics
2. Check the troubleshooting guide: `escai help troubleshooting`
3. Review logs for detailed error information
4. Contact support with system information and error details
"""
        
        return doc.strip()
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive system diagnostic information.
        
        Returns:
            Dictionary containing system diagnostic information
        """
        return {
            "system_info": self._system_info,
            "python_info": {
                "version": sys.version,
                "executable": sys.executable,
                "path": sys.path[:3]  # First 3 entries for brevity
            },
            "environment": {
                "platform": platform.platform(),
                "architecture": platform.architecture(),
                "processor": platform.processor()
            },
            "escai_info": self._get_escai_info()
        }
    
    def _check_prerequisite(self, prereq: Prerequisite) -> Tuple[PrerequisiteStatus, str]:
        """Check a single prerequisite and return status and details."""
        try:
            if prereq.type == PrerequisiteType.PYTHON_VERSION:
                return self._check_python_version(prereq.requirement)
            elif prereq.type == PrerequisiteType.PYTHON_PACKAGE:
                return self._check_python_package(prereq.requirement)
            elif prereq.type == PrerequisiteType.SYSTEM_COMMAND:
                return self._check_system_command(prereq.requirement)
            elif prereq.type == PrerequisiteType.ENVIRONMENT_VARIABLE:
                return self._check_environment_variable(prereq.requirement)
            elif prereq.type == PrerequisiteType.FILE_PATH:
                return self._check_file_path(prereq.requirement)
            elif prereq.type == PrerequisiteType.DIRECTORY_PATH:
                return self._check_directory_path(prereq.requirement)
            else:
                return PrerequisiteStatus.UNKNOWN, "Check method not implemented"
        except Exception as e:
            return PrerequisiteStatus.UNKNOWN, f"Error checking prerequisite: {str(e)}"
    
    def _check_python_version(self, requirement: str) -> Tuple[PrerequisiteStatus, str]:
        """Check Python version requirement."""
        current_version = sys.version_info
        # Parse requirement like ">=3.8"
        if ">=" in requirement:
            min_version = tuple(map(int, requirement.split(">=")[1].split(".")))
            if current_version >= min_version:
                # Handle both sys.version_info and tuple for testing
                if hasattr(current_version, 'major'):
                    version_str = f"Python {current_version.major}.{current_version.minor}.{current_version.micro}"
                else:
                    version_str = f"Python {current_version[0]}.{current_version[1]}.{current_version[2]}"
                return PrerequisiteStatus.SATISFIED, version_str
            else:
                # Handle both sys.version_info and tuple for testing
                if hasattr(current_version, 'major'):
                    version_str = f"Python {current_version.major}.{current_version.minor}.{current_version.micro}"
                else:
                    version_str = f"Python {current_version[0]}.{current_version[1]}.{current_version[2]}"
                return PrerequisiteStatus.NOT_SATISFIED, f"{version_str} < {requirement}"
        return PrerequisiteStatus.UNKNOWN, "Version requirement format not supported"
    
    def _check_python_package(self, package_name: str) -> Tuple[PrerequisiteStatus, str]:
        """Check if Python package is installed."""
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            return PrerequisiteStatus.SATISFIED, f"Installed version: {version}"
        except ImportError:
            return PrerequisiteStatus.NOT_SATISFIED, "Package not installed"
    
    def _check_system_command(self, command: str) -> Tuple[PrerequisiteStatus, str]:
        """Check if system command is available."""
        try:
            result = subprocess.run(['which', command], capture_output=True, text=True)
            if result.returncode == 0:
                return PrerequisiteStatus.SATISFIED, f"Found at: {result.stdout.strip()}"
            else:
                return PrerequisiteStatus.NOT_SATISFIED, "Command not found in PATH"
        except Exception:
            return PrerequisiteStatus.UNKNOWN, "Could not check command availability"
    
    def _check_environment_variable(self, var_name: str) -> Tuple[PrerequisiteStatus, str]:
        """Check if environment variable is set."""
        value = os.environ.get(var_name)
        if value:
            return PrerequisiteStatus.SATISFIED, f"Set to: {value[:50]}..." if len(value) > 50 else f"Set to: {value}"
        else:
            return PrerequisiteStatus.NOT_SATISFIED, "Environment variable not set"
    
    def _check_file_path(self, file_path: str) -> Tuple[PrerequisiteStatus, str]:
        """Check if file exists and is accessible."""
        if os.path.isfile(file_path):
            if os.access(file_path, os.R_OK):
                return PrerequisiteStatus.SATISFIED, "File exists and is readable"
            else:
                return PrerequisiteStatus.WARNING, "File exists but is not readable"
        else:
            return PrerequisiteStatus.NOT_SATISFIED, "File does not exist"
    
    def _check_directory_path(self, dir_path: str) -> Tuple[PrerequisiteStatus, str]:
        """Check if directory exists and is accessible."""
        if os.path.isdir(dir_path):
            if os.access(dir_path, os.R_OK):
                return PrerequisiteStatus.SATISFIED, "Directory exists and is accessible"
            else:
                return PrerequisiteStatus.WARNING, "Directory exists but is not accessible"
        else:
            return PrerequisiteStatus.NOT_SATISFIED, "Directory does not exist"
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather basic system information."""
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "python_executable": sys.executable
        }
    
    def _get_escai_info(self) -> Dict[str, Any]:
        """Get ESCAI framework information."""
        try:
            import escai_framework
            return {
                "version": getattr(escai_framework, '__version__', 'unknown'),
                "installation_path": escai_framework.__file__
            }
        except ImportError:
            return {
                "version": "not_installed",
                "installation_path": "not_found"
            }
    
    def _initialize_command_prerequisites(self) -> Dict[str, List[Prerequisite]]:
        """Initialize prerequisite definitions for all commands."""
        return {
            "monitor": [
                Prerequisite(
                    name="Python 3.8+",
                    type=PrerequisiteType.PYTHON_VERSION,
                    description="Python version 3.8 or higher required for async support",
                    requirement=">=3.8",
                    check_method="sys.version_info",
                    install_instructions="Install Python 3.8+ from python.org or use package manager",
                    importance="required"
                ),
                Prerequisite(
                    name="ESCAI Framework",
                    type=PrerequisiteType.PYTHON_PACKAGE,
                    description="Core ESCAI framework for agent monitoring",
                    requirement="escai_framework",
                    check_method="importlib.import_module",
                    install_instructions="pip install escai-framework",
                    importance="required"
                ),
                Prerequisite(
                    name="Rich Console",
                    type=PrerequisiteType.PYTHON_PACKAGE,
                    description="Rich library for advanced terminal UI",
                    requirement="rich",
                    check_method="importlib.import_module",
                    install_instructions="pip install rich",
                    importance="required"
                ),
                Prerequisite(
                    name="Target Framework",
                    type=PrerequisiteType.PYTHON_PACKAGE,
                    description="Agent framework to monitor (LangChain, AutoGen, etc.)",
                    requirement="langchain",  # Example - would be dynamic
                    check_method="importlib.import_module",
                    install_instructions="pip install langchain (or your target framework)",
                    importance="required"
                ),
                Prerequisite(
                    name="SQLite3",
                    type=PrerequisiteType.PYTHON_PACKAGE,
                    description="Database for session storage",
                    requirement="sqlite3",
                    check_method="importlib.import_module",
                    install_instructions="Usually included with Python",
                    importance="required"
                ),
                Prerequisite(
                    name="Pandas",
                    type=PrerequisiteType.PYTHON_PACKAGE,
                    description="Data manipulation for analysis",
                    requirement="pandas",
                    check_method="importlib.import_module",
                    install_instructions="pip install pandas",
                    importance="recommended"
                ),
                Prerequisite(
                    name="NumPy",
                    type=PrerequisiteType.PYTHON_PACKAGE,
                    description="Numerical computing support",
                    requirement="numpy",
                    check_method="importlib.import_module",
                    install_instructions="pip install numpy",
                    importance="recommended"
                )
            ],
            "analyze": [
                Prerequisite(
                    name="Python 3.8+",
                    type=PrerequisiteType.PYTHON_VERSION,
                    description="Python version 3.8 or higher required",
                    requirement=">=3.8",
                    check_method="sys.version_info",
                    install_instructions="Install Python 3.8+ from python.org",
                    importance="required"
                ),
                Prerequisite(
                    name="SciPy",
                    type=PrerequisiteType.PYTHON_PACKAGE,
                    description="Scientific computing library for statistical analysis",
                    requirement="scipy",
                    check_method="importlib.import_module",
                    install_instructions="pip install scipy",
                    importance="required"
                ),
                Prerequisite(
                    name="Scikit-learn",
                    type=PrerequisiteType.PYTHON_PACKAGE,
                    description="Machine learning library for pattern analysis",
                    requirement="sklearn",
                    check_method="importlib.import_module",
                    install_instructions="pip install scikit-learn",
                    importance="required"
                ),
                Prerequisite(
                    name="Statsmodels",
                    type=PrerequisiteType.PYTHON_PACKAGE,
                    description="Statistical modeling and analysis",
                    requirement="statsmodels",
                    check_method="importlib.import_module",
                    install_instructions="pip install statsmodels",
                    importance="recommended"
                ),
                Prerequisite(
                    name="NetworkX",
                    type=PrerequisiteType.PYTHON_PACKAGE,
                    description="Graph analysis for causal relationships",
                    requirement="networkx",
                    check_method="importlib.import_module",
                    install_instructions="pip install networkx",
                    importance="recommended"
                ),
                Prerequisite(
                    name="Matplotlib",
                    type=PrerequisiteType.PYTHON_PACKAGE,
                    description="Plotting library for visualizations",
                    requirement="matplotlib",
                    check_method="importlib.import_module",
                    install_instructions="pip install matplotlib",
                    importance="optional"
                )
            ],
            "config": [
                Prerequisite(
                    name="PyYAML",
                    type=PrerequisiteType.PYTHON_PACKAGE,
                    description="YAML parsing for configuration files",
                    requirement="yaml",
                    check_method="importlib.import_module",
                    install_instructions="pip install pyyaml",
                    importance="required"
                ),
                Prerequisite(
                    name="Pydantic",
                    type=PrerequisiteType.PYTHON_PACKAGE,
                    description="Data validation for configuration",
                    requirement="pydantic",
                    check_method="importlib.import_module",
                    install_instructions="pip install pydantic",
                    importance="required"
                )
            ]
        }