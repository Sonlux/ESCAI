"""
Test coverage analysis utilities for ESCAI Framework.
Provides tools to measure and analyze test coverage across all components.
"""

import ast
import os
import sys
from typing import Dict, List, Set, Any, Optional, Tuple
from pathlib import Path
import subprocess
import json
import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CoverageReport:
    """Coverage report data structure."""
    module_name: str
    total_lines: int
    covered_lines: int
    missing_lines: List[int]
    coverage_percentage: float
    functions: Dict[str, bool]  # function_name -> is_covered
    classes: Dict[str, float]   # class_name -> coverage_percentage


@dataclass
class ComponentCoverage:
    """Coverage data for a specific component."""
    component_name: str
    files: Dict[str, CoverageReport]
    overall_coverage: float
    critical_functions_covered: int
    total_critical_functions: int


class CoverageAnalyzer:
    """Analyze test coverage for ESCAI Framework components."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.source_dir = self.project_root / "escai_framework"
        self.test_dir = self.project_root / "tests"
        
        # Define critical functions that must be tested
        self.critical_functions = {
            "instrumentation": [
                "start_monitoring", "stop_monitoring", "capture_event",
                "process_event", "get_monitoring_status"
            ],
            "core": [
                "extract_beliefs", "extract_knowledge", "extract_goals",
                "mine_patterns", "detect_anomalies", "discover_relationships",
                "predict_success", "explain_behavior"
            ],
            "analytics": [
                "train_model", "evaluate_model", "predict", "analyze_patterns",
                "calculate_statistics", "generate_insights"
            ],
            "api": [
                "create_app", "start_monitoring_endpoint", "get_analysis_endpoint",
                "authenticate_user", "validate_request"
            ],
            "storage": [
                "connect", "save", "load", "query", "migrate", "backup"
            ]
        }
    
    def run_coverage_analysis(self, include_integration: bool = True) -> Dict[str, ComponentCoverage]:
        """Run comprehensive coverage analysis."""
        print("Running coverage analysis...")
        
        # Run pytest with coverage
        coverage_data = self._run_pytest_coverage(include_integration)
        
        # Analyze coverage by component
        component_coverage = {}
        
        for component in ["instrumentation", "core", "analytics", "api", "storage", "models", "utils"]:
            coverage = self._analyze_component_coverage(component, coverage_data)
            if coverage:
                component_coverage[component] = coverage
        
        return component_coverage
    
    def _run_pytest_coverage(self, include_integration: bool) -> Dict[str, Any]:
        """Run pytest with coverage measurement."""
        
        # Prepare pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            "--cov=escai_framework",
            "--cov-report=json",
            "--cov-report=term-missing",
            "-v"
        ]
        
        # Add test directories
        if include_integration:
            cmd.extend([
                str(self.test_dir / "unit"),
                str(self.test_dir / "integration"),
                str(self.test_dir / "performance"),
                str(self.test_dir / "accuracy")
            ])
        else:
            cmd.append(str(self.test_dir / "unit"))
        
        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Load coverage JSON report
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    return json.load(f)
            else:
                print("Warning: Coverage JSON file not found")
                return {}
        
        except subprocess.TimeoutExpired:
            print("Error: Coverage analysis timed out")
            return {}
        except Exception as e:
            print(f"Error running coverage analysis: {e}")
            return {}
    
    def _analyze_component_coverage(self, component: str, coverage_data: Dict[str, Any]) -> Optional[ComponentCoverage]:
        """Analyze coverage for a specific component."""
        
        component_dir = self.source_dir / component
        if not component_dir.exists():
            return None
        
        files_coverage = {}
        total_lines = 0
        covered_lines = 0
        
        # Analyze each Python file in the component
        for py_file in component_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            
            relative_path = str(py_file.relative_to(self.project_root))
            
            # Get coverage data for this file
            file_coverage_data = coverage_data.get("files", {}).get(relative_path, {})
            
            if file_coverage_data:
                file_report = self._create_file_coverage_report(py_file, file_coverage_data)
                files_coverage[relative_path] = file_report
                
                total_lines += file_report.total_lines
                covered_lines += file_report.covered_lines
        
        if total_lines == 0:
            return None
        
        overall_coverage = (covered_lines / total_lines) * 100
        
        # Check critical function coverage
        critical_covered, critical_total = self._check_critical_function_coverage(
            component, files_coverage
        )
        
        return ComponentCoverage(
            component_name=component,
            files=files_coverage,
            overall_coverage=overall_coverage,
            critical_functions_covered=critical_covered,
            total_critical_functions=critical_total
        )
    
    def _create_file_coverage_report(self, file_path: Path, coverage_data: Dict[str, Any]) -> CoverageReport:
        """Create coverage report for a single file."""
        
        # Parse the Python file to get function and class information
        functions, classes = self._parse_python_file(file_path)
        
        # Get coverage information
        executed_lines = set(coverage_data.get("executed_lines", []))
        missing_lines = coverage_data.get("missing_lines", [])
        
        # Calculate total lines (excluding empty lines and comments)
        total_lines = len(self._get_code_lines(file_path))
        covered_lines = len(executed_lines)
        
        coverage_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
        
        # Determine function coverage
        function_coverage = {}
        for func_name, func_lines in functions.items():
            is_covered = any(line in executed_lines for line in func_lines)
            function_coverage[func_name] = is_covered
        
        # Determine class coverage
        class_coverage = {}
        for class_name, class_lines in classes.items():
            covered_class_lines = sum(1 for line in class_lines if line in executed_lines)
            class_coverage_pct = (covered_class_lines / len(class_lines) * 100) if class_lines else 0
            class_coverage[class_name] = class_coverage_pct
        
        return CoverageReport(
            module_name=str(file_path.relative_to(self.source_dir)),
            total_lines=total_lines,
            covered_lines=covered_lines,
            missing_lines=missing_lines,
            coverage_percentage=coverage_percentage,
            functions=function_coverage,
            classes=class_coverage
        )
    
    def _parse_python_file(self, file_path: Path) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """Parse Python file to extract function and class line numbers."""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            functions = {}
            classes = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_lines = list(range(node.lineno, node.end_lineno + 1 if node.end_lineno else node.lineno + 1))
                    functions[node.name] = func_lines
                elif isinstance(node, ast.ClassDef):
                    class_lines = list(range(node.lineno, node.end_lineno + 1 if node.end_lineno else node.lineno + 1))
                    classes[node.name] = class_lines
            
            return functions, classes
        
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return {}, {}
    
    def _get_code_lines(self, file_path: Path) -> List[int]:
        """Get line numbers that contain actual code (not comments or empty lines)."""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            code_lines = []
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    code_lines.append(i)
            
            return code_lines
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
    
    def _check_critical_function_coverage(
        self, 
        component: str, 
        files_coverage: Dict[str, CoverageReport]
    ) -> Tuple[int, int]:
        """Check coverage of critical functions for a component."""
        
        critical_functions = self.critical_functions.get(component, [])
        if not critical_functions:
            return 0, 0
        
        covered_count = 0
        total_count = 0
        
        for func_name in critical_functions:
            total_count += 1
            
            # Check if this function is covered in any file
            for file_report in files_coverage.values():
                if func_name in file_report.functions and file_report.functions[func_name]:
                    covered_count += 1
                    break
        
        return covered_count, total_count
    
    def generate_coverage_report(self, coverage_data: Dict[str, ComponentCoverage]) -> str:
        """Generate a comprehensive coverage report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ESCAI FRAMEWORK TEST COVERAGE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Overall summary
        total_coverage = 0
        total_components = 0
        total_critical_covered = 0
        total_critical_functions = 0
        
        for component_coverage in coverage_data.values():
            total_coverage += component_coverage.overall_coverage
            total_components += 1
            total_critical_covered += component_coverage.critical_functions_covered
            total_critical_functions += component_coverage.total_critical_functions
        
        overall_avg = total_coverage / total_components if total_components > 0 else 0
        critical_coverage = (total_critical_covered / total_critical_functions * 100) if total_critical_functions > 0 else 0
        
        report_lines.append("OVERALL SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Average Coverage: {overall_avg:.1f}%")
        report_lines.append(f"Critical Functions Coverage: {critical_coverage:.1f}% ({total_critical_covered}/{total_critical_functions})")
        report_lines.append("")
        
        # Component details
        for component_name, component_coverage in coverage_data.items():
            report_lines.append(f"COMPONENT: {component_name.upper()}")
            report_lines.append("-" * 40)
            report_lines.append(f"Overall Coverage: {component_coverage.overall_coverage:.1f}%")
            report_lines.append(f"Critical Functions: {component_coverage.critical_functions_covered}/{component_coverage.total_critical_functions}")
            report_lines.append("")
            
            # File details
            for file_path, file_report in component_coverage.files.items():
                report_lines.append(f"  {file_report.module_name}: {file_report.coverage_percentage:.1f}%")
                
                # Show uncovered critical functions
                uncovered_functions = [
                    func for func, covered in file_report.functions.items() 
                    if not covered and func in self.critical_functions.get(component_name, [])
                ]
                
                if uncovered_functions:
                    report_lines.append(f"    Uncovered critical functions: {', '.join(uncovered_functions)}")
            
            report_lines.append("")
        
        # Coverage requirements check
        report_lines.append("COVERAGE REQUIREMENTS CHECK")
        report_lines.append("-" * 40)
        
        requirements_met = True
        
        if overall_avg < 95.0:
            report_lines.append(f"❌ Overall coverage {overall_avg:.1f}% below 95% requirement")
            requirements_met = False
        else:
            report_lines.append(f"✅ Overall coverage {overall_avg:.1f}% meets 95% requirement")
        
        if critical_coverage < 100.0:
            report_lines.append(f"❌ Critical function coverage {critical_coverage:.1f}% below 100% requirement")
            requirements_met = False
        else:
            report_lines.append(f"✅ Critical function coverage {critical_coverage:.1f}% meets 100% requirement")
        
        # Component-specific requirements
        for component_name, component_coverage in coverage_data.items():
            if component_coverage.overall_coverage < 90.0:
                report_lines.append(f"❌ {component_name} coverage {component_coverage.overall_coverage:.1f}% below 90% requirement")
                requirements_met = False
        
        report_lines.append("")
        
        if requirements_met:
            report_lines.append("🎉 ALL COVERAGE REQUIREMENTS MET!")
        else:
            report_lines.append("⚠️  COVERAGE REQUIREMENTS NOT MET - ADDITIONAL TESTS NEEDED")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def identify_coverage_gaps(self, coverage_data: Dict[str, ComponentCoverage]) -> Dict[str, List[str]]:
        """Identify specific areas that need more test coverage."""
        
        gaps = {}
        
        for component_name, component_coverage in coverage_data.items():
            component_gaps = []
            
            # Check overall component coverage
            if component_coverage.overall_coverage < 90.0:
                component_gaps.append(f"Overall coverage {component_coverage.overall_coverage:.1f}% below 90%")
            
            # Check critical function coverage
            if component_coverage.critical_functions_covered < component_coverage.total_critical_functions:
                missing = component_coverage.total_critical_functions - component_coverage.critical_functions_covered
                component_gaps.append(f"{missing} critical functions not covered")
            
            # Check individual files
            for file_path, file_report in component_coverage.files.items():
                if file_report.coverage_percentage < 85.0:
                    component_gaps.append(f"{file_report.module_name}: {file_report.coverage_percentage:.1f}% coverage")
                
                # Check for uncovered functions
                uncovered_functions = [
                    func for func, covered in file_report.functions.items() if not covered
                ]
                
                if uncovered_functions:
                    component_gaps.append(f"{file_report.module_name}: uncovered functions - {', '.join(uncovered_functions[:3])}{'...' if len(uncovered_functions) > 3 else ''}")
            
            if component_gaps:
                gaps[component_name] = component_gaps
        
        return gaps
    
    def suggest_test_improvements(self, coverage_data: Dict[str, ComponentCoverage]) -> List[str]:
        """Suggest specific improvements to increase test coverage."""
        
        suggestions = []
        
        gaps = self.identify_coverage_gaps(coverage_data)
        
        for component_name, component_gaps in gaps.items():
            suggestions.append(f"\n{component_name.upper()} Component:")
            
            for gap in component_gaps:
                if "critical functions" in gap:
                    suggestions.append(f"  - Add unit tests for critical functions in {component_name}")
                elif "uncovered functions" in gap:
                    suggestions.append(f"  - Add tests for specific functions: {gap}")
                elif "coverage" in gap and "%" in gap:
                    suggestions.append(f"  - Increase test coverage for: {gap}")
        
        # General suggestions
        if any("critical functions" in gap for gaps_list in gaps.values() for gap in gaps_list):
            suggestions.append("\nGeneral Suggestions:")
            suggestions.append("  - Focus on testing critical business logic functions")
            suggestions.append("  - Add integration tests for component interactions")
            suggestions.append("  - Include error handling and edge case tests")
        
        return suggestions
    
    def save_coverage_report(self, coverage_data: Dict[str, ComponentCoverage], output_file: str):
        """Save coverage report to file."""
        
        report = self.generate_coverage_report(coverage_data)
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Coverage report saved to: {output_file}")


def main():
    """Main function to run coverage analysis."""
    
    analyzer = CoverageAnalyzer()
    
    print("Starting comprehensive test coverage analysis...")
    
    # Run coverage analysis
    coverage_data = analyzer.run_coverage_analysis(include_integration=True)
    
    if not coverage_data:
        print("No coverage data available. Make sure tests are properly configured.")
        return
    
    # Generate and display report
    report = analyzer.generate_coverage_report(coverage_data)
    print(report)
    
    # Save report to file
    analyzer.save_coverage_report(coverage_data, "coverage_report.txt")
    
    # Show improvement suggestions
    suggestions = analyzer.suggest_test_improvements(coverage_data)
    if suggestions:
        print("\nSUGGESTED IMPROVEMENTS:")
        print("=" * 40)
        for suggestion in suggestions:
            print(suggestion)


if __name__ == "__main__":
    main()