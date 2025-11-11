#!/usr/bin/env python3
"""
Architecture Compliance Checker

Automatically verifies code compliance with ARCHITECTURE.md standards.

Run with: python scripts/check_architecture_compliance.py
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class ArchitectureComplianceChecker:
    """Checks code compliance with architecture standards."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.violations: List[Dict[str, str]] = []
        self.checked_files = 0
        
    def check_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Check a single Python file for compliance."""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            violations.append({
                'file': str(file_path),
                'type': 'syntax_error',
                'message': f"Syntax error: {e}",
                'line': e.lineno
            })
            return violations
        
        # Check for type hints
        violations.extend(self._check_type_hints(tree, file_path, content))
        
        # Check for docstrings
        violations.extend(self._check_docstrings(tree, file_path))
        
        # Check for bare except clauses
        violations.extend(self._check_bare_except(tree, file_path))
        
        # Check naming conventions
        violations.extend(self._check_naming_conventions(tree, file_path))
        
        # Check for hardcoded values
        violations.extend(self._check_hardcoded_values(content, file_path))
        
        return violations
    
    def _check_type_hints(self, tree: ast.AST, file_path: Path, content: str) -> List[Dict[str, str]]:
        """Check that all public functions have type hints."""
        violations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private methods
                if node.name.startswith('_'):
                    continue
                
                # Check if function has type hints
                if not node.returns and not any(
                    isinstance(arg.annotation, ast.Name) or 
                    isinstance(arg.annotation, ast.Constant) or
                    isinstance(arg.annotation, ast.Subscript)
                    for arg in node.args.args
                ):
                    # Check if it's a test file (tests don't need type hints)
                    if 'test' in str(file_path).lower():
                        continue
                    
                    violations.append({
                        'file': str(file_path),
                        'type': 'missing_type_hints',
                        'message': f"Function '{node.name}' missing type hints",
                        'line': node.lineno
                    })
        
        return violations
    
    def _check_docstrings(self, tree: ast.AST, file_path: Path) -> List[Dict[str, str]]:
        """Check that all public functions and classes have docstrings."""
        violations = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # Skip private methods
                if node.name.startswith('_'):
                    continue
                
                # Skip test files
                if 'test' in str(file_path).lower():
                    continue
                
                # Check for docstring
                if not ast.get_docstring(node):
                    violations.append({
                        'file': str(file_path),
                        'type': 'missing_docstring',
                        'message': f"{'Class' if isinstance(node, ast.ClassDef) else 'Function'} '{node.name}' missing docstring",
                        'line': node.lineno
                    })
        
        return violations
    
    def _check_bare_except(self, tree: ast.AST, file_path: Path) -> List[Dict[str, str]]:
        """Check for bare except clauses."""
        violations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    violations.append({
                        'file': str(file_path),
                        'type': 'bare_except',
                        'message': "Bare except clause found (use 'except Exception as e:')",
                        'line': node.lineno
                    })
        
        return violations
    
    def _check_naming_conventions(self, tree: ast.AST, file_path: Path) -> List[Dict[str, str]]:
        """Check naming conventions."""
        violations = []
        
        for node in ast.walk(tree):
            # Check class names (PascalCase)
            if isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    violations.append({
                        'file': str(file_path),
                        'type': 'naming_convention',
                        'message': f"Class '{node.name}' should be PascalCase",
                        'line': node.lineno
                    })
            
            # Check function names (snake_case)
            if isinstance(node, ast.FunctionDef):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    violations.append({
                        'file': str(file_path),
                        'type': 'naming_convention',
                        'message': f"Function '{node.name}' should be snake_case",
                        'line': node.lineno
                    })
        
        return violations
    
    def _check_hardcoded_values(self, content: str, file_path: Path) -> List[Dict[str, str]]:
        """Check for hardcoded values that should be in config."""
        violations = []
        
        # Common hardcoded values to flag
        hardcoded_patterns = [
            (r'["\']postgresql://', 'Database URL should be in config'),
            (r'["\']https?://api\.', 'API URLs should be in config'),
            (r'["\']sk-', 'API keys should be in environment variables'),
            (r'["\']Bearer\s+', 'Tokens should be in environment variables'),
        ]
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, message in hardcoded_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Skip test files and comments
                    if 'test' in str(file_path).lower() or line.strip().startswith('#'):
                        continue
                    
                    violations.append({
                        'file': str(file_path),
                        'type': 'hardcoded_value',
                        'message': message,
                        'line': i
                    })
                    break
        
        return violations
    
    def check_directory(self, directory: Path, exclude_patterns: List[str] = None) -> Dict[str, any]:
        """Check all Python files in a directory."""
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', 'venv', 'node_modules', '.pytest_cache']
        
        all_violations = []
        
        for py_file in directory.rglob('*.py'):
            # Skip excluded directories
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue
            
            # Skip test files for now (they have different standards)
            if 'test' in str(py_file).lower():
                continue
            
            violations = self.check_file(py_file)
            all_violations.extend(violations)
            self.checked_files += 1
        
        return {
            'total_files': self.checked_files,
            'violations': all_violations,
            'violation_count': len(all_violations)
        }
    
    def print_report(self, results: Dict[str, any]):
        """Print compliance report."""
        print("="*80)
        print("ARCHITECTURE COMPLIANCE REPORT")
        print("="*80)
        print(f"\nFiles Checked: {results['total_files']}")
        print(f"Violations Found: {results['violation_count']}")
        
        if results['violation_count'] == 0:
            print("\n✅ ALL FILES COMPLY WITH ARCHITECTURE STANDARDS")
            return
        
        # Group violations by type
        violations_by_type = {}
        for v in results['violations']:
            v_type = v['type']
            if v_type not in violations_by_type:
                violations_by_type[v_type] = []
            violations_by_type[v_type].append(v)
        
        print("\n" + "="*80)
        print("VIOLATIONS BY TYPE")
        print("="*80)
        
        for v_type, violations in violations_by_type.items():
            print(f"\n{v_type.upper()}: {len(violations)} violations")
            for v in violations[:10]:  # Show first 10
                print(f"  {v['file']}:{v['line']} - {v['message']}")
            if len(violations) > 10:
                print(f"  ... and {len(violations) - 10} more")
        
        print("\n" + "="*80)
        print("RECOMMENDATION: Fix violations before merging to main branch")
        print("="*80)


def main():
    """Run architecture compliance check."""
    root_dir = Path(__file__).parent.parent
    src_dir = root_dir / "src"
    
    if not src_dir.exists():
        print(f"Error: {src_dir} not found")
        sys.exit(1)
    
    checker = ArchitectureComplianceChecker(root_dir)
    results = checker.check_directory(src_dir)
    checker.print_report(results)
    
    # Exit with error code if violations found
    if results['violation_count'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

