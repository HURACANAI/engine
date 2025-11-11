#!/usr/bin/env python3
"""
Script to automatically replace print statements with structured logging.

Usage:
    python scripts/fix_print_statements.py [file_path]
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def has_structlog_import(content: str) -> bool:
    """Check if file already imports structlog."""
    return "import structlog" in content or "from structlog" in content


def has_logger_declaration(content: str) -> bool:
    """Check if file already has logger declaration."""
    return "logger = structlog.get_logger" in content


def add_structlog_import(lines: List[str]) -> Tuple[List[str], bool]:
    """Add structlog import if needed."""
    content = "\n".join(lines)

    if has_structlog_import(content):
        return lines, False

    # Find the last import line
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            last_import_idx = i

    # Insert after last import
    if last_import_idx > 0:
        lines.insert(last_import_idx + 1, "import structlog")
        lines.insert(last_import_idx + 2, "")
        return lines, True

    # If no imports, add at top after docstring
    insert_idx = 0
    in_docstring = False
    for i, line in enumerate(lines):
        if '"""' in line or "'''" in line:
            if not in_docstring:
                in_docstring = True
            else:
                insert_idx = i + 1
                break

    lines.insert(insert_idx, "import structlog")
    lines.insert(insert_idx + 1, "")
    return lines, True


def add_logger_declaration(lines: List[str]) -> Tuple[List[str], bool]:
    """Add logger declaration if needed."""
    content = "\n".join(lines)

    if has_logger_declaration(content):
        return lines, False

    # Find position after imports
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            last_import_idx = i

    # Insert logger after imports
    if last_import_idx > 0:
        lines.insert(last_import_idx + 1, "")
        lines.insert(last_import_idx + 2, 'logger = structlog.get_logger(__name__)')
        lines.insert(last_import_idx + 3, "")
        return lines, True

    return lines, False


def convert_print_to_logger(line: str, indent: str) -> str:
    """Convert a print statement to logger.info()."""
    # Extract the content inside print()
    match = re.match(r'(\s*)print\((.*)\)\s*$', line)
    if not match:
        return line

    indent = match.group(1)
    content = match.group(2).strip()

    # Simple string
    if content.startswith('"') or content.startswith("'"):
        return f'{indent}logger.info({content})'

    # f-string - try to convert to structured logging
    if content.startswith('f"') or content.startswith("f'"):
        # For now, just replace with logger.info
        return f'{indent}logger.info({content})'

    # Complex expression
    return f'{indent}logger.info("output", message={content})'


def fix_file(file_path: Path) -> bool:
    """Fix print statements in a file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        original_lines = lines.copy()

        # Remove trailing whitespace and newlines for processing
        lines = [line.rstrip() for line in lines]

        # Add imports if needed
        lines, added_import = add_structlog_import(lines)
        lines, added_logger = add_logger_declaration(lines)

        # Convert print statements
        modified = False
        for i, line in enumerate(lines):
            if re.match(r'^\s*print\(', line):
                indent = re.match(r'^(\s*)', line).group(1)
                lines[i] = convert_print_to_logger(line, indent)
                modified = True

        if added_import or added_logger or modified:
            # Write back with newlines
            with open(file_path, 'w') as f:
                f.write('\n'.join(lines) + '\n')

            print(f"✓ Fixed: {file_path}")
            return True
        else:
            print(f"○ No changes: {file_path}")
            return False

    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Fix specific file
        file_path = Path(sys.argv[1])
        fix_file(file_path)
    else:
        # Fix all Python files in src/
        src_dir = Path(__file__).parent.parent / "src"
        python_files = list(src_dir.rglob("*.py"))

        fixed_count = 0
        for file_path in python_files:
            if fix_file(file_path):
                fixed_count += 1

        print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
