#!/usr/bin/env python3
"""
Titan V8 Project Setup Script
Multi-Agent Volatility Prediction Research Project

This script creates the project directory structure in the CURRENT WORKING DIRECTORY.
"""

from pathlib import Path


def create_project_structure():
    """Create the Titan V8 project directory structure in the current directory."""
    
    # Use current working directory as base
    base_dir = Path.cwd()
    
    # Define all directories to create (relative to cwd)
    directories = [
        # Configuration
        base_dir / "conf" / "base",
        
        # Data directories
        base_dir / "data" / "raw",
        base_dir / "data" / "processed",
        base_dir / "data" / "external",
        
        # MLflow storage
        base_dir / "mlruns",
        
        # Notebooks
        base_dir / "notebooks",
        
        # Source code
        base_dir / "src" / "agents",
        base_dir / "src" / "pipeline",
        base_dir / "src" / "utils",
        
        # Tests
        base_dir / "tests",
    ]
    
    # Create directories
    print(f"Creating Titan V8 project structure in: {base_dir}")
    print("-" * 50)
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        rel_path = directory.relative_to(base_dir)
        print(f"  ✓ Created: ./{rel_path}")
    
    # Create __init__.py files for Python packages
    init_files = [
        base_dir / "src" / "__init__.py",
        base_dir / "src" / "agents" / "__init__.py",
        base_dir / "src" / "pipeline" / "__init__.py",
        base_dir / "src" / "utils" / "__init__.py",
        base_dir / "tests" / "__init__.py",
    ]
    
    print("\nCreating Python package files...")
    print("-" * 50)
    
    for init_file in init_files:
        init_file.touch(exist_ok=True)
        rel_path = init_file.relative_to(base_dir)
        print(f"  ✓ Created: ./{rel_path}")
    
    # Create .gitignore
    gitignore_content = """# Titan V8 .gitignore

# Data directories (large files)
data/

# MLflow local storage
mlruns/

# Environment variables
.env

# Python cache
__pycache__/
*.py[cod]
*$py.class

# Parquet files
*.parquet

# Jupyter checkpoints
.ipynb_checkpoints/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db
"""
    
    gitignore_path = base_dir / ".gitignore"
    gitignore_path.write_text(gitignore_content)
    
    print("\nCreating .gitignore...")
    print("-" * 50)
    print(f"  ✓ Created: ./.gitignore")
    
    print("\n" + "=" * 50)
    print("Titan V8 project structure created successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("  1. Create a .env file with your POLYGON_API_KEY")
    print("  2. Install dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    create_project_structure()
