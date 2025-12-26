#!/usr/bin/env python3
"""
Clean up WorldModel directory for public sharing
"""
import os
import shutil
import glob

def cleanup_project():
    """Clean up temporary files and organize project structure"""
    
    print("🧹 Cleaning up WorldModel project for sharing...")
    
    # Files and directories to remove
    cleanup_items = [
        # Development/debugging files
        "debug_*.py",
        "test_*.py", 
        "download_model.py",
        
        # Temporary directories
        "debug_training_test/",
        "phi4_test/", 
        "phi4_worldmodel_finetuned/",
        "phi4_worldmodel_finetuned_rocm/",
        "qwen2.5_worldmodel_finetuned/",
        "quick_test/",
        "training_output/",
        "sft_checkpoints/",
        
        # Cache and temporary files
        ".pytest_cache/",
        "pytest.ini",
        "logs/",
        "training.log",
        "venv/",
        "~/",
        
        # Test directories  
        "tests/",
    ]
    
    # Files to keep (core functionality)
    keep_files = [
        "main.py",
        "config.json", 
        "requirements.txt",
        "src/",
        "data/",
        "spec/",
        
        # Training scripts
        "build_training_dataset.py",
        "combine_datasets.py", 
        "combine_all_examples.py",
        "generate_science_examples.py",
        "convert_to_llama_cpp.py",
        "train_rocm.sh",
        "run_training_with_rocm.sh",
        "demo_worldmodel.py",
        
        # Documentation will be added
        "README.md",
        "INSTALL.md", 
        "TRAINING.md",
    ]
    
    cleaned_count = 0
    
    for pattern in cleanup_items:
        if pattern.endswith('/'):
            # Directory
            dir_name = pattern[:-1]
            if os.path.exists(dir_name):
                print(f"   🗑️  Removing directory: {dir_name}")
                shutil.rmtree(dir_name)
                cleaned_count += 1
        else:
            # File pattern
            files = glob.glob(pattern)
            for file in files:
                if os.path.exists(file):
                    print(f"   🗑️  Removing file: {file}")
                    os.remove(file)
                    cleaned_count += 1
    
    # Clean up __pycache__ directories
    for root, dirs, files in os.walk('.'):
        for dir in dirs:
            if dir == '__pycache__':
                pycache_path = os.path.join(root, dir)
                print(f"   🗑️  Removing __pycache__: {pycache_path}")
                shutil.rmtree(pycache_path)
                cleaned_count += 1
    
    print(f"\n✅ Cleanup complete!")
    print(f"   🗑️  Removed {cleaned_count} items")
    
    # Show final structure
    print(f"\n📁 Final project structure:")
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in sorted(files):
            if not file.startswith('.'):
                print(f"{subindent}{file}")

if __name__ == "__main__":
    cleanup_project()