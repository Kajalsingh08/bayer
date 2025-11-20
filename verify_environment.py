#!/usr/bin/env python3
"""
Environment Verification for SageMaker Studio
Validates Python 3.12 setup for schema-aware SLM training
"""

import sys
import importlib
from typing import Dict, List, Tuple

def check_python_version() -> bool:
    """Verify Python 3.12"""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    print(f"Python Path: {sys.executable}\n")
    
    if version.major == 3 and version.minor == 12:
        print("✅ Python 3.12 confirmed\n")
        return True
    else:
        print(f"⚠️  Expected Python 3.12, got {version.major}.{version.minor}\n")
        return False

def check_imports() -> Dict[str, str]:
    """Check all required packages"""
    
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'accelerate': 'Accelerate',
        'peft': 'PEFT',
        'sentencepiece': 'SentencePiece',
        'tensorboard': 'TensorBoard',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
    }
    
    print("=" * 60)
    print("Package Verification")
    print("=" * 60 + "\n")
    
    results = {}
    for module, name in packages.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {name:20s} {version}")
            results[name] = version
        except ImportError as e:
            print(f"❌ {name:20s} NOT FOUND")
            results[name] = None
    
    return results

def check_gpu() -> Dict[str, any]:
    """Check GPU availability and specs"""
    
    print("\n" + "=" * 60)
    print("GPU Verification")
    print("=" * 60 + "\n")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available - running on CPU")
            print("   Note: Training will be very slow on CPU")
            return {'available': False}
        
        gpu_info = {
            'available': True,
            'count': torch.cuda.device_count(),
            'name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9
        }
        
        print(f"✅ CUDA Available: Yes")
        print(f"   CUDA Version: {gpu_info['cuda_version']}")
        print(f"   GPU Count: {gpu_info['count']}")
        print(f"   GPU Name: {gpu_info['name']}")
        print(f"   GPU Memory: {gpu_info['memory_gb']:.2f} GB")
        
        # Memory test
        print("\n   Testing GPU memory...")
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("   ✅ GPU memory test passed")
        except Exception as e:
            print(f"   ❌ GPU memory test failed: {e}")
        
        return gpu_info
        
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return {'available': False, 'error': str(e)}

def estimate_capacity() -> None:
    """Estimate training capacity"""
    
    print("\n" + "=" * 60)
    print("Training Capacity Estimate")
    print("=" * 60 + "\n")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("CPU-only mode:")
            print("  - Suitable for: Corpus generation only")
            print("  - NOT suitable for: Model training")
            print("  - Recommendation: Use GPU instance")
            return
        
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"Available GPU Memory: {memory_gb:.2f} GB\n")
        
        if memory_gb >= 80:
            print("✅ Excellent (80+ GB)")
            print("   - Can train: 70B models with full precision")
            print("   - Batch size: 4-8")
            print("   - Gradient accumulation: 2-4")
        elif memory_gb >= 40:
            print("✅ Very Good (40-80 GB)")
            print("   - Can train: 30B models with full precision")
            print("   - Can train: 70B models with quantization")
            print("   - Batch size: 2-4")
            print("   - Gradient accumulation: 4-8")
        elif memory_gb >= 24:
            print("✅ Good (24-40 GB)")
            print("   - Can train: 7B-13B models with full precision")
            print("   - Can train: 30B models with quantization")
            print("   - Batch size: 1-2")
            print("   - Gradient accumulation: 8-16")
            print("   - Recommended: Mistral-7B, Llama2-7B")
        elif memory_gb >= 16:
            print("⚠️  Adequate (16-24 GB)")
            print("   - Can train: 7B models with quantization (QLoRA)")
            print("   - Batch size: 1")
            print("   - Gradient accumulation: 16-32")
            print("   - Recommendation: Use gradient checkpointing")
        else:
            print("❌ Insufficient (<16 GB)")
            print("   - Cannot train models efficiently")
            print("   - Recommendation: Upgrade to ml.g5.2xlarge or larger")
    
    except Exception as e:
        print(f"Error estimating capacity: {e}")

def main():
    """Run all verification checks"""
    
    print("\n" + "=" * 60)
    print("SageMaker Studio Environment Verification")
    print("Schema-Aware SLM Training Setup")
    print("=" * 60 + "\n")
    
    # Python version
    python_ok = check_python_version()
    
    # Package imports
    packages = check_imports()
    packages_ok = all(v is not None for v in packages.values())
    
    # GPU check
    gpu_info = check_gpu()
    
    # Capacity estimate
    estimate_capacity()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60 + "\n")
    
    if python_ok and packages_ok and gpu_info.get('available'):
        print("✅ Environment is ready for SLM training!")
        print("   - Python 3.12: ✅")
        print("   - All packages: ✅")
        print("   - GPU available: ✅")
        print("\n   Next steps:")
        print("   1. Generate corpus: python scripts/generate_corpus.py")
        print("   2. Start training: python scripts/pretrain_model.py")
    elif python_ok and packages_ok:
        print("⚠️  Environment partially ready")
        print("   - Python 3.12: ✅")
        print("   - All packages: ✅")
        print("   - GPU available: ❌")
        print("\n   Actions:")
        print("   1. Can generate corpus on CPU")
        print("   2. Switch to GPU instance for training")
    else:
        print("❌ Environment setup incomplete")
        if not python_ok:
            print("   - Python 3.12: ❌ (wrong version)")
        if not packages_ok:
            print("   - Packages: ❌ (some missing)")
        print("\n   Actions:")
        print("   1. Fix Python version")
        print("   2. Install missing packages: pip install -r requirements.txt")
    
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()