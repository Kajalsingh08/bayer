#!/usr/bin/env python3
"""
Auto-generate requirements.txt with latest compatible versions from PyPI
"""

import subprocess
import sys
from datetime import datetime
from typing import Dict, Optional

def get_latest_version(package: str) -> Optional[str]:
    """Get latest version of a package from PyPI"""
    try:
        result = subprocess.run(
            ['pip', 'index', 'versions', package],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Parse output: "Available versions: 2.1.0, 2.0.1, ..."
        for line in result.stdout.split('\n'):
            if 'Available versions:' in line:
                versions = line.split(':')[1].strip()
                latest = versions.split(',')[0].strip()
                return latest
        
        return None
    except Exception as e:
        print(f"Warning: Could not fetch version for {package}: {e}")
        return None

def get_next_major_version(version: str) -> str:
    """Get next major version for upper bound"""
    try:
        parts = version.split('.')
        major = int(parts[0])
        return f"{major + 1}.0.0"
    except:
        return "999.0.0"

def generate_requirements():
    """Generate requirements.txt with latest versions"""
    
    # Package categories with descriptions
    packages = {
        "Core ML Framework": {
            "note": "Install separately: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "packages": {
                "torch": ">=2.1.0,<3.0.0",
                "torchvision": ">=0.16.0,<1.0.0",
                "torchaudio": ">=2.1.0,<3.0.0",
            }
        },
        "Transformers & Training": {
            "packages": {
                "transformers": None,
                "datasets": None,
                "accelerate": None,
                "peft": None,
            }
        },
        "Tokenization & Serialization": {
            "packages": {
                "sentencepiece": None,
                "protobuf": None,
            }
        },
        "Training Monitoring": {
            "packages": {
                "tensorboard": None,
            }
        },
        "Data Processing": {
            "packages": {
                "pandas": None,
                "numpy": None,
                "tqdm": None,
            }
        },
        "Visualization": {
            "packages": {
                "matplotlib": None,
                "seaborn": None,
            }
        },
        "Development Tools": {
            "packages": {
                "jupyter": None,
                "ipykernel": None,
                "black": None,
                "flake8": None,
                "isort": None,
            }
        },
        "Additional Utilities": {
            "packages": {
                "rich": None,
                "pyyaml": None,
            }
        },
        "Quantization (Optional)": {
            "packages": {
                "bitsandbytes": None,
            },
            "condition": "; platform_system != \"Darwin\""
        }
    }
    
    output_lines = [
        f"# Auto-generated requirements.txt - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "# Latest compatible versions for Schema-Aware SLM Training",
        "# Python 3.12+ required",
        "",
    ]
    
    print("Fetching latest versions from PyPI...")
    print("This may take a minute...\n")
    
    for category, config in packages.items():
        print(f"Processing: {category}")
        
        # Category header
        output_lines.append(f"# {category}")
        
        # Add note if present
        if "note" in config:
            output_lines.append(f"# {config['note']}")
        
        # Get packages
        pkg_dict = config.get("packages", {})
        condition = config.get("condition", "")
        
        for package, default_version in pkg_dict.items():
            if default_version:
                # Use provided version
                output_lines.append(f"{package}{default_version}")
                print(f"  ✓ {package}: {default_version} (pinned)")
            else:
                # Fetch latest
                latest = get_latest_version(package)
                if latest:
                    next_major = get_next_major_version(latest)
                    version_spec = f">={latest},<{next_major}"
                    line = f"{package}{version_spec}{condition}"
                    output_lines.append(line)
                    print(f"  ✓ {package}: {version_spec}")
                else:
                    # Fallback
                    line = f"{package}{condition}"
                    output_lines.append(line)
                    print(f"  ⚠ {package}: Could not fetch version, using unversioned")
        
        output_lines.append("")  # Blank line between categories
        print()
    
    # Write to file
    output_file = "requirements.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"✅ Generated {output_file}")
    print(f"   Lines: {len(output_lines)}")
    print("\nTo install:")
    print("  1. Install PyTorch: make install-pytorch")
    print("  2. Install others:  pip install -r requirements.txt")

if __name__ == "__main__":
    try:
        generate_requirements()
    except KeyboardInterrupt:
        print("\n\nAborted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)