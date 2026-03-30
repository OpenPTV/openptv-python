"""Test script specifically for ext_sequence_splitter plugin"""

import sys
import subprocess
import shutil
from pathlib import Path


def _copy_splitter_case(tmp_path):
    source = Path(__file__).parent.parent / "testing_folder" / "test_splitter"
    destination = tmp_path / "test_splitter"
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def test_ext_sequence_splitter(tmp_path):
    """Test the ext_sequence_splitter plugin using batch command (proven working approach)"""
    test_path = _copy_splitter_case(tmp_path)
    
    if not test_path.exists():
        print(f"❌ Test data not found: {test_path}")
        return False
    
    print(f"🔍 Testing ext_sequence_splitter with data from: {test_path}")
    
    # Use the proven working batch script approach
    script_path = Path(__file__).parent.parent / "pyptv" / "pyptv_batch_plugins.py"
    yaml_file = test_path / "parameters_Run1.yaml"
    if not script_path.exists():
        print(f"❌ Batch script not found: {script_path}")
        return False
    if not yaml_file.exists():
        print(f"❌ YAML file not found: {yaml_file}")
        return False
    # Run just 2 frames for quick testing
    cmd = [
        sys.executable, 
        str(script_path), 
        str(yaml_file), 
        "1000001", 
        "1000002",  # Just 2 frames for quick test
        "--sequence", "ext_sequence_splitter"
    ]
    
    print(f"🚀 Running batch command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=60
        )
        
        # Check that it completed successfully
        if result.returncode != 0:
            print(f"❌ Process failed with return code {result.returncode}")
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            return False
        
        # Check for expected success indicators
        success_indicators = [
            "Processing frame 1000001",
            "Processing frame 1000002", 
            "correspondences",
            "Sequence completed successfully"
        ]
        
        missing_indicators = []
        for indicator in success_indicators:
            if indicator not in result.stdout:
                missing_indicators.append(indicator)
        
        if missing_indicators:
            print(f"❌ Missing expected output: {missing_indicators}")
            print("Full output:")
            print(result.stdout)
            return False
        
        print("✅ ext_sequence_splitter test completed successfully")
        return True
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False


def test_batch_command(tmp_path):
    """Test using the batch command line interface"""
    
    # Fix paths for running from tests/ directory
    script_path = Path(__file__).parent.parent / "pyptv" / "pyptv_batch_plugins.py"
    test_exp_path = _copy_splitter_case(tmp_path)
    
    if not script_path.exists():
        print(f"❌ Batch script not found: {script_path}")
        return False
        
    if not test_exp_path.exists():
        print(f"❌ Test experiment not found: {test_exp_path}")
        return False
    
    yaml_file = test_exp_path / "parameters_Run1.yaml"
    if not yaml_file.exists():
        print(f"❌ YAML file not found: {yaml_file}")
        return False
    # Run just 2 frames for quick testing
    cmd = [
        sys.executable, 
        str(script_path), 
        str(yaml_file), 
        "1000001", 
        "1000002",  # Just 2 frames for quick test
        "--sequence", "ext_sequence_splitter"
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        if result.stdout:
            print("📄 STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("📄 STDERR:")  
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Batch command completed successfully")
            return True
        else:
            print(f"❌ Batch command failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing ext_sequence_splitter plugin")
    print("="*50)
    
    print("\n1️⃣ Testing ext_sequence_splitter via batch command...")
    test1_success = test_ext_sequence_splitter()
    
    print("\n2️⃣ Testing batch command interface (alternative approach)...")
    test2_success = test_batch_command()
    
    print("\n" + "="*50)
    if test1_success and test2_success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        if not test1_success:
            print("   - Primary ext_sequence_splitter test failed")
        if not test2_success:
            print("   - Secondary batch command test failed") 
        sys.exit(1)
