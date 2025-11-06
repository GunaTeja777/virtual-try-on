"""
Monitor training progress
"""
import time
import os
from pathlib import Path

def monitor_training():
    checkpoint_dir = Path('checkpoints/segmentation')
    
    print("="*60)
    print("🔍 Training Monitor - Segmentation Model")
    print("="*60)
    print("\nPress Ctrl+C to stop monitoring\n")
    
    last_mtime = 0
    
    try:
        while True:
            # Check if checkpoint directory exists
            if checkpoint_dir.exists():
                # List all checkpoint files
                checkpoints = sorted(checkpoint_dir.glob('*.pth'))
                
                if checkpoints:
                    latest_checkpoint = checkpoints[-1]
                    mtime = latest_checkpoint.stat().st_mtime
                    
                    if mtime != last_mtime:
                        last_mtime = mtime
                        mod_time = time.strftime('%H:%M:%S', time.localtime(mtime))
                        size_mb = latest_checkpoint.stat().st_size / (1024 * 1024)
                        
                        print(f"[{time.strftime('%H:%M:%S')}] Latest checkpoint: {latest_checkpoint.name}")
                        print(f"              Modified: {mod_time}, Size: {size_mb:.1f} MB")
                        print(f"              Total checkpoints: {len(checkpoints)}")
                        print("-" * 60)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No checkpoints yet...")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Waiting for training to start...")
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")
        print("="*60)

if __name__ == '__main__':
    monitor_training()
