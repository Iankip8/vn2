#!/usr/bin/env python3
"""
Monitor forecast training progress in real-time.
Watches models/checkpoints/progress.json and displays updates.
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime

progress_file = Path('models/checkpoints/progress.json')

def load_progress():
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return None

def monitor_progress(interval=5):
    """Monitor training progress with periodic updates"""
    print("=" * 70)
    print("FORECAST TRAINING PROGRESS MONITOR")
    print("=" * 70)
    print(f"Watching: {progress_file}")
    print(f"Update interval: {interval} seconds")
    print(f"Press Ctrl+C to exit")
    print("=" * 70)
    
    last_completed_count = 0
    start_time = time.time()
    
    try:
        while True:
            progress = load_progress()
            
            if progress is None:
                print(f"\r⏳ Waiting for training to start...", end='', flush=True)
            else:
                completed = len(progress.get('completed', []))
                failed = len(progress.get('failed', []))
                total = completed + failed  # This is only what we've attempted so far
                
                # Calculate rate
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                
                # Display progress
                status_line = (
                    f"✓ Completed: {completed:4d} | "
                    f"✗ Failed: {failed:3d} | "
                    f"Rate: {rate:5.2f} tasks/sec | "
                    f"Elapsed: {elapsed:6.1f}s"
                )
                
                # Show progress change
                if completed > last_completed_count:
                    delta = completed - last_completed_count
                    print(f"\r{status_line} (+{delta})", flush=True)
                    last_completed_count = completed
                else:
                    print(f"\r{status_line}", end='', flush=True)
                
                # Check if there's been no progress for a while
                if completed == 0 and elapsed > 30:
                    print("\n⚠️  WARNING: No progress after 30 seconds. Training may be stuck.")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Monitoring stopped.")
        if progress:
            print(f"Final stats: {len(progress['completed'])} completed, {len(progress['failed'])} failed")
        print("=" * 70)

if __name__ == '__main__':
    # Check if progress file exists or can be created
    if not progress_file.parent.exists():
        print(f"Error: Directory {progress_file.parent} does not exist.")
        print("Make sure you're running this from the project root.")
        sys.exit(1)
    
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    monitor_progress(interval)
