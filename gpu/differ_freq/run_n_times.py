#!/usr/bin/env python3
"""
Script to run train_cd2nn_model.py multiple times.
Useful for running multiple training experiments with different random seeds.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def run_training(run_number, total_runs):
    """Run a single training session."""
    print(f"\n{'='*60}")
    print(f"STARTING RUN {run_number}/{total_runs}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the training script
        result = subprocess.run([
            sys.executable, 'train_cd2nn_model.py'
        ], 
        cwd=Path(__file__).parent,
        capture_output=False,  # Show output in real-time
        text=True,
        check=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"COMPLETED RUN {run_number}/{total_runs}")
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"FAILED RUN {run_number}/{total_runs}")
        print(f"Error: {e}")
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"{'='*60}")
        
        return False, duration
    
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print(f"INTERRUPTED RUN {run_number}/{total_runs}")
        print(f"{'='*60}")
        raise

def main():
    """Main function to run multiple training sessions."""
    
    # Configuration
    NUM_RUNS = 10  # Change this to run more times
    
    print(f"Starting {NUM_RUNS} training runs...")
    print(f"Script: train_cd2nn_model.py")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    successful_runs = 0
    failed_runs = 0
    total_time = 0
    
    overall_start_time = time.time()
    
    try:
        for i in range(1, NUM_RUNS + 1):
            success, duration = run_training(i, NUM_RUNS)
            results.append((i, success, duration))
            total_time += duration
            
            if success:
                successful_runs += 1
            else:
                failed_runs += 1
                
            # Small break between runs (optional)
            if i < NUM_RUNS:
                print(f"\nWaiting 10 seconds before next run...")
                time.sleep(10)
    
    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted by user!")
        
    finally:
        # Print summary
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        
        print(f"\n\n{'='*80}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"Total runs attempted: {len(results)}")
        print(f"Successful runs: {successful_runs}")
        print(f"Failed runs: {failed_runs}")
        print(f"Total time: {overall_duration/3600:.1f} hours ({overall_duration/60:.1f} minutes)")
        print(f"Average time per run: {(total_time/len(results))/60:.1f} minutes")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nDetailed Results:")
        for run_num, success, duration in results:
            status = "SUCCESS" if success else "FAILED"
            print(f"  Run {run_num}: {status} - {duration/60:.1f} minutes")
        
        print(f"{'='*80}")

if __name__ == "__main__":
    main()
