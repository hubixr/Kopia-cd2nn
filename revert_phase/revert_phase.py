#!/usr/bin/env python3
"""
Script to reverse scale BMP phase maps.
Takes BMP files from input folder and reverses their scaling so that:
- Maximum values become minimum values
- Minimum values become maximum values
This effectively inverts the phase map while preserving the grayscale range.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import sys

def reverse_scale_image(image_array):
    """
    Reverse scale an image array so max becomes min and min becomes max.
    
    Args:
        image_array: numpy array with image data
        
    Returns:
        numpy array with reversed scaling
    """
    # Get min and max values
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    
    # Reverse the scaling: new_value = max_val - (old_value - min_val) + min_val
    # Simplified: new_value = max_val + min_val - old_value
    reversed_array = max_val + min_val - image_array
    
    return reversed_array

def process_bmp_files(input_dir, output_dir=None, suffix="_reversed"):
    """
    Process all BMP files in input directory and reverse their scaling.
    
    Args:
        input_dir: Path to input directory containing BMP files
        output_dir: Path to output directory (if None, uses input_dir)
        suffix: Suffix to add to output filenames
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
        
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all BMP files
    bmp_files = list(input_path.glob("*.bmp")) + list(input_path.glob("*.BMP"))
    
    if not bmp_files:
        print(f"No BMP files found in '{input_dir}'")
        return
        
    print(f"Found {len(bmp_files)} BMP files in '{input_dir}'")
    
    for bmp_file in bmp_files:
        try:
            # Load image
            image = Image.open(bmp_file).convert('L')  # Convert to grayscale
            image_array = np.array(image, dtype=np.float32)
            
            print(f"Processing {bmp_file.name}...")
            print(f"  Original range: [{np.min(image_array):.1f}, {np.max(image_array):.1f}]")
            
            # Reverse scale the image
            reversed_array = reverse_scale_image(image_array)
            
            print(f"  Reversed range: [{np.min(reversed_array):.1f}, {np.max(reversed_array):.1f}]")
            
            # Convert back to uint8
            reversed_uint8 = reversed_array.astype(np.uint8)
            
            # Create output filename
            output_filename = bmp_file.stem + suffix + bmp_file.suffix
            output_file = output_path / output_filename
            
            # Save reversed image
            reversed_image = Image.fromarray(reversed_uint8, mode='L')
            reversed_image.save(output_file)
            
            print(f"  Saved to: {output_file}")
            
        except Exception as e:
            print(f"Error processing {bmp_file.name}: {e}")
            continue
    
    print(f"\nProcessing complete! Processed {len(bmp_files)} files.")

def main():
    parser = argparse.ArgumentParser(description='Reverse scale BMP phase maps')
    parser.add_argument('input_dir', nargs='?', default='.', help='Input directory containing BMP files (default: current folder)')
    parser.add_argument('-o', '--output_dir', help='Output directory (default: same as input)')
    parser.add_argument('-s', '--suffix', default='_reversed', 
                       help='Suffix for output filenames (default: _reversed)')
    parser.add_argument('--preview', action='store_true',
                       help='Preview what files would be processed without actually processing them')
    
    args = parser.parse_args()
    
    if args.preview:
        input_path = Path(args.input_dir)
        if not input_path.exists():
            print(f"Error: Input directory '{args.input_dir}' does not exist")
            return
            
        bmp_files = list(input_path.glob("*.bmp")) + list(input_path.glob("*.BMP"))
        print(f"Found {len(bmp_files)} BMP files in '{args.input_dir}':")
        for bmp_file in bmp_files:
            print(f"  - {bmp_file.name}")
        
        if args.output_dir:
            print(f"Output directory: {args.output_dir}")
        else:
            print(f"Output directory: {args.input_dir} (same as input)")
            
        print(f"Output suffix: {args.suffix}")
        return
    
    # Process the files
    process_bmp_files(args.input_dir, args.output_dir, args.suffix)

if __name__ == "__main__":
    main()