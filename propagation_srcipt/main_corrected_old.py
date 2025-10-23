#!/usr/bin/env python3
"""
CORRECTED PROPAGATION IMPLEMENTATION
Fixes the mathematical errors in the complex field propagation
"""

from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
PIXEL_SIZE = 9e-4  # [m]
C = 299792458  # [m/s]

# Corrected distance configuration
DISTANCE_BETWEEN_DOE = 0.106        # [m] - spacing between consecutive DOEs
DISTANCE_LAST_DOE_TO_TARGET = 0.095 # [m] - distance from last DOE to target

# Frequency configuration
FREQUENCY_MIN = 160 * 1e9  # [Hz]
FREQUENCY_MAX = 200 * 1e9  # [Hz]
FREQUENCY_STEP = 2 * 1e9   # [Hz]

PADDING_MULTIPLIER = 10
H_BEFORE_PADDING, W_BEFORE_PADDING = 128, 128
H = 128 * (PADDING_MULTIPLIER + 1)
W = 128 * (PADDING_MULTIPLIER + 1)

def propagate_field_corrected(re_u, im_u, wavelength, distance, pixel_size):
    """
    Corrected complex field propagation using Fresnel diffraction
    
    Parameters:
    - re_u, im_u: Real and imaginary parts of input field
    - wavelength: Wavelength in meters
    - distance: Propagation distance in meters 
    - pixel_size: Pixel size in meters
    
    Returns:
    - re_out, im_out: Real and imaginary parts of output field
    """
    
    # Combine into complex field
    u_complex = (re_u + 1j * im_u).astype(np.complex128)
    
    # Get dimensions
    H, W = u_complex.shape
    
    # Calculate spatial frequency grids
    fx = np.fft.fftfreq(W, d=pixel_size)
    fy = np.fft.fftfreq(H, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    
    # Calculate transfer function for Fresnel propagation
    k = 2 * np.pi / wavelength
    h = np.exp(1j * k * distance) * np.exp(-1j * np.pi * wavelength * distance * (FX**2 + FY**2))
    
    # Apply propagation in frequency domain
    u_fft = np.fft.fft2(u_complex)
    u_propagated_fft = u_fft * h
    u_propagated = np.fft.ifft2(u_propagated_fft)
    
    return np.real(u_propagated), np.imag(u_propagated)

def apply_phase_mask(re_u, im_u, phase_mask):
    """
    Apply phase mask to complex field
    
    Parameters:
    - re_u, im_u: Real and imaginary parts of input field
    - phase_mask: Phase mask array (0 to 2π)
    
    Returns:
    - re_out, im_out: Real and imaginary parts after phase modulation
    """
    
    # Apply phase modulation: u_out = u_in * exp(j * phase)
    cos_phase = np.cos(phase_mask)
    sin_phase = np.sin(phase_mask)
    
    re_out = re_u * cos_phase - im_u * sin_phase
    im_out = re_u * sin_phase + im_u * cos_phase
    
    return re_out, im_out

def validate_distances(num_masks, distance_between_doe, distance_last_to_target):
    """Validate the distance configuration"""
    
    print("="*60)
    print("DISTANCE VALIDATION")
    print("="*60)
    
    distances = []
    total_distance = 0
    
    for i in range(num_masks):
        if i == num_masks - 1:
            distance = distance_last_to_target
            stage = f"DOE {i+1} → Target"
        else:
            distance = distance_between_doe
            stage = f"DOE {i+1} → DOE {i+2}"
        
        distances.append(distance)
        total_distance += distance
        print(f"  {stage:<15} Distance: {distance*1000:6.1f} mm")
    
    print(f"  Total optical path: {total_distance*1000:.1f} mm")
    
    # Check physical consistency
    if distance_last_to_target <= 0:
        print("  ⚠️  ERROR: Distance to target must be positive!")
        return False
    
    if distance_between_doe <= 0:
        print("  ⚠️  ERROR: Distance between DOEs must be positive!")
        return False
    
    print("  ✓ Distance configuration is physically valid")
    return True, distances

def corrected_propagation_demo():
    """Demonstrate the corrected propagation"""
    
    print("="*80)
    print("CORRECTED PROPAGATION DEMONSTRATION")
    print("="*80)
    
    # Load phase masks (assuming they exist)
    phase_masks_dir = Path('data/phase_masks')
    if not phase_masks_dir.exists():
        print(f"Creating demo phase masks...")
        phase_masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple demo phase mask
        demo_phase = np.random.uniform(0, 2*np.pi, (128, 128))
        demo_img = Image.fromarray((demo_phase / (2*np.pi) * 255).astype(np.uint8))
        demo_img.save(phase_masks_dir / 'demo_mask.bmp')
    
    phase_mask_files = sorted(phase_masks_dir.glob('*.bmp'))
    if not phase_mask_files:
        print("No phase mask files found!")
        return
    
    # Load phase masks
    phase_masks = []
    for bmp_file in phase_mask_files:
        img = Image.open(bmp_file).convert('L')
        img_array = np.array(img, dtype=np.float32)
        img_norm = img_array / 255.0 * 2 * np.pi
        phase_masks.append(img_norm)
        print(f'Loaded {bmp_file.name}')
    
    num_masks = len(phase_masks)
    
    # Validate distances
    is_valid, distances = validate_distances(num_masks, DISTANCE_BETWEEN_DOE, DISTANCE_LAST_DOE_TO_TARGET)
    if not is_valid:
        return
    
    # Create input field (Gaussian beam)
    x = np.linspace(-H_BEFORE_PADDING//2, H_BEFORE_PADDING//2, H_BEFORE_PADDING) * PIXEL_SIZE
    y = np.linspace(-W_BEFORE_PADDING//2, W_BEFORE_PADDING//2, W_BEFORE_PADDING) * PIXEL_SIZE
    X, Y = np.meshgrid(x, y)
    
    # Gaussian input beam
    beam_waist = 20e-3  # 20 mm beam waist
    input_field = np.exp(-(X**2 + Y**2) / beam_waist**2)
    
    # Test at 180 GHz
    frequency = 180e9
    wavelength = C / frequency
    
    print(f"\nTesting at {frequency/1e9:.0f} GHz (λ = {wavelength*1e3:.2f} mm)")
    
    # Initialize field
    re_u = input_field.copy()
    im_u = np.zeros_like(input_field)
    
    # Add padding
    pad_size = int(H / (PADDING_MULTIPLIER + 1) * PADDING_MULTIPLIER / 2)
    
    # Store results
    results = []
    
    for i in range(num_masks):
        print(f"\nProcessing mask {i+1}/{num_masks}")
        
        # Pad field
        re_u_padded = np.pad(re_u, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
        im_u_padded = np.pad(im_u, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
        
        # Apply phase mask
        phase_mask_padded = np.pad(phase_masks[i], ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
        re_u_padded, im_u_padded = apply_phase_mask(re_u_padded, im_u_padded, phase_mask_padded)
        
        # Propagate
        distance = distances[i]
        re_u_padded, im_u_padded = propagate_field_corrected(re_u_padded, im_u_padded, wavelength, distance, PIXEL_SIZE)
        
        # Crop back to original size
        re_u = re_u_padded[pad_size:pad_size+H_BEFORE_PADDING, pad_size:pad_size+W_BEFORE_PADDING]
        im_u = im_u_padded[pad_size:pad_size+H_BEFORE_PADDING, pad_size:pad_size+W_BEFORE_PADDING]
        
        # Calculate intensity
        intensity = re_u**2 + im_u**2
        results.append(intensity)
        
        print(f"  Max intensity: {np.max(intensity):.6f}")
        print(f"  Total power: {np.sum(intensity):.6f}")
    
    # Save results
    results_dir = Path('results_corrected')
    results_dir.mkdir(exist_ok=True)
    
    # Plot comparison
    fig, axes = plt.subplots(1, num_masks + 1, figsize=(5*(num_masks+1), 5))
    if num_masks == 0:
        axes = [axes]
    
    # Input
    axes[0].imshow(input_field**2, cmap='hot')
    axes[0].set_title('Input Intensity')
    axes[0].axis('off')
    
    # Results after each mask
    for i, intensity in enumerate(results):
        axes[i+1].imshow(intensity, cmap='hot')
        axes[i+1].set_title(f'After Mask {i+1}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'corrected_propagation_results.png', dpi=300)
    plt.close()
    
    print(f"\nResults saved to {results_dir}/")
    
    return results

if __name__ == "__main__":
    corrected_propagation_demo()
    
    print("\n" + "="*80)
    print("KEY CORRECTIONS IMPLEMENTED:")
    print("="*80)
    print("1. ✓ Fixed complex field multiplication (mathematically correct)")
    print("2. ✓ Clarified distance definitions (DOE-to-DOE vs DOE-to-target)")
    print("3. ✓ Added distance validation checks")
    print("4. ✓ Simplified complex propagation using direct complex operations")
    print("5. ✓ Added proper type handling (complex128)")
    print("\nThe corrected version should give accurate propagation results!")
    print("="*80)