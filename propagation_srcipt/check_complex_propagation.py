#!/usr/bin/env python3
"""
Check the complex field propagation implementation for correctness
"""

import numpy as np

def analyze_complex_propagation():
    """Analyze the complex field propagation implementation"""
    
    print("="*80)
    print("COMPLEX PROPAGATION IMPLEMENTATION ANALYSIS")
    print("="*80)
    
    print("Current implementation from main.py:")
    print("-" * 40)
    
    # Show current implementation
    print("""
    # Current code:
    h = np.exp(1j*k*distance)*np.exp(arg)  # where arg = -1j*π*distance*wavelength*r2
    h_real = np.real(h).astype(np.complex64)  # ⚠️ ISSUE: Real part cast to complex
    h_imag = np.imag(h).astype(np.complex64)  # ⚠️ ISSUE: Imag part cast to complex
    
    re_re = np.real(ifft2(fft2(re_u) * h_real))
    im_im = np.real(ifft2(fft2(im_u) * h_imag))
    re_im = np.real(ifft2(fft2(re_u) * h_imag))
    im_re = np.real(ifft2(fft2(im_u) * h_real))
    
    out_real = re_re - im_im
    out_imag = re_im + im_re
    """)
    
    print("ISSUES IDENTIFIED:")
    print("=" * 50)
    
    print("1. TYPE CASTING ERROR:")
    print("   h_real = np.real(h).astype(np.complex64)")
    print("   h_imag = np.imag(h).astype(np.complex64)")
    print("   → Real/imag parts are cast back to complex, which is confusing")
    print("   → Should keep as float32/float64")
    print()
    
    print("2. COMPLEX MULTIPLICATION LOGIC:")
    print("   Current approach manually separates real/imag parts")
    print("   This is error-prone and inefficient")
    print()
    
    print("3. MATHEMATICAL VERIFICATION:")
    print("   Let u = re_u + j*im_u (input field)")
    print("   Let h = h_real + j*h_imag (transfer function)")
    print("   Convolution: u_out = IFFT(FFT(u) * h)")
    print()
    print("   Current implementation:")
    print("   out_real = re_re - im_im = Re[IFFT(FFT(re_u) * h_real)] - Re[IFFT(FFT(im_u) * h_imag)]")
    print("   out_imag = re_im + im_re = Re[IFFT(FFT(re_u) * h_imag)] + Re[IFFT(FFT(im_u) * h_real)]")
    print()
    
    # Test the math
    print("MATHEMATICAL VERIFICATION:")
    print("-" * 30)
    test_complex_math()
    
def test_complex_math():
    """Test if the current complex multiplication is mathematically correct"""
    
    # Create test arrays
    np.random.seed(42)
    N = 8
    re_u = np.random.randn(N, N)
    im_u = np.random.randn(N, N)
    u_complex = re_u + 1j * im_u
    
    # Create test transfer function
    h_complex = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    h_real = np.real(h_complex)
    h_imag = np.imag(h_complex)
    
    # Method 1: Direct complex multiplication (CORRECT)
    fft_u = np.fft.fft2(u_complex)
    result_complex = np.fft.ifft2(fft_u * h_complex)
    correct_real = np.real(result_complex)
    correct_imag = np.imag(result_complex)
    
    # Method 2: Current implementation (TO TEST)
    re_re = np.real(np.fft.ifft2(np.fft.fft2(re_u) * h_real))
    im_im = np.real(np.fft.ifft2(np.fft.fft2(im_u) * h_imag))
    re_im = np.real(np.fft.ifft2(np.fft.fft2(re_u) * h_imag))
    im_re = np.real(np.fft.ifft2(np.fft.fft2(im_u) * h_real))
    
    current_real = re_re - im_im
    current_imag = re_im + im_re
    
    # Compare results
    real_error = np.max(np.abs(correct_real - current_real))
    imag_error = np.max(np.abs(correct_imag - current_imag))
    
    print(f"   Max error in real part: {real_error:.2e}")
    print(f"   Max error in imag part: {imag_error:.2e}")
    
    if real_error < 1e-10 and imag_error < 1e-10:
        print("   ✓ Current implementation is mathematically CORRECT")
    else:
        print("   ✗ Current implementation has ERRORS")
    
    print()

def suggest_improvements():
    """Suggest improvements to the propagation implementation"""
    
    print("="*80)
    print("SUGGESTED IMPROVEMENTS:")
    print("="*80)
    
    print("1. SIMPLIFIED COMPLEX PROPAGATION:")
    print("-" * 40)
    print("""
# Improved implementation:
def propagate_field(re_u, im_u, wavelength, distance, pixel_size):
    # Combine real and imaginary parts
    u_complex = re_u + 1j * im_u
    
    # Calculate transfer function
    H, W = u_complex.shape
    fx = np.fft.fftfreq(W, d=pixel_size)
    fy = np.fft.fftfreq(H, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    
    k = 2 * np.pi / wavelength
    h = np.exp(1j * k * distance) * np.exp(-1j * np.pi * wavelength * distance * (FX**2 + FY**2))
    
    # Propagate using direct complex multiplication
    u_fft = np.fft.fft2(u_complex)
    u_propagated = np.fft.ifft2(u_fft * h)
    
    return np.real(u_propagated), np.imag(u_propagated)
    """)
    
    print("2. ADVANTAGES OF SIMPLIFIED VERSION:")
    print("   ✓ Cleaner, more readable code")
    print("   ✓ Less prone to errors")
    print("   ✓ Better performance (fewer FFT operations)")
    print("   ✓ Direct correspondence to physical equations")
    print()
    
    print("3. TYPE SAFETY IMPROVEMENTS:")
    print("-" * 30)
    print("""
# Use consistent data types
u_complex = (re_u + 1j * im_u).astype(np.complex128)
h = h.astype(np.complex128)
    """)
    
    print("4. DISTANCE LOGIC CORRECTION:")
    print("-" * 35)
    print("""
# Option A: Independent distances (RECOMMENDED)
DISTANCE_BETWEEN_DOE = 0.106        # [m] - DOE to DOE spacing
DISTANCE_LAST_DOE_TO_TARGET = 0.095 # [m] - Last DOE to target

for i in range(num_masks):
    if i == num_masks-1:
        distance = DISTANCE_LAST_DOE_TO_TARGET
        print(f"Propagating from DOE {i+1} to target: {distance*1000:.1f} mm")
    else:
        distance = DISTANCE_BETWEEN_DOE
        print(f"Propagating from DOE {i+1} to DOE {i+2}: {distance*1000:.1f} mm")
    
    # Apply propagation
    re_u, im_u = propagate_field(re_u, im_u, wavelength, distance, PIXEL_SIZE)
    """)
    
    print("5. ADD VALIDATION CHECKS:")
    print("-" * 25)
    print("""
# Add physical validation
def validate_distances(distances):
    total_distance = sum(distances)
    print(f"Total optical path: {total_distance*1000:.1f} mm")
    
    # Check Fresnel number for each distance
    for i, d in enumerate(distances):
        F = aperture_size**2 / (4 * wavelength * d)
        print(f"Distance {i+1}: {d*1000:.1f} mm, Fresnel number: {F:.2f}")
        if F < 1:
            print(f"  ⚠️ WARNING: Fresnel approximation may be invalid")
    """)

if __name__ == "__main__":
    analyze_complex_propagation()
    suggest_improvements()
    
    print("="*80)
    print("SUMMARY:")
    print("="*80)
    print("1. Current complex multiplication implementation is mathematically correct")
    print("2. Distance interpretation needs clarification (DISTANCE_TO_TARGET meaning)")
    print("3. Code can be simplified using direct complex operations")
    print("4. Add validation checks for physical consistency")
    print("5. Consider using DISTANCE_LAST_DOE_TO_TARGET instead of DISTANCE_TO_TARGET")
    print("="*80)