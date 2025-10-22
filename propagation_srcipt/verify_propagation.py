#!/usr/bin/env python3
"""
Verification script for propagation distances in main.py
Checks if distances are processed correctly through the optical system.
"""

import numpy as np
from pathlib import Path

def analyze_propagation_logic():
    """Analyze the propagation distance logic from main.py"""
    
    print("="*80)
    print("PROPAGATION DISTANCE VERIFICATION")
    print("="*80)
    
    # Configuration from main.py
    DISTANCE_BETWEEN_DOE = 0.106  # [m]
    DISTANCE_TO_TARGET = 0.201    # [m] 
    
    print(f"Configuration:")
    print(f"  DISTANCE_BETWEEN_DOE = {DISTANCE_BETWEEN_DOE*1000:.1f} mm")
    print(f"  DISTANCE_TO_TARGET = {DISTANCE_TO_TARGET*1000:.1f} mm")
    print()
    
    # Simulate the logic from main.py for different numbers of masks
    for num_masks in range(1, 6):
        print(f"Analysis for {num_masks} masks:")
        print("-" * 40)
        
        total_distance = 0
        distances = []
        
        for i in range(num_masks):
            if i == num_masks-1:
                distance = DISTANCE_TO_TARGET  # Last mask uses distance to target
                stage_description = f"  Mask {i+1} -> Target"
            else:
                distance = DISTANCE_BETWEEN_DOE  # Other masks use distance between DOE
                stage_description = f"  Mask {i+1} -> Mask {i+2}"
            
            distances.append(distance)
            total_distance += distance
            print(f"  Step {i+1}: {stage_description:<20} Distance: {distance*1000:6.1f} mm")
        
        print(f"  Total optical path: {total_distance*1000:.1f} mm")
        print(f"  Distance breakdown: {[f'{d*1000:.1f}mm' for d in distances]}")
        print()
    
    print("="*80)
    print("POTENTIAL ISSUES IDENTIFIED:")
    print("="*80)
    
    issues = []
    
    # Issue 1: Distance interpretation
    print("1. DISTANCE INTERPRETATION:")
    print("   Current logic uses DISTANCE_TO_TARGET for the last propagation step.")
    print("   Question: Is this the distance from the last DOE to target,")
    print("   or the total distance from first DOE to target?")
    print()
    print("   If DISTANCE_TO_TARGET = total distance from first DOE to target:")
    print("   - For 2 DOEs: Should be DOE1->DOE2->Target")
    print("   - Distance DOE1->DOE2 = DISTANCE_BETWEEN_DOE")
    print("   - Distance DOE2->Target = DISTANCE_TO_TARGET - DISTANCE_BETWEEN_DOE")
    print()
    
    # Issue 2: Multiple DOEs spacing
    print("2. MULTIPLE DOE SPACING:")
    print("   Current logic assumes all DOEs are equally spaced by DISTANCE_BETWEEN_DOE.")
    print("   For 3+ DOEs, this may not be realistic.")
    print()
    
    # Issue 3: Physical realism check
    print("3. PHYSICAL REALISM CHECK:")
    print(f"   DISTANCE_BETWEEN_DOE = {DISTANCE_BETWEEN_DOE*1000:.1f} mm")
    print(f"   DISTANCE_TO_TARGET = {DISTANCE_TO_TARGET*1000:.1f} mm")
    print()
    if DISTANCE_TO_TARGET < DISTANCE_BETWEEN_DOE:
        print("   ⚠️  WARNING: DISTANCE_TO_TARGET < DISTANCE_BETWEEN_DOE")
        print("      This means the target is closer than the spacing between DOEs!")
        print("      This is physically impossible for multiple DOEs.")
    else:
        print("   ✓ Distance relationships are physically plausible")
    print()
    
    return issues

def suggest_corrections():
    """Suggest corrections for the propagation logic"""
    
    print("="*80)
    print("SUGGESTED CORRECTIONS:")
    print("="*80)
    
    print("1. CLARIFY DISTANCE DEFINITIONS:")
    print("   Define clearly what each distance represents:")
    print("   - d_between_doe: Physical spacing between consecutive DOEs")
    print("   - d_last_doe_to_target: Distance from last DOE to target plane")
    print("   - d_total: Total optical path length")
    print()
    
    print("2. CORRECTED LOGIC (Option A - Independent distances):")
    print("""
   # Configuration
   DISTANCE_BETWEEN_DOE = 0.106  # [m] - spacing between consecutive DOEs
   DISTANCE_LAST_DOE_TO_TARGET = 0.095  # [m] - last DOE to target
   
   for i in range(num_masks):
       if i == num_masks-1:
           distance = DISTANCE_LAST_DOE_TO_TARGET  # Last DOE to target
       else:
           distance = DISTANCE_BETWEEN_DOE  # Between consecutive DOEs
""")
    
    print("3. CORRECTED LOGIC (Option B - Total distance based):")
    print("""
   # Configuration
   TOTAL_DISTANCE_TO_TARGET = 0.201  # [m] - total from first DOE to target
   DISTANCE_BETWEEN_DOE = 0.106      # [m] - spacing between consecutive DOEs
   
   for i in range(num_masks):
       if i == num_masks-1:
           # Calculate remaining distance to target
           distance_covered = i * DISTANCE_BETWEEN_DOE
           distance = TOTAL_DISTANCE_TO_TARGET - distance_covered
       else:
           distance = DISTANCE_BETWEEN_DOE
""")
    
    print("4. VALIDATION CHECKS TO ADD:")
    print("   - Verify total optical path makes physical sense")
    print("   - Check that last distance > 0")
    print("   - Ensure DOEs don't overlap in space")
    print("   - Validate against experimental setup")
    print()

def check_propagation_formula():
    """Check if the Fresnel propagation formula is correctly implemented"""
    
    print("="*80)
    print("PROPAGATION FORMULA VERIFICATION:")
    print("="*80)
    
    print("Current implementation uses:")
    print("  H(fx, fy) = exp(j*k*z) * exp(-j*π*λ*z*(fx² + fy²))")
    print()
    print("Where:")
    print("  k = 2π/λ (wave number)")
    print("  z = propagation distance")
    print("  fx, fy = spatial frequencies")
    print("  λ = wavelength")
    print()
    
    print("This corresponds to the Fresnel diffraction transfer function.")
    print("Key points to verify:")
    print("  ✓ Fresnel approximation validity: z >> (x² + y²)/(4λ)")
    print("  ✓ Sampling requirements for FFT")
    print("  ✓ Proper handling of complex multiplication")
    print()
    
    # Check Fresnel number
    print("FRESNEL NUMBER CHECK:")
    PIXEL_SIZE = 9e-4  # [m]
    aperture_size = 128 * PIXEL_SIZE  # [m]
    max_distance = 0.201  # [m]
    wavelength = 299792458 / (180e9)  # [m] for 180 GHz
    
    fresnel_number = aperture_size**2 / (4 * wavelength * max_distance)
    print(f"  Aperture size: {aperture_size*1000:.1f} mm")
    print(f"  Max distance: {max_distance*1000:.1f} mm") 
    print(f"  Wavelength: {wavelength*1e6:.1f} μm")
    print(f"  Fresnel number: {fresnel_number:.2f}")
    
    if fresnel_number > 1:
        print("  ✓ Fresnel approximation should be valid (F > 1)")
    else:
        print("  ⚠️  Fresnel approximation may be questionable (F < 1)")
        print("     Consider using more accurate propagation methods")
    print()

if __name__ == "__main__":
    analyze_propagation_logic()
    suggest_corrections()
    check_propagation_formula()
    
    print("="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print("1. Clarify the physical meaning of DISTANCE_TO_TARGET")
    print("2. Implement proper distance calculation for multiple DOEs")
    print("3. Add validation checks for physical consistency") 
    print("4. Consider the Fresnel number limitations")
    print("5. Test with known analytical solutions")
    print("="*80)