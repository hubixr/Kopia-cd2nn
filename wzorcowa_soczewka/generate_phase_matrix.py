import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

C = 299792458  # [m/s]
FREQUENCY = 175 * 1e9
WAVELENGTH = C / (FREQUENCY)
print(f"Wavelength: {WAVELENGTH*1e6:.2f} μm")
FOCAL_LENGTH = 0.2  # [m]
PIXEL_SIZE = 9e-4  # [m] - pixel size in meters (0.9 mm)

def generate_phase_matrix(size=128, wavelength=WAVELENGTH, focal_length=FOCAL_LENGTH, pixel_size=PIXEL_SIZE):
    """
    Generate a phase matrix using the formula: φ(r) = -(2π/λ)√(r² + f²)
    
    Parameters:
    - size: Size of the matrix (size x size)
    - wavelength: Wavelength λ [m]
    - focal_length: Focal length f [mm]
    - pixel_size: Physical size of each pixel [m]
    
    Returns:
    - phase_matrix: 128x128 matrix with phase values normalized to 0-255
    """
    
    # Create coordinate grids in pixel units
    x_pixels = np.linspace(-size//2, size//2, size)
    y_pixels = np.linspace(-size//2, size//2, size)
    X_pixels, Y_pixels = np.meshgrid(x_pixels, y_pixels)
    
    # Convert pixel coordinates to physical coordinates (meters)
    X = X_pixels * pixel_size  # [m]
    Y = Y_pixels * pixel_size  # [m]
    
    # Calculate r = √(x² + y²) in meters
    r = np.sqrt(X**2 + Y**2)  # [m]
    
    # Convert focal length from mm to meters for consistent units
    focal_length_m = focal_length   # [m]
    
    # Calculate phase using the formula: φ(r) = (2π/λ)√(r² + f²)
    # All units are now in meters
    phase = -(2 * np.pi / wavelength) * np.sqrt(r**2 + focal_length_m**2)
    
    # Normalize phase to 0-2π range using modulo
    phase_normalized = np.mod(phase, 2 * np.pi)
    
    # Scale to 0-255 range
    phase_matrix = (phase_normalized / (2 * np.pi) * 255).astype(np.uint8)
    
    return phase_matrix, phase_normalized

def save_phase_matrix(phase_matrix, filename):
    """Save the phase matrix as an image"""
    img = Image.fromarray(phase_matrix, mode='L')
    img.save(filename)
    print(f"Phase matrix saved as {filename}")

def visualize_phase_matrix(phase_matrix, phase_normalized):
    """Visualize the phase matrix"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the 0-255 normalized matrix
    im1 = ax1.imshow(phase_matrix, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Phase Matrix (0-255)')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    plt.colorbar(im1, ax=ax1)
    
    # Plot the 0-2π normalized matrix
    im2 = ax2.imshow(phase_normalized, cmap='hsv', vmin=0, vmax=2*np.pi)
    ax2.set_title('Phase Matrix (0-2π)')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    plt.colorbar(im2, ax=ax2, label='Phase (radians)')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Phase matrix statistics:")
    print(f"  Min value (0-255): {np.min(phase_matrix)}")
    print(f"  Max value (0-255): {np.max(phase_matrix)}")
    print(f"  Mean value (0-255): {np.mean(phase_matrix):.2f}")
    print(f"  Min phase (0-2π): {np.min(phase_normalized):.4f}")
    print(f"  Max phase (0-2π): {np.max(phase_normalized):.4f}")
    print(f"  Mean phase (0-2π): {np.mean(phase_normalized):.4f}")

if __name__ == "__main__":
    # Parameters
    SIZE = 128
    
    print(f"Generating {SIZE}x{SIZE} phase matrix using formula: φ(r) = (2π/λ)√(r² + f²)")
    print(f"Parameters:")
    print(f"  λ = {WAVELENGTH*1e6:.2f} μm (wavelength)")
    print(f"  f = {FOCAL_LENGTH:.1f} mm (focal length)")
    print(f"  pixel size = {PIXEL_SIZE*1e3:.2f} mm")
    print(f"  frequency = {FREQUENCY/1e9:.0f} GHz")
    
    # Generate the phase matrix
    phase_matrix, phase_normalized = generate_phase_matrix(SIZE, WAVELENGTH, FOCAL_LENGTH, PIXEL_SIZE)
    
    # Save as image
    save_phase_matrix(phase_matrix, 'phase_matrix_128x128.bmp')
    
    # Save as numpy array for later use
    np.save('phase_matrix_128x128.npy', phase_matrix)
    np.save('phase_normalized_128x128.npy', phase_normalized)
    print("Phase matrices saved as .npy files")
    
    # Visualize the results
    visualize_phase_matrix(phase_matrix, phase_normalized)
    
    # Save a text file with the center cross-section for verification
    center = SIZE // 2
    cross_section = phase_normalized[center, :]
    np.savetxt('phase_cross_section.txt', cross_section, 
               header=f'Phase cross-section at y={center} (0-2π range)')
    print("Phase cross-section saved as phase_cross_section.txt")