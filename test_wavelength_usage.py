import numpy as np
import tensorflow as tf

print("="*80)
print("ANALYSIS: How Wavelengths Are Used in PropagationLayer")
print("="*80)

C = 3e8
FREQUENCY_MIN = 150 * 1e9
FREQUENCY_MAX = 200 * 1e9
FREQUENCY_STEP = 0.5 * 1e9

WAVELENGTH_MAX = C / FREQUENCY_MIN
WAVELENGTH_MIN = C / FREQUENCY_MAX

print(f"\n1. WAVELENGTHS IN BUILD METHOD")
print(f"   Generated range: {WAVELENGTH_MIN*1e3:.4f} - {WAVELENGTH_MAX*1e3:.4f} mm")

# Your code does this
wavelengths = np.arange(WAVELENGTH_MIN, WAVELENGTH_MAX + FREQUENCY_STEP / (C / (FREQUENCY_MAX * FREQUENCY_MIN)), FREQUENCY_STEP / (C / (FREQUENCY_MAX * FREQUENCY_MIN)))
print(f"   Number of wavelengths: {len(wavelengths)}")
print(f"   First 5: {wavelengths[:5]*1e3} mm")
print(f"   Last 5: {wavelengths[-5:]*1e3} mm")

print(f"\n2. PROBLEM 1: Wavelengths are in ASCENDING order")
print(f"   wavelengths[0] = {wavelengths[0]*1e3:.4f} mm (MINIMUM wavelength)")
print(f"   wavelengths[-1] = {wavelengths[-1]*1e3:.4f} mm (MAXIMUM wavelength)")

# Convert back to frequencies to see the issue
freqs_from_wl = C / wavelengths
print(f"\n   When converted back to frequencies:")
print(f"   C / wavelengths[0] = {freqs_from_wl[0]/1e9:.1f} GHz (MAXIMUM frequency)")
print(f"   C / wavelengths[-1] = {freqs_from_wl[-1]/1e9:.1f} GHz (MINIMUM frequency)")
print(f"   ✗ Frequencies are in DESCENDING order!")

print(f"\n3. PROBLEM 2: In call() method, you do:")
print(f"""
   wl_values = wavelength[:, 0, 0]  # Input wavelength from data
   wl_indices = tf.map_fn(
       lambda wl: tf.argmin(tf.abs(self.wavelengths - wl)),
       wl_values,
       dtype=tf.int64
   )
   h_real_batch = tf.gather(self.h_table[..., 0], wl_indices)
""")

print(f"\n   This finds the CLOSEST wavelength in the table.")
print(f"   BUT: What wavelengths are actually in your INPUT DATA?")

print(f"\n4. PROBLEM 3: Input data wavelength vs table wavelength")
print(f"   Your input has 3 channels: [real, imag, wavelength]")
print(f"   But WHERE does this wavelength come from?")
print(f"   - Is it from the data file?")
print(f"   - Or is it set manually?")
print(f"   - Does it match the table wavelengths?")

print(f"\n5. PROBLEM 4: Kernel truncation is STILL THERE!")
print(f"   Line: h_real_wavelength = h_real_batch[:, :, :H_padding]")
print(f"   Where H_padding = int(self.H // 2 + 1) = int(768 // 2 + 1) = 385")
print(f"   But h_real_batch shape is (batch, 768, 768)")
print(f"   ✗ You're truncating to (batch, 768, 385) - losing 50% of data!")

print(f"\n6. PROBLEM 5: FFT dimension mismatch")
print(f"   rfft2d(768×768) produces shape (768, 385)")
print(f"   Your kernel truncated to (768, 385) ✓ Shape matches")
print(f"   BUT: You're throwing away the spatial domain kernel data!")

print(f"\n" + "="*80)
print(f"SUMMARY OF ISSUES")
print(f"="*80)
print(f"""
CRITICAL ISSUES:

1. ✗ KERNEL TRUNCATION (Line 111-112)
   h_real_wavelength = h_real_batch[:, :, :H_padding]
   This discards 50% of the spatial domain kernel!
   
2. ✗ INCORRECT FFT LOGIC
   You compute kernels in SPATIAL domain (h_real, h_imag)
   But then try to use them in FREQUENCY domain
   This doesn't work properly!

3. ✗ COMPLEX ARITHMETIC IS WRONG
   You do: re_re = iFFT(FFT(re_u) * h_real)
           im_im = iFFT(FFT(im_u) * h_imag)
           re_im = iFFT(FFT(re_u) * h_imag)
           im_re = iFFT(FFT(im_u) * h_real)
   
   Then: out_real = re_re - im_im
         out_imag = re_im + im_re
   
   ✗ WRONG! This is not correct complex multiplication!
   
   Correct: out_complex = IFFT(FFT(in_complex) * FFT(kernel_complex))
   
4. ✗ UNNECESSARY PADDING/UNPADDING
   Using Keras ZeroPadding2D and Cropping2D - slow!
   Should use tf.pad and slicing - faster!

5. ⚠️  WAVELENGTH ORDERING
   Wavelengths are ascending, but frequencies are descending
   Make sure this matches your input data!
""")

print(f"\n" + "="*80)
print(f"WHAT SHOULD HAPPEN")
print(f"="*80)
print(f"""
CORRECT APPROACH:

1. Kernels in SPATIAL domain:
   h(x,y) = exp(i*k*z) * exp(-i*π*z*λ*(fx²+fy²))
   
2. FFT both input and kernel:
   FFT_input = rfft2d(input_spatial)
   FFT_kernel = rfft2d(kernel_spatial)
   
3. Multiply in frequency domain:
   FFT_output = FFT_input * FFT_kernel
   
4. Inverse FFT:
   output_spatial = irfft2d(FFT_output)

YOUR CURRENT CODE tries to:
   - Store kernel in spatial domain (✓ good)
   - Use it in frequency domain (✗ bad - need to FFT it first!)
   - Truncate it to match rfft2d output (✗ bad - this is wrong approach)
   - Do weird complex arithmetic (✗ bad - use proper complex numbers!)
""")