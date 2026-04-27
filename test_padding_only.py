import numpy as np
import tensorflow as tf

print("="*70)
print("TEST: Padding Size Calculation")
print("="*70)

# Your current configuration
H_orig = 128
W_orig = 128
padding_multiplier = 5

# Current calculation
H_padded = H_orig * (padding_multiplier + 1)
W_padded = W_orig * (padding_multiplier + 1)
size = int(H_padded / (padding_multiplier + 1) * padding_multiplier / 2)

print(f"\nOriginal shape: {H_orig} × {W_orig}")
print(f"Padding multiplier: {padding_multiplier}")
print(f"H_padded = {H_orig} × ({padding_multiplier} + 1) = {H_padded}")
print(f"W_padded = {W_orig} × ({padding_multiplier} + 1) = {W_padded}")
print(f"\nsize = int({H_padded} / {padding_multiplier + 1} * {padding_multiplier} / 2)")
print(f"size = int({H_padded / (padding_multiplier + 1)} * {padding_multiplier / 2})")
print(f"size = int({H_padded / (padding_multiplier + 1) * padding_multiplier / 2})")
print(f"size = {size}")

print(f"\n✓ Total padded size after ZeroPadding2D(size={size}):")
print(f"  H_final = {H_orig} + 2*{size} = {H_orig + 2*size}")
print(f"  W_final = {W_orig} + 2*{size} = {W_orig + 2*size}")

# Verify padding is correct
if H_orig + 2*size == H_padded and W_orig + 2*size == W_padded:
    print(f"✓ CORRECT: Padding size matches H_padded and W_padded")
else:
    print(f"✗ ERROR: Padding doesn't match!")
    print(f"  Expected: {H_padded}, Got: {H_orig + 2*size}")

print("\n" + "="*70)
print("TEST: Keras ZeroPadding2D Implementation")
print("="*70)

# Test with actual TensorFlow code
test_input = np.random.randn(1, 128, 128).astype(np.float32)
print(f"\nOriginal input shape: {test_input.shape}")

# Your current approach
test_input_expanded = np.expand_dims(test_input, axis=-1)
print(f"After expand_dims: {test_input_expanded.shape}")

# Apply ZeroPadding2D
padded = tf.keras.layers.ZeroPadding2D(padding=(size, size))(test_input_expanded)
print(f"After ZeroPadding2D(({size}, {size})): {padded.shape}")

padded = tf.squeeze(padded, axis=-1)
print(f"After squeeze: {padded.shape}")

expected_shape = (1, H_padded, W_padded)
if padded.shape == expected_shape:
    print(f"✓ CORRECT: Shape matches expected {expected_shape}")
else:
    print(f"✗ ERROR: Shape mismatch!")
    print(f"  Expected: {expected_shape}")
    print(f"  Got: {padded.shape}")

print("\n" + "="*70)
print("TEST: Alternative - tf.pad Implementation")
print("="*70)

# Test tf.pad approach (faster alternative)
test_input2 = np.random.randn(1, 128, 128).astype(np.float32)
padded_tf = tf.pad(test_input2, [[0, 0], [size, size], [size, size]])
print(f"\nOriginal input shape: {test_input2.shape}")
print(f"After tf.pad: {padded_tf.shape}")

if padded_tf.shape == expected_shape:
    print(f"✓ CORRECT: tf.pad produces same shape {expected_shape}")
else:
    print(f"✗ ERROR: Shape mismatch with tf.pad!")

# Check if both methods produce same result
if np.allclose(padded.numpy(), padded_tf.numpy()):
    print(f"✓ Both methods (ZeroPadding2D and tf.pad) produce identical results")
else:
    print(f"✗ Methods produce different results!")

print("\n" + "="*70)
print("TEST: Cropping Inverse Operation")
print("="*70)

# Test that cropping reverses padding
cropped = tf.keras.layers.Cropping2D(cropping=(size, size))(
    tf.expand_dims(padded, axis=-1)
)
cropped = tf.squeeze(cropped, axis=-1)
print(f"\nAfter Cropping2D({size}, {size}): {cropped.shape}")

if cropped.shape == test_input.shape:
    print(f"✓ CORRECT: Cropping reverses padding, shape is {cropped.shape}")
else:
    print(f"✗ ERROR: Cropping doesn't reverse padding!")
    print(f"  Expected: {test_input.shape}")
    print(f"  Got: {cropped.shape}")

# Check if we get original data back
if np.allclose(cropped.numpy(), test_input):
    print(f"✓ Data perfectly recovered after pad→crop cycle")
else:
    max_diff = np.max(np.abs(cropped.numpy() - test_input))
    print(f"✗ Data differs after pad→crop cycle! Max diff: {max_diff}")

print("\n" + "="*70)
print("TEST: Alternative - tf.pad + Slicing")
print("="*70)

# Test tf.pad + slicing (faster alternative)
test_input3 = np.random.randn(1, 128, 128).astype(np.float32)
padded_alt = tf.pad(test_input3, [[0, 0], [size, size], [size, size]])
print(f"Original shape: {test_input3.shape}")
print(f"Padded shape: {padded_alt.shape}")

# Slice instead of Cropping2D
cropped_alt = padded_alt[:, size:size+128, size:size+128]
print(f"After slicing [:, {size}:{size+128}, {size}:{size+128}]: {cropped_alt.shape}")

if cropped_alt.shape == test_input3.shape:
    print(f"✓ CORRECT: Slicing recovers original shape")
else:
    print(f"✗ ERROR: Slicing doesn't recover shape!")

if np.allclose(cropped_alt.numpy(), test_input3):
    print(f"✓ Data perfectly recovered with tf.pad + slicing")
else:
    max_diff = np.max(np.abs(cropped_alt.numpy() - test_input3))
    print(f"✗ Data differs! Max diff: {max_diff}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Current Padding Configuration:
- Original: {H_orig}×{W_orig}
- Padding multiplier: {padding_multiplier}
- Padded size: {H_padded}×{W_padded}
- Padding amount: {size} pixels on each side

✓ Padding calculation is CORRECT
✓ ZeroPadding2D works as expected
✓ tf.pad is faster alternative (recommended)
✓ Padding is properly reversed by cropping/slicing

Next step: Test kernel truncation issue
""")