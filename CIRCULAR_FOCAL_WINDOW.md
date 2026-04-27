# Circular Focal Window Implementation

## Summary

Changed the focal intensity calculation in the training loss function from a **rectangular window** to a **circular mask**. This provides a more physically accurate representation of the focal spot geometry.

## Changes Made

### 1. Parameter Renamed
```python
# OLD:
FOCAL_WINDOW_SIZE = 6  # Size of the focal window (default 4)

# NEW:
FOCAL_WINDOW_RADIUS = 6  # Radius of the circular focal window in pixels (default 6)
```

### 2. Loss Function Updated

**Old Implementation (Rectangular Crop):**
```python
shape = tf.shape(y_pred)
center_y = shape[1] // 2
center_x = shape[2] // 2
half_window = FOCAL_WINDOW_SIZE // 2
focal_patch = y_pred[:, 
                     center_y-half_window:center_y+half_window, 
                     center_x-half_window:center_x+half_window]
focal_intensity = tf.reduce_mean(focal_patch)
```

**New Implementation (Circular Mask):**
```python
# Create circular mask for focal region
shape = tf.shape(y_pred)
center_y = tf.cast(shape[1] // 2, tf.float32)
center_x = tf.cast(shape[2] // 2, tf.float32)

# Create coordinate grids
y_coords = tf.cast(tf.range(shape[1]), tf.float32)
x_coords = tf.cast(tf.range(shape[2]), tf.float32)
y_grid, x_grid = tf.meshgrid(y_coords, x_coords, indexing='ij')

# Calculate distance from center
dist = tf.sqrt(tf.square(y_grid - center_y) + tf.square(x_grid - center_x))

# Create circular mask (1.0 inside radius, 0.0 outside)
circular_mask = tf.cast(dist <= tf.cast(FOCAL_WINDOW_RADIUS, tf.float32), tf.float32)

# Apply mask to predictions (expand mask dimensions for broadcasting)
masked_pred = y_pred * circular_mask[tf.newaxis, :, :, tf.newaxis]

# Calculate focal intensity as mean over circular region
focal_intensity = tf.reduce_sum(masked_pred) / (tf.reduce_sum(circular_mask) * tf.cast(shape[0] * shape[3], tf.float32))
```

## Key Differences

### Area Coverage
- **Old (Square):** Area = (2 × FOCAL_WINDOW_SIZE)² = (2 × 6)² = 144 pixels
- **New (Circle):** Area = π × FOCAL_WINDOW_RADIUS² = π × 36 ≈ 113 pixels
- The circular mask covers **~78% of the rectangular area**

### Geometry
- **Rectangular:** All pixels within square boundaries contribute equally
- **Circular:** Only pixels within radius R from center contribute (more physically realistic for optical focal spots)

### Intensity Calculation
- **Old:** Simple mean over cropped patch
- **New:** Sum of masked values divided by (circular area × batch size × channels)

## Physical Motivation

Optical focal spots (especially from diffractive elements) naturally have **circular symmetry**. Using a circular focal region:

1. **More accurate representation** of actual focal spot geometry
2. **Avoids corner effects** that would be present in a square window
3. **Better alignment** with Airy disk patterns and other circular diffraction patterns
4. **Improved training signal** by excluding irrelevant corner pixels

## Impact on Training

With `FOCAL_INTENSITY_WEIGHT = 0.0` (currently disabled), this change has **no immediate effect** on training. However, when focal intensity loss is re-enabled:

- The loss function will focus on a **circular region** around the center
- This may lead to **better convergence** for focusing applications
- The focal spot shape may become **more circular** during training
- Less emphasis on maintaining intensity in the corners of the focal window

## Testing Recommendations

1. **Keep current radius:** `FOCAL_WINDOW_RADIUS = 6` (comparable to previous square half-width)
2. **Re-enable focal intensity:** Try `FOCAL_INTENSITY_WEIGHT = 0.8` or similar
3. **Monitor focal spot shape:** Check if the output focal spot becomes more circular
4. **Compare training curves:** Evaluate if convergence improves

## File Modified

- `/workspace/CD2NN-for-THz/gpu/differ_freq/train_cd2nn_model.py`
  - Line 60: Parameter renamed
  - Lines 321-338: Circular mask implementation

## Related Parameters

- `FOCAL_INTENSITY_WEIGHT = 0.0` - Currently disabled (set to non-zero to activate focal loss)
- `DOE_SHAPE = (128, 128)` - Output grid size
- Center coordinates: `(64, 64)` for 128×128 grid
- Radius = 6 pixels → focal region spans ~12-pixel diameter circle
