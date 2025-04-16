import numpy as np

def conv2d_manual(image, kernel):
    """
    Wykonuje 2D splot (z flipem jądra) na obrazie z daną maską.

    image: np.array, 2D – wejściowy obraz
    kernel: np.array, 2D – filtr (jądro)

    Zwraca:
    output: np.array, 2D – wynik konwolucji
    """
    # Handle multi-dimensional kernels
    if len(kernel.shape) > 2:
        kh, kw, _, _ = kernel.shape  # Extract height and width of the kernel
        kernel = kernel[:, :, 0, 0]  # Reduce to 2D by selecting the first channel

    # Handle 4D tensors by iterating over the batch dimension
    if len(image.shape) == 4:
        b, h, w, c = image.shape
        kh, kw = kernel.shape
        output = np.zeros((b, h - kh + 1, w - kw + 1, c))

        for batch in range(b):
            output[batch] = conv2d_manual(image[batch], kernel)
        return output

    # Handle 3D tensors by iterating over channels
    if len(image.shape) == 3:
        h, w, c = image.shape
        kh, kw = kernel.shape
        output = np.zeros((h - kh + 1, w - kw + 1, c))

        for channel in range(c):
            output[..., channel] = conv2d_manual(image[..., channel], kernel)
        return output

    # Handle 2D tensors (original implementation)
    h, w = image.shape
    kh, kw = kernel.shape

    # Odwrócenie jądra
    kernel_flipped = np.flipud(np.fliplr(kernel))

    # Rozmiar wyjścia (bez paddingu)
    out_h = h - kh + 1
    out_w = w - kw + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel_flipped)

    return output
