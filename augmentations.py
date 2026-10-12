"""
Strong image augmentation module
================
Contains pixel-level augmentations that simulate cartoon, sketch, oil-painting, and other styles.
All classes accept and return PIL Images and can be used in torchvision.transforms.Compose.
"""
import random
import numpy as np
import cv2
from PIL import Image

class ColorQuantization:
    """Reduce the number of image colors with K-means to simulate flat-color artwork."""
    def __init__(self, p=0.3, n_colors=6, sample_ratio=0.2):
        self.p = p
        self.n_colors = n_colors
        self.sample_ratio = sample_ratio

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_np = np.array(img).astype(np.float32)
        h, w, c = img_np.shape
        n_pixels = h * w
        n_samples = max(1000, int(n_pixels * self.sample_ratio))
        indices = np.random.choice(n_pixels, n_samples, replace=False)
        samples = img_np.reshape(-1, c)[indices].astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(samples, self.n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = centers.astype(np.float32)
        pixels = img_np.reshape(-1, c)
        dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)
        quantized = centers[nearest].reshape(h, w, c)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        return Image.fromarray(quantized)

class EdgePreservingFilter:
    """Apply bilateral filtering and sharpening for smooth interiors and crisp cartoon-like edges."""
    def __init__(self, p=0.3, d=9, sigma_color=75, sigma_space=75, sharp_strength=0.3):
        self.p = p
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.sharp_strength = sharp_strength

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_np = np.array(img).astype(np.uint8)
        filtered = cv2.bilateralFilter(img_np, self.d, self.sigma_color, self.sigma_space)
        sharpened = cv2.addWeighted(img_np, 1.0 + self.sharp_strength, filtered, -self.sharp_strength, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return Image.fromarray(sharpened)

class RandomElasticDeformation:
    """Apply a smoothed random displacement field to simulate irregular hand-drawn lines."""
    def __init__(self, p=0.3, alpha=50, sigma=5):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.array(img)
        h, w, c = arr.shape
        dx = (np.random.randn(h, w) * self.alpha).astype(np.float32)
        dy = (np.random.randn(h, w) * self.alpha).astype(np.float32)
        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma)
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        x_map = (x + dx).astype(np.float32)
        y_map = (y + dy).astype(np.float32)
        deformed = cv2.remap(arr, x_map, y_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(deformed)

class RandomEdgeEnhance:
    """Extract Canny/Laplacian edges and blend them with the original image."""
    def __init__(self, p=0.3, mode='sketch'):
        self.p = p
        self.mode = mode

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        if self.mode == 'sketch':
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            return Image.blend(img, Image.fromarray(edges), 0.5)
        elif self.mode == 'laplacian':
            edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            return Image.blend(img, Image.fromarray(edges), 0.4)
        return img

class RandomHighPass:
    """Apply a Fourier-domain high-pass filter to retain high-frequency details."""
    def __init__(self, p=0.3, cutoff_ratio=0.2):
        self.p = p
        self.cutoff_ratio = cutoff_ratio

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_np = np.array(img).astype(np.float32)
        rows, cols = img_np.shape[:2]
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.float32)
        mask[crow - int(self.cutoff_ratio * crow):crow + int(self.cutoff_ratio * crow),
             ccol - int(self.cutoff_ratio * ccol):ccol + int(self.cutoff_ratio * ccol)] = 0
        result = np.zeros_like(img_np)
        for ch in range(3):
            f = np.fft.fftshift(np.fft.fft2(img_np[:, :, ch]))
            f_filtered = f * mask
            img_channel = np.fft.ifft2(np.fft.ifftshift(f_filtered)).real
            result[:, :, ch] = np.clip(img_channel, 0, 255)
        return Image.fromarray(result.astype(np.uint8))

class StrongColorJitter:
    """Apply strong random hue, saturation, and value changes in HSV space."""
    def __init__(self, p=0.5, hue=0.1, sat=0.5, val=0.5):
        self.p = p
        self.hue = hue
        self.sat = sat
        self.val = val

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_np = np.array(img).astype(np.uint8)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-self.hue, self.hue) * 180) % 180
        hsv[..., 1] *= random.uniform(1 - self.sat, 1 + self.sat)
        hsv[..., 2] *= random.uniform(1 - self.val, 1 + self.val)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

class RandomGrayscaleWithEdge:
    """Combine grayscale conversion and edge inversion to create black lines on a white background."""
    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        gray = img.convert('L')
        img_np = np.array(gray)
        edges = cv2.Canny(img_np, 80, 200)
        edges = 255 - edges
        return Image.fromarray(edges).convert('RGB')

class MedianCutQuantization:
    """Quantize colors with median cut, handling empty slices and division by zero."""
    def __init__(self, p=0.25, n_colors=6):
        self.p = p
        self.n_colors = n_colors

    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.array(img).astype(np.float32)
        h, w, c = arr.shape
        pixels = arr.reshape(-1, c)

        def median_cut(pixels, n_colors):
            # Return a black palette for an empty pixel set.
            if len(pixels) == 0:
                return [np.zeros(c, dtype=np.float32)]
            if n_colors <= 1 or len(pixels) <= 1:
                return [pixels.mean(axis=0)]

            # Select the dimension with the largest range.
            ranges = pixels.max(axis=0) - pixels.min(axis=0)
            dim = np.argmax(ranges)

            # If all pixels are identical, return any pixel directly.
            if ranges[dim] == 0:
                return [pixels[0]]

            median = np.median(pixels[:, dim])
            left = pixels[pixels[:, dim] <= median]
            right = pixels[pixels[:, dim] > median]

            # Ensure neither side is empty.
            if len(left) == 0:
                left = right
                right = np.empty((0, c), dtype=np.float32)
            if len(right) == 0 and len(left) > 1:
                right = left[-1:].copy()
                left = left[:-1]
            if len(left) == 0 or len(right) == 0:
                return [pixels.mean(axis=0)]

            total = len(pixels)
            n_left = max(1, min(n_colors - 1, int(round(n_colors * len(left) / total))))
            n_right = n_colors - n_left
            return median_cut(left, n_left) + median_cut(right, n_right)

        palette = median_cut(pixels, self.n_colors)
        palette = np.array(palette, dtype=np.float32)

        if len(palette) == 0:
            return img

        dists = np.linalg.norm(pixels[:, None, :] - palette[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)
        quantized = palette[nearest].reshape(h, w, c)

        quantized = np.nan_to_num(quantized, nan=0.0, posinf=255.0, neginf=0.0)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        return Image.fromarray(quantized)

class WhiteningTransform:
    """Apply PCA whitening to pixel values and normalize variance to alter image color statistics."""
    def __init__(self, p=0.25, eps=1e-6):
        self.p = p
        self.eps = eps

    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.array(img).astype(np.float32)
        h, w, c = arr.shape
        pixels = arr.reshape(-1, c)
        mean = pixels.mean(axis=0)
        centered = pixels - mean
        cov = np.cov(centered, rowvar=False) + self.eps * np.eye(c)
        eigvals, eigvecs = np.linalg.eigh(cov)
        D = np.diag(1.0 / np.sqrt(eigvals))
        W = eigvecs @ D @ eigvecs.T
        whitened = centered @ W
        whitened += mean
        whitened = np.clip(whitened, 0, 255).astype(np.uint8)
        return Image.fromarray(whitened.reshape(h, w, c))

class LocalElasticDeformation:
    """Apply local elastic deformation with a smaller alpha and larger sigma for local distortion."""
    def __init__(self, p=0.2, alpha=25, sigma=3):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.array(img)
        h, w, c = arr.shape
        dx = (np.random.randn(h, w) * self.alpha).astype(np.float32)
        dy = (np.random.randn(h, w) * self.alpha).astype(np.float32)
        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma)
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        x_map = (x + dx).astype(np.float32)
        y_map = (y + dy).astype(np.float32)
        deformed = cv2.remap(arr, x_map, y_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(deformed)
