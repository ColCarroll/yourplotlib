import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
from scipy.ndimage import gaussian_filter


def load_image(image_file, max_pix=1800):
    """Load filename into a numpy array, filling in transparency with 0's.

    Parameters
    ----------
    image_file : str
        File to load. Usually works with .jpg and .png.

    Returns
    -------
    numpy.ndarray of resulting image. Has shape (w, h), (w, h, 3), or (w, h, 4)
        if black and white, color, or color with alpha channel, respectively.
    """
    image = Image.open(image_file)
    size = np.array(image.size)
    if size.max() > max_pix:
        new_size = size * max_pix // size.max()
        image = image.resize(new_size)

    mode = "L"
    alpha = image.convert("RGBA").split()[-1]
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.paste(image, mask=alpha)
    img = np.asarray(background.convert(mode))
    img = img / 255
    return img


def smooth(y, box_pts):
    """Simple smoother."""
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


class ImageLines:
    def __init__(self, image):
        self.image = image

    def make_mask(self, n_lines):
        vrows, step = np.linspace(
            0, self.image.shape[0], n_lines, dtype=int, endpoint=False, retstep=True
        )
        step = int(step)
        mask = np.zeros_like(self.image)
        mask[vrows] = 1
        return mask, vrows, step

    def make_smooth_mask(self, mask, sigma):
        smoothed = gaussian_filter(mask.astype(float), mode="mirror", sigma=sigma)
        smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
        return smoothed

    def _threshold_image(self, smoothed_image, pixel_prop, vrows, step):
        for t in np.linspace(0, 1, 200):
            if (
                (smoothed_image * self.image) > t
            ).mean() * pixel_prop < self.image.mean():
                break

        lines = ((smoothed_image * self.image) > t).astype(int)
        lines[vrows] = 1
        lines[step // 2 :: step] = 0
        return lines, t

    def make_segments(self, thresholded_img, step, cmap):
        if cmap is None:
            cmap = lambda x: "black"
        segments, widths, colors = [], [], []
        for row in np.arange(step // 2, thresholded_img.shape[0], step):
            line = thresholded_img[row - step // 2 : row + step // 2]
            iszero = np.vstack(
                (
                    np.zeros((1, line.shape[1]), int),
                    np.equal(line, 0).view(np.int8),
                    np.zeros((1, line.shape[1]), int),
                )
            )
            absdiff = np.abs(np.diff(iszero, axis=0))
            x, y = np.where(absdiff == 1)
            color = cmap(row / thresholded_img.shape[0])
            for col in np.arange(line.shape[1]):
                zero_runs = x[y == col].reshape(-1, 2)
                zero_run = zero_runs[
                    np.any(step // 2 - zero_runs < 0, axis=1)
                ].flatten()
                assert zero_run.ndim == 1
                width = max(0, zero_run.max())
                segments.append([(col - 1, -row), (col + 1, -row)])
                widths.append(width)
                colors.append(color)
        return segments, np.array(widths), np.array(colors)

    def make_line_collection(
        self,
        n_lines=50,
        sigma=None,
        smooth_pts=5,
        pixel_prop=1,
        width_scale=None,
        cmap=None,
    ):
        mask, vrows, step = self.make_mask(n_lines)
        if sigma is None:
            sigma = step / 3
        smoothed = self.make_smooth_mask(mask, sigma)
        thresholded_img, t = self._threshold_image(smoothed, pixel_prop, vrows, step)
        segments, widths, colors = self.make_segments(thresholded_img, step, cmap)

        smoothed_widths = smooth(widths, smooth_pts)
        smoothed_widths = (smoothed_widths - smoothed_widths.min()) / (
            smoothed_widths.max() - smoothed_widths.min()
        )
        if width_scale is None:
            width_scale = step / 2.2

        kwargs = {
            "n_lines": n_lines,
            "sigma": sigma,
            "smooth_pts": smooth_pts,
            "pixel_prop": pixel_prop,
            "width_scale": width_scale,
            "threshold": t,
        }
        return (
            LineCollection(
                segments,
                linewidths=width_scale * (smoothed_widths + 1) ** 2,
                colors=colors,
            ),
            kwargs,
        )
