from typing import Optional
import cv2
import cupy as cp
from cupy.fft import fft2, ifft2

def psf2otf(psf: cp.ndarray, out_size: tuple) -> cp.ndarray:
    """
    Convert Point Spread Function (PSF) to Optical Transfer Function (OTF).
    Rewritten to use cupy.

    Parameters:
    -----------
    psf : cp.ndarray
        The PSF array
    out_size : tuple
        Desired output size of the OTF

    Returns:
    --------
    otf : cp.ndarray
        The resulting OTF (in frequency domain).
    """
    # Copy psf so that we don't modify the original
    psf = psf.astype(cp.float32)

    # Pad psf to out_size
    psf_shape = psf.shape
    pad = []
    for i in range(len(psf_shape)):
        pad_i = out_size[i] - psf_shape[i]
        pad.append((0, pad_i))
    psf_padded = cp.pad(psf, pad, mode='constant', constant_values=0)

    # Circularly shift psf so that center is at (0,0)
    for axis, axis_size in enumerate(psf_shape):
        psf_padded = cp.roll(psf_padded, -int(axis_size // 2), axis=axis)

    # Compute OTF by taking FFT
    otf = fft2(psf_padded)

    return otf

class L0Smoothing:
    """
    L0 Smoothing using cupy for array operations and FFT.
    """

    def __init__(self,
                 param_lambda: Optional[float] = 0.25,
                 param_kappa: Optional[float] = 2.0,
                 ):
        """
        Initialization of parameters.
        """
        self._lambda = param_lambda
        self._kappa = param_kappa
        self._beta_max = 1e5

    def run(self, img_cpu, isGray=False):
        """
        L0 smoothing implementation using cupy.
        """
        # Move image to GPU
        if img_cpu.ndim == 2:
            S = cp.array(img_cpu, dtype=cp.float32)
            S = S[..., cp.newaxis]  # shape becomes (H,W,1)
        else:
            S = cp.array(img_cpu, dtype=cp.float32)

        N, M, D = S.shape
        beta = 2.0 * self._lambda

        # Create PSFs on CPU, then move to GPU
        psf_x_cpu = cp.array([[-1, 1]], dtype=cp.float32)
        psf_y_cpu = cp.array([[-1], [1]], dtype=cp.float32)

        # Generate OTFs on GPU
        otfx = psf2otf(psf_x_cpu, (N, M))
        otfy = psf2otf(psf_y_cpu, (N, M))

        # Compute components used in denominator
        Denormin2 = cp.abs(otfx)**2 + cp.abs(otfy)**2
        if D > 1:
            Denormin2 = Denormin2[..., cp.newaxis]  # (N,M,1)
            Denormin2 = cp.tile(Denormin2, (1, 1, D))  # (N,M,D)

        # Initial Normin1 = FFT(S) over spatial dimensions
        # S has shape (N, M, D), we do FFT along axes=(0,1)
        Normin1 = fft2(S, axes=(0, 1))

        while beta < self._beta_max:
            Denormin = 1.0 + beta * Denormin2

            # Compute horizontal difference
            h = cp.diff(S, axis=1)  # shape (N, M-1, D)
            # wrap-around difference in the last column
            last_col = (S[:, 0, :] - S[:, -1, :])[:, cp.newaxis, :]
            h = cp.concatenate([h, last_col], axis=1)  # shape (N, M, D)

            # Compute vertical difference
            v = cp.diff(S, axis=0)  # shape (N-1, M, D)
            # wrap-around difference in the last row
            last_row = (S[0, :, :] - S[-1, :, :])[cp.newaxis, ...]
            v = cp.concatenate([v, last_row], axis=0)  # shape (N, M, D)

            # Compute gradient magnitude squared (for each pixel)
            grad = h**2 + v**2
            # sum over color channels if D > 1
            if D > 1:
                grad_sum = cp.sum(grad, axis=2)  # (N,M)
                idx = grad_sum < (self._lambda / beta)
                # Expand and tile to match shape (N,M,D)
                idx = idx[..., cp.newaxis]
                idx = cp.tile(idx, (1, 1, D))
            else:
                # Single channel
                grad_sum = grad[:, :, 0]  # shape (N,M)
                idx = grad_sum < (self._lambda / beta)

            # Zero out the gradients where condition is satisfied
            h[idx] = 0
            v[idx] = 0

            # Divergence computations
            # Horizontal
            h_diff = -cp.diff(h, axis=1)
            first_col = (h[:, -1, :] - h[:, 0, :])[:, cp.newaxis, :]
            h_diff = cp.concatenate([first_col, h_diff], axis=1)

            # Vertical
            v_diff = -cp.diff(v, axis=0)
            first_row = (v[-1, :, :] - v[0, :, :])[cp.newaxis, ...]
            v_diff = cp.concatenate([first_row, v_diff], axis=0)

            Normin2 = beta * fft2(h_diff + v_diff, axes=(0, 1))

            # Solve for S in Fourier domain
            FS = (Normin1 + Normin2) / Denormin
            S = cp.real(ifft2(FS, axes=(0, 1)))

            # Ensure shape consistency
            if S.ndim < 3:
                S = S[..., cp.newaxis]

            # Update beta
            beta *= self._kappa

        S_cpu = cp.asnumpy(S)
        S_cpu = cp.clip(S_cpu, -1, 1)
        return S_cpu
