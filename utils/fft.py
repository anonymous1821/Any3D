import torch
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO
from pysteps.utils import spectral

def nc_get_low_or_high_fft(x, scale=1., is_low=False):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    B, C, H, W = x_freq.shape
    
    # extract
    if is_low:
        mask = torch.zeros((B, C, H, W), device=x.device)
        crow, ccol = H // 2, W // 2
        mask[..., crow - int(crow * scale):crow + int(crow * scale), ccol - int(ccol * scale):ccol + int(ccol * scale)] = 1
    else:
        mask = torch.ones((B, C, H, W), device=x.device)
        crow, ccol = H // 2, W //2
        mask[..., crow - int(crow * scale):crow + int(crow * scale), ccol - int(ccol * scale):ccol + int(ccol * scale)] = 0

    x_freq = x_freq * mask
    
    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered

def fftshift(x, dim=None):
    """
        Shift the zero-frequency component to the center of the spectrum.
    """
    if dim is None:
        dim = tuple(range(x.dim()))
    for d in dim:
        n = x.size(d)
        shift = n // 2
        x = torch.roll(x, shifts=shift, dims=d)
    return x

def ifftshift(x, dim=None):
    """
        Inverse shift: shift the zero-frequency component back to the original position.
    """
    if dim is None:
        dim = tuple(range(x.dim()))
    for d in dim:
        n = x.size(d)
        shift = -(n // 2)
        x = torch.roll(x, shifts=shift, dims=d)
    return x

def extract_low_frequency_and_mask(latent, low_freq_ratio=0.0625):
    # Apply 2D FFT
    latent_fft = torch.fft.fft2(latent)
        
    # Shift zero frequency to center
    latent_fft_shifted = fftshift(latent_fft, dim=(-2, -1))
                    
    # Create low-frequency mask
    mask = create_low_frequency_mask(latent.shape, low_freq_ratio).to(latent.device)
                                
    # Expand mask to match the number of channels
    mask = mask.expand(latent.shape[0], latent.shape[1], -1, -1)
                                            
    # Apply mask
    latent_low_freq = latent_fft_shifted * mask
            
    # Inverse shift
    latent_low_freq = ifftshift(latent_low_freq, dim=(-2, -1))
    
    # Inverse FFT to get back to spatial domain
    latent_low = torch.fft.ifft2(latent_low_freq).real  # Take the real part
        
    return latent_low, mask

def create_low_frequency_mask(latent_shape, low_freq_ratio=0.0625):
    """
        Create a Gaussian low-frequency mask with a smooth transition.
    """
    
    _, _, H, W = latent_shape
    mask = torch.zeros(1, 1, H, W, dtype=torch.float32)

    fy = torch.linspace(-0.5, 0.5, H, dtype=torch.float32)
    fx = torch.linspace(-0.5, 0.5, W, dtype=torch.float32)
    FY, FX = torch.meshgrid(fy, fx, indexing='ij')
    distance = torch.sqrt(FX**2 + FY**2)
    sigma = low_freq_ratio
    mask = torch.exp(-0.5 * (distance / sigma)**2)
                                                                            
    return mask

def create_high_frequency_mask(latent_shape, low_freq_ratio=0.0625):
    return 1 - create_low_frequency_mask(latent_shape, low_freq_ratio=low_freq_ratio)

def extract_high_frequency(latent, low_freq_ratio=0.0625):
    # Apply 2D FFT
    latent_fft = torch.fft.fft2(latent)

    # Shift zero frequency to center
    latent_fft_shifted = fftshift(latent_fft, dim=(-2, -1))

    # Create high-frequency mask
    mask = create_high_frequency_mask(latent.shape, low_freq_ratio).to(latent.device)

    # Expand mask to match the number of channels
    mask = mask.expand(latent.shape[0], latent.shape[1], -1, -1)

    # Apply mask
    latent_high_freq = latent_fft_shifted * mask

    # Inverse shift
    latent_high_freq = ifftshift(latent_high_freq, dim=(-2, -1))

    # Inverse FFT to get back to spatial domain
    latent_high = torch.fft.ifft2(latent_high_freq).real  # Take the real part

    return latent_high

def create_mid_frequency_mask_smooth(latent_shape, low_freq_ratio=0.25, high_freq_ratio=0.99):
    """
    Create a smooth mask that gradually transitions over the mid-frequency range.
    """
    _, _, H, W = latent_shape
    y = torch.linspace(-1, 1, H).unsqueeze(1).repeat(1, W)
    x = torch.linspace(-1, 1, W).unsqueeze(0).repeat(H, 1)
    distance = torch.sqrt(x**2 + y**2) / torch.sqrt(torch.tensor(2.0))
    
    # Create low-frequency Gaussian
    sigma_low = low_freq_ratio / 2
    low_mask = torch.exp(-0.5 * (distance / sigma_low)**2)
    
    # Create high-frequency Gaussian
    sigma_high = (1 - high_freq_ratio) / 2
    high_mask = torch.exp(-0.5 * ((distance - 1) / sigma_high)**2)
    
    # Combine masks to get mid-frequency mask
    mask = low_mask + high_mask
    mask = torch.clamp(mask, 0, 1)
    
    return 1 - mask  # Invert mask to zero out mid frequencies

def combine_latents(latent_low_freq, latent_high_freq):
    return latent_low_freq + latent_high_freq

def compute_spectrum_pil(image):
    """
    Compute the log magnitude spectrum of a PIL image and return it as a PIL Image.

    Parameters:
    - image: PIL.Image.Image object

    Returns:
    - spectrum_image: PIL.Image.Image object representing the spectrum
    """
    # Ensure the image is in grayscale
    image_gray = image.convert('L')
    # Convert the image to a numpy array
    image_array = np.asarray(image_gray)
    # Compute the 2D Fast Fourier Transform (FFT)
    spec = np.fft.fft2(image_array)
    # Shift the zero-frequency component to the center of the spectrum
    spec_shifted = np.fft.fftshift(spec)
    # Compute the magnitude spectrum and apply logarithmic scaling
    spectrum = np.log(np.abs(spec_shifted) + 1e-8)  # Add a small value to avoid log(0)
    # Normalize the spectrum to the range [0, 255] for image representation
    spectrum_norm = spectrum - spectrum.min()
    spectrum_norm = spectrum_norm / spectrum_norm.max()  # Normalize to [0,1]
    spectrum_uint8 = (spectrum_norm * 255).astype(np.uint8)
    # Convert the numpy array back to a PIL Image
    spectrum_image = Image.fromarray(spectrum_uint8)
    return spectrum_image

def compute_rapsd_pil(image_pil, title='RAPSD'):
    """
    Compute the Radially Averaged Power Spectral Density (RAPSD) of a PIL image
    and return the RAPSD plot as a PIL Image.

    Parameters:
    - image_pil: PIL.Image.Image object

    Returns:
    - rapsd_pil_image: PIL.Image.Image object representing the RAPSD plot
    """
    # Ensure the image is in grayscale
    image_gray = image_pil.convert('L')
    # Convert the image to a numpy array
    image_array = np.asarray(image_gray)
    # Calculate the RAPSD
    rapsd, frequencies = spectral.rapsd(image_array, fft_method=np.fft, return_freq=True)
    # Generate the plot
    plt.figure()
    plt.plot(frequencies[1:], rapsd[1:], c='red', marker='o', markersize=3)  # Exclude the DC component
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density')
    plt.title(title)
    plt.tight_layout()

    # Save the plot to a BytesIO buffer and convert it to a PIL image
    buf = BytesIO()
    plt.savefig(buf, format='PNG', dpi=300)
    plt.close()
    buf.seek(0)
    rapsd_pil_image = Image.open(buf)
    return rapsd_pil_image