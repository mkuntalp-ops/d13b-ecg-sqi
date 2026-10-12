"""
ECG Waveform Image Rendering Engine & Visual Perturbation Suite.
Produces standardized clinical ECG images for multimodal vision models and enables
testing imaging robustness against layout, DPI, grid style, and line width variations.
"""

from typing import Optional, Tuple
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def render_ecg_image(
    ecg: np.ndarray,
    fs: float,
    dpi: int = 100,
    fig_size: Tuple[float, float] = (5.12, 2.56),
    line_width: float = 1.0,
    line_color: str = "#000000",
    grid_style: str = "clinical",  # 'clinical', 'gray', 'none'
    y_limits: Optional[Tuple[float, float]] = None
) -> Image.Image:
    """
    Renders an ECG signal into a standard PIL Image.
    
    Args:
        ecg: 1D numpy array of ECG amplitudes (in mV or standard units)
        fs: sampling frequency (Hz)
        dpi: resolution dots per inch
        fig_size: (width_inches, height_inches)
        line_width: plot stroke width
        line_color: color of the ECG trace
        grid_style: 'clinical' (traditional ECG pink/red grid), 'gray', or 'none'
        y_limits: optional fixed vertical bounds (vmin, vmax)
    """
    ecg = np.asarray(ecg).flatten()
    # Normalize units: if signal amplitude is in microvolts (e.g. BUT QDB > 20 uV), convert to mV
    abs_max = np.max(np.abs(ecg))
    if abs_max > 20.0:
        ecg = ecg / 1000.0
        
    duration = len(ecg) / fs
    t = np.linspace(0, duration, len(ecg), endpoint=False)
    
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Auto-center robustly using percentiles to avoid extreme artifacts blowing out scale
    if y_limits is not None:
        y_min, y_max = y_limits
    else:
        p1, p99 = np.percentile(ecg, [1, 99])
        margin = max(0.5, (p99 - p1) * 0.2)
        y_min, y_max = p1 - margin, p99 + margin
        # Ensure at least 1.0 mV span
        if (y_max - y_min) < 1.0:
            mid = 0.5 * (y_min + y_max)
            y_min, y_max = mid - 0.5, mid + 0.5
            
    ax.set_xlim(0, duration)
    ax.set_ylim(y_min, y_max)
    
    # Grid styling
    if grid_style == "clinical":
        # Safe grid spacing
        y_span = y_max - y_min
        y_step = 0.5 if y_span <= 5.0 else max(1.0, y_span / 8.0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(y_step))
        ax.grid(which='major', color='#ffcccc', linestyle='-', linewidth=0.8)
        ax.grid(which='minor', color='#ffe6e6', linestyle=':', linewidth=0.4)
    elif grid_style == "gray":
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.grid(which='major', color='#dddddd', linestyle='-', linewidth=0.6)
    else:
        # None
        ax.grid(False)
        
    # Plot waveform
    ax.plot(t, ecg, color=line_color, linewidth=line_width, antialiased=True)
        
    # Remove tick marks, labels, and borders for a clean ECG strip
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_color('#cccccc' if grid_style != 'none' else 'white')
        
    plt.tight_layout(pad=0.1)
    
    # Save to buffer and convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    
    img = Image.open(buf).convert('RGB')
    return img


def render_perturbed_ecg_variants(ecg: np.ndarray, fs: float) -> dict[str, Image.Image]:
    """
    Renders multiple variants with deliberate visual perturbations:
    1. Standard clinical strip (baseline)
    2. Low-resolution / pixelated (50 DPI)
    3. High-resolution (150 DPI)
    4. Thin line (0.5 pt)
    5. Thick line (2.5 pt)
    6. No-grid plain background
    7. High-contrast monochrome gray grid
    """
    variants = {}
    variants["baseline_clinical"] = render_ecg_image(ecg, fs, dpi=100, line_width=1.0, grid_style="clinical")
    variants["low_dpi_50"] = render_ecg_image(ecg, fs, dpi=50, line_width=1.0, grid_style="clinical")
    variants["high_dpi_150"] = render_ecg_image(ecg, fs, dpi=150, line_width=1.0, grid_style="clinical")
    variants["thin_line_0.5"] = render_ecg_image(ecg, fs, dpi=100, line_width=0.5, grid_style="clinical")
    variants["thick_line_2.5"] = render_ecg_image(ecg, fs, dpi=100, line_width=2.5, grid_style="clinical")
    variants["no_grid"] = render_ecg_image(ecg, fs, dpi=100, line_width=1.0, grid_style="none")
    variants["gray_grid"] = render_ecg_image(ecg, fs, dpi=100, line_width=1.0, grid_style="gray")
    return variants
