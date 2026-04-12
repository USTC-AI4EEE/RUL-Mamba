import os
import warnings

import torch


_AUTO_CPU_FFT_FALLBACK = False
_FALLBACK_WARNING_EMITTED = False


def _force_cpu_fft_enabled():
    value = os.environ.get("RULMAMBA_FORCE_CPU_FFT", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_cufft_error(error):
    message = str(error)
    return "cuFFT error" in message or "CUFFT_" in message or "cufft" in message.lower()


def _warn_cpu_fft_fallback_once():
    global _FALLBACK_WARNING_EMITTED
    if _FALLBACK_WARNING_EMITTED:
        return
    warnings.warn(
        "Detected cuFFT failure on CUDA. Falling back to CPU FFT for compatibility; "
        "training will be slower. Set RULMAMBA_FORCE_CPU_FFT=1 to force this backend from startup.",
        RuntimeWarning,
        stacklevel=3,
    )
    _FALLBACK_WARNING_EMITTED = True


def _run_fft_on_cpu(input_tensor, fft_op):
    output = fft_op(input_tensor.to("cpu"))
    return output.to(input_tensor.device)


def _run_fft_with_fallback(input_tensor, fft_op):
    global _AUTO_CPU_FFT_FALLBACK

    if not isinstance(input_tensor, torch.Tensor) or not input_tensor.is_cuda:
        return fft_op(input_tensor)

    if _force_cpu_fft_enabled() or _AUTO_CPU_FFT_FALLBACK:
        return _run_fft_on_cpu(input_tensor, fft_op)

    try:
        return fft_op(input_tensor)
    except RuntimeError as error:
        if not _is_cufft_error(error):
            raise
        _AUTO_CPU_FFT_FALLBACK = True
        _warn_cpu_fft_fallback_once()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return _run_fft_on_cpu(input_tensor, fft_op)


def compat_rfft(input_tensor, n=None, dim=-1, norm=None):
    return _run_fft_with_fallback(
        input_tensor,
        lambda tensor: torch.fft.rfft(tensor, n=n, dim=dim, norm=norm),
    )


def compat_irfft(input_tensor, n=None, dim=-1, norm=None):
    return _run_fft_with_fallback(
        input_tensor,
        lambda tensor: torch.fft.irfft(tensor, n=n, dim=dim, norm=norm),
    )
