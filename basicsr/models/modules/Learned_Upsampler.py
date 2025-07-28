import torch
import torch.nn as nn
from typing import Tuple


class Learned_Upsampler(nn.Module):
    """
    This is a learned CNN module that upsamples the compressed signal to the correct spatial size. It is assumed that the input is of size (B, 1, m, T),
    where B is the batch size, m is the number of channels after compression, and T is the number of time steps. The number of levels of the upsampler is
    determined by the number of prime factors of the compression_factor. Example: if compression_factor = 6, the upsampler will have 2 levels (2, 3).
    For a given level with prime factor p, the number of filters learned will be 2p and the size of the filters will be (1, 1, p+1, p+1), Resulting in an
    output of size (B, 2p, m, T). The following the NAFBlock logic every other channel will be dotted together, resulting in an output of size (B, p, m, T), then finally
    pixel shuffle is applied to get the final output of size (B, 1, m*p, T). This is repeated for all levels/prime factors.
    """
    def __init__(self, compression_factor: int):
        super(Learned_Upsampler, self).__init__()

        if compression_factor < 1:
            raise ValueError("Compression factor must be a positive integer.")

        self.compression_factor = compression_factor
        # Get prime factors and break up any > 5 into 2s, 3s, 5s (overshoot if needed)
        raw_levels = self._get_prime_factors(compression_factor)
        self.levels = []
        for p in raw_levels:
            if p > 5:
                self.levels.extend(self._break_large_factor(p))
            else:
                self.levels.append(p)

        self.upsample_layers = nn.ModuleList()
        for p in self.levels:
            num_filters = p  # Only p filters needed for 1D upsampling
            filter_size = (p + 1, 1)
            upsample_layer = nn.Conv2d(1, num_filters, filter_size, padding=(p // 2, 0))
            self.upsample_layers.append(upsample_layer)

    def _break_large_factor(self, p):
        # Always overshoot using 2s and 3s (no 5s), then crop at the end
        factors = []
        product = 1
        while product < p:
            if (product * 3) <= p:
                factors.append(3)
                product *= 3
            else:
                factors.append(2)
                product *= 2
        # If we haven't reached or overshot, keep multiplying by 2
        while product < p:
            factors.append(2)
            product *= 2
        # If we overshot, keep multiplying by 2 until we pass p
        while product < p:
            factors.append(2)
            product *= 2
        # If we still haven't reached, add one more 2 to overshoot
        if product < p:
            factors.append(2)
        # If product < p, keep multiplying by 2
        while product < p:
            factors.append(2)
            product *= 2
        # If product > p, that's fine, we'll crop at the end
        return factors

    def _get_prime_factors(self, n: int) -> list:
        factors = []
        for i in range(2, n + 1):
            while n % i == 0:
                factors.append(i)
                n //= i
        return factors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_height = x.shape[2]
        for idx, upsample_layer in enumerate(self.upsample_layers):
            p = self.levels[idx]
            # Calculate expected output height after upsampling
            expected_out_height = x.shape[2] * p
            # Calculate required padding for height to preserve size
            kernel_height = p + 1
            pad_h = (kernel_height - 1)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            # Pad only the height dimension (3rd dim)
            x = nn.functional.pad(x, (0, 0, pad_top, pad_bottom), mode='replicate')
            x = upsample_layer(x)
            B, C, H, W = x.shape
            # Reshape to (B, 1, H*p, W) for 1D pixel shuffle (always 4D)
            x = x.permute(0, 2, 1, 3).contiguous().view(B, 1, H * p, W)
            # If output height is larger than expected (can happen due to padding), crop
            if x.shape[2] > expected_out_height:
                x = x[:, :, :expected_out_height, :]
        # Final crop to ensure output height is exactly input_height * compression_factor
        final_height = input_height * self.compression_factor
        if x.shape[2] > final_height:
            x = x[:, :, :final_height, :]
        # No padding branch: always overshoot and crop, never pad
        return x
    

# Temporary code to ensure the module can be imported and used
if __name__ == "__main__":
    # Example usage
    compression_factor = 101

    upsampler = Learned_Upsampler(compression_factor)
    input_tensor = torch.randn(1, 1, 12, 10)  # Example input tensor
    output_tensor = upsampler(input_tensor)
    print(f"Output shape: {output_tensor.shape}")  # Should reflect the upsampled dimensions
    print(f"Expected output shape: (1, 1, {12 * compression_factor}, 10)")
    print(f"Number of levels: {len(upsampler.levels)}")
    print(f"Levels (prime factors): {upsampler.levels}")