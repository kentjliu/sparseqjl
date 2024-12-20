import numpy as np
import torch
import torch.nn as nn


def quantize(x, scale, zero, maxq):
    """
    Helper function that performs asymmetric quantization.
    """
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):
    """
    PyTorch module that encapsulates the process of quantization, including configuring parameters, 
    finding the optimal scaling factors and zero points, and quantizing input tensors.
    """

    def __init__(self, shape=1):
        """
        Registers three buffers: maxq, scale, and zero, which are used for storing the quantization parameters. 
        These are registered as buffers to ensure they are part of the model's state but are not trained parameters 
        (i.e., they are not updated by backpropagation).
        """
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
            self,
            bits, perchannel=False, sym=True, 
            mse=False, norm=2.4, grid=100, maxshrink=.8,
            grouprows=1
        ):
        """
        configures the quantizer by setting various parameters:
        bits: Number of bits for quantization (e.g., 8 for 8-bit quantization).
        perchannel: If True, the quantization parameters (scale, zero) will be computed per channel. Otherwise, they are computed globally for the entire tensor.
        sym: If True, symmetric quantization is used (i.e., the range for both positive and negative values is symmetric).
        mse: If True, mean squared error (MSE) optimization is applied to find the best quantization parameters.
        norm: The norm used to compute the error in MSE optimization (default is 2.4).
        grid: The number of possible scale values to test during MSE optimization.
        maxshrink: A parameter for controlling the maximum shrinkage of the range in MSE optimization.
        grouprows: Number of rows to group for the perchannel quantization case.
        """
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        self.grouprows = grouprows

    def find_params(self, x, weight=False):
        """
        Computes the quantization parameters (scale and zero) for a given tensor x.
        """
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
                if self.grouprows > 1: 
                    x = x.reshape((x.shape[0] // self.grouprows, -1))
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            if self.grouprows > 1:
                self.scale = self.scale.unsqueeze(1).repeat(1, self.grouprows)
                self.zero = self.zero.unsqueeze(1).repeat(1, self.grouprows)
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        """
        Performs the actual quantization of a tensor x using the computed scale and zero values.
        """
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        """
        Returns True if the quantizer is enabled (i.e., if maxq > 0), and False otherwise.
        """
        return self.maxq > 0

    def ready(self):
        """
        Returns True if the quantizer has valid scale and zero parameters (i.e., scale != 0), 
        indicating that it is ready to perform quantization. If the scale is zero, the quantizer is not ready.
        """
        return torch.all(self.scale != 0)
    