import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseKAN(nn.Module):
    def __init__(self, units: int, use_bias: bool = True, grid_size: int = 5, spline_order: int = 3,
                 grid_range=(-1.0, 1.0),
                 spline_initialize_stddev: float = 0.1,
                 device='cpu',
                 kan_name="kan"):
        super(DenseKAN, self).__init__()
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.use_bias = use_bias
        self.device = device
        self.kan_name = kan_name

        # initialize parameters
        self.spline_initialize_stddev = spline_initialize_stddev

        self.build()

    def calc_spline_output(self, inputs):
        # calculate the B-spline output
        spline_in = self.calc_spline_values(inputs, self.grid, self.spline_order)  # (B, in_size, grid_basis_size)
        # matrix multiply: (batch, in_size, grid_basis_size) @ (in_size, grid_basis_size, out_size) -> (batch, in_size, out_size)
        spline_out = torch.einsum("bik,iko->bio", spline_in.to(self.device), self.spline_kernel.to(self.device))
        return spline_out

    def calc_spline_values(self, x: torch.Tensor, grid: torch.Tensor, spline_order: int):
        assert len(x.shape) == 2

        # add a extra dimension to do broadcasting with shape (batch_size, in_size, 1)
        x = x.unsqueeze(-1)

        # init the order-0 B-spline bases
        bases = (x >= grid[:, :-1]) & (x < grid[:, 1:])
        bases = bases.to(x.dtype)

        # iter to calculate the B-spline values
        for k in range(1, spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) \
                    + ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])

        return bases.to(self.device)

    def build(self):
        in_size = self.units

        self.in_size = in_size
        self.spline_basis_size = self.grid_size + self.spline_order
        bound = self.grid_range[1] - self.grid_range[0]

        # build grid
        grid = torch.linspace(
            self.grid_range[0] - self.spline_order * bound / self.grid_size,
            self.grid_range[1] + self.spline_order * bound / self.grid_size,
            self.grid_size + 2 * self.spline_order + 1
        )
        # expand the grid to (in_size, -1)
        grid = grid.repeat(in_size, 1)

        self.grid = nn.Parameter(
            data=grid.float(),
            requires_grad=False
        ).to(self.device)

        # the linear weights of the spline activation
        self.spline_kernel = nn.Parameter(
            data=torch.randn(self.in_size, self.spline_basis_size, self.units) * self.spline_initialize_stddev
        )

        # build scaler weights C
        self.scale_factor = nn.Parameter(
            data=torch.randn(self.in_size, self.units) * self.spline_initialize_stddev).to(self.device)

        # build bias
        if self.use_bias:
            self.bias = nn.Parameter(
                data=torch.zeros(self.units)
            ).to(self.device)
        else:
            self.bias = None

    def _check_and_reshape_inputs(self, inputs):
        shape = inputs.shape
        ndim = len(shape)
        if ndim < 2:
            raise ValueError(f"expected min_ndim=2, found ndim={ndim}. Full shape received: {shape}")

        if inputs.shape[-1] != self.in_size:
            raise ValueError(f"expected last dimension of inputs to be {self.in_size}, found {shape[-1]}")

        # reshape the inputs to (-1, in_size)
        orig_shape = shape[:-1]
        inputs = inputs.view(-1, self.in_size)

        return inputs, orig_shape

    def forward(self, inputs):
        # check the inputs, and reshape inputs into 2D tensor (-1, in_size)
        inputs = inputs.to(self.device)
        inputs, orig_shape = self._check_and_reshape_inputs(inputs)
        output_shape = torch.cat([torch.tensor(orig_shape), torch.tensor([self.units])], dim=0)

        # calculate the B-spline output
        spline_out = self.calc_spline_output(inputs)

        # calculate the basis b(x) with shape (batch_size, in_size)
        # add basis to the spline_out: phi(x) = c * (b(x) + spline(x)) using broadcasting
        spline_out += torch.nn.functional.silu(inputs).unsqueeze(-1)

        # scale the output
        spline_out *= self.scale_factor.unsqueeze(0)

        # aggregate the output using sum (on in_size dim) and reshape into the original shape
        spline_out = spline_out.sum(dim=-2).view(*output_shape)

        # add bias
        if self.use_bias:
            spline_out += self.bias

        return spline_out