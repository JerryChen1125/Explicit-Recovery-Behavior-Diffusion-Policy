import copy
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F



def ravel_multi_index(multi_index: torch.Tensor, dims):
    """Converts a tuple of index arrays into an array of flat indices,
    applying boundary modes to the multi-index.

    See Also: https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html

    Args:
        multi_index: [N, D] tensor.
        dims: [D]

    Returns:
        raveled_indices: [N] tensor.
    """
    dims = torch.as_tensor(dims, dtype=multi_index.dtype, device=multi_index.device)
    dims = torch.flip(dims, [0])
    strides = torch.cumprod(dims, 0)
    strides = torch.flip(strides, [0])
    strides = F.pad(strides, [0, 1], value=1)[1:]
    raveled_indices = multi_index * strides  # [N, D]
    raveled_indices = raveled_indices.sum(-1)
    return raveled_indices


def scatter_reduce(src: torch.Tensor, dim, index, reduce, output_size):
    shape = list(src.size())
    shape[dim] = output_size
    return src.new_zeros(shape).scatter_reduce(
        dim, index, src, reduce, include_self=False
    )


def scatter_add(src: torch.Tensor, dim, index, output_size):
    shape = list(src.size())
    shape[dim] = output_size
    return src.new_zeros(shape).scatter_add_(dim, index, src)


def scatter_reduce_gather(src: torch.Tensor, dim, index, reduce, output_size):
    index = index.expand_as(src)
    return torch.gather(
        scatter_reduce(src, dim, index, reduce, output_size), dim, index
    )


def get_batch_inds(points: torch.Tensor):
    return points.new_zeros([points.size(0), 1], dtype=torch.long)


class PointTensor:
    def __init__(
        self,
        points: torch.Tensor,  # [N, 3]
        feats: torch.Tensor = None,  # [N, D]
        batch_inds: torch.Tensor = None,  # [N, 1]
        bs: int = 0,
    ):
        assert points.dim() == 2 and points.size(1) == 3, points.size()

        self.points = points
        self.feats = feats
        if batch_inds is None:
            batch_inds = get_batch_inds(points)
        self.batch_inds = batch_inds
        self.bs = bs or self.batch_inds.max() + 1

    def __len__(self):
        return self.points.size(0)

    @property
    def dim(self):
        return self.points.size(1)

    @property
    def fdim(self):
        return 0 if self.feats is None else self.feats.size(1)

    def __str__(self) -> str:
        n = len(self)
        m = getattr(self, "num_cells", None)
        s = "{}(b={},n={},m={})".format(self.__class__.__name__, self.bs, n, m)
        return s

    @torch.no_grad()
    def to_grid(
        self,
        stride: Union[float, torch.Tensor],
        origin: torch.Tensor,
        round_fn=torch.round,
    ):
        coords = round_fn((self.points - origin) / stride).long()
        return torch.cat([self.batch_inds, coords], dim=1)

    @torch.no_grad()
    def voxelization(self, stride, dims):
        # Compute the origin (min coord)
        self.origin = scatter_reduce_gather(
            self.points, 0, self.batch_inds, "amin", self.bs
        )  # [N, 3]
        # Get discrete coordinates in the grid
        if isinstance(stride, (tuple, list)):
            stride = self.points.new_tensor(stride)
        self.stride = stride
        self.coords = self.to_grid(stride, self.origin)  # [N, 4]
        # Ravel into a single index
        self.inds = ravel_multi_index(self.coords, (self.bs, *dims))  # [N]
        unique_inds, inverse_inds = torch.unique(self.inds, return_inverse=True)
        self.num_cells = unique_inds.size(0)
        self.inverse_inds = inverse_inds.unsqueeze(1)  # [N, 1]

    # -------------------------------------------------------------------------- #
    # Available after voxelization
    # -------------------------------------------------------------------------- #
    def grid_center(self):
        return self.coords[:, 1:].type(torch.float) * self.stride + self.origin

    def scatter_reduce(self, x, reduce="mean", batch=False):
        if batch:
            return scatter_reduce(x, 0, self.batch_inds.expand_as(x), reduce, self.bs)
        else:
            return scatter_reduce(
                x, 0, self.inverse_inds.expand_as(x), reduce, self.num_cells
            )

    def gather(self, x):
        return torch.gather(x, 0, self.inverse_inds.expand(-1, x.size(1)))

    def downsample_v0(self, reduce):
        n = self.points.size(0)
        m = self.num_cells
        device = self.points.device

        # # Gather grid features
        # self.feats = self.gather(self.scatter_reduce(self.feats, reduce))

        # Select grid center
        sel_inds = self.batch_inds.new_zeros([m])
        sel_inds.scatter_(
            0, self.inverse_inds.squeeze(1), torch.arange(n, device=device)
        )

        points = self.grid_center()[sel_inds]
        feats = self.feats[sel_inds] if self.feats is not None else None
        batch_inds = self.batch_inds[sel_inds]
        return PointTensor(points, feats, batch_inds=batch_inds, bs=self.bs)

    def downsample_v1(self, reduce):
        n = self.points.size(0)
        m = self.num_cells
        device = self.points.device

        # Gather grid features
        self.feats = self.gather(self.scatter_reduce(self.feats, reduce))

        # Randomly select one in each cell
        sel_inds = self.batch_inds.new_zeros([m])
        sel_inds.scatter_(
            0, self.inverse_inds.squeeze(1), torch.arange(n, device=device)
        )

        points = self.points[sel_inds]
        feats = self.feats[sel_inds] if self.feats is not None else None
        batch_inds = self.batch_inds[sel_inds]
        return PointTensor(points, feats, batch_inds=batch_inds, bs=self.bs)

