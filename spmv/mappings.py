"""Definition of local to global maps"""

import numpy as np
from mpi4py import MPI


def upper_bound(arr, value):
    """Python version of std::upper_bound."""

    for index, elem in enumerate(arr):
        if elem > value:
            return index

    return len(arr)


class L2gMap:
    """
    Local to global map.

    Maps from local indices on the current process to global
    indices across all processes. The local process owns a
    contiguous set of the global indices, starting at
    "global_offset". Any indicies, which are not owned appear
    as ghost entries at the end of the local range.

    """

    def __init__(self, comm, local_size, ghosts):
        """
        Initialize the map.

        Parameters
        ----------
        comm: mpi4py communicator object
            The MPI communicator.
        local_size: Integer
            The number of local dofs
        ghosts: numpy.int64 array
            Array of ghost indices

        """

        self._local_size = local_size
        self._ghosts = np.sort(ghosts)
        self._comm = comm
        self._mpi_size = comm.Get_size()
        self._mpi_rank = comm.Get_rank()
        self._global_to_local = {}
        self._ghost_count = np.zeros(self._mpi_size, dtype=np.int64)
        self._ghost_local = []

        self._ranges = np.empty(1 + self._mpi_size, dtype=np.int64)
        self._ranges[0] = 0
        comm.Allgather(np.array([local_size], dtype=np.int64), self._ranges[1:])

        for index in range(self._mpi_size):
            self._ranges[index + 1] += self._ranges[index]

        r0 = self._ranges[self._mpi_rank]
        r1 = self._ranges[1 + self._mpi_rank]

        for index, ghost_index in enumerate(ghosts):
            if r0 <= ghost_index and ghost_index < r1:
                raise ValueError("Ghost index in local range.")
            self._global_to_local[ghost_index] = local_size + index
            p = upper_bound(self._ranges, ghost_index) - 1
            self._ghost_count[p] += 1
            self._ghost_local.append(ghost_index - self._ranges[p])

        self._remote_count = np.zeros(self._mpi_size, dtype=np.int64)

        comm.Alltoall(self._ghost_count, self._remote_count)

        self._neighbors = []
        self._send_count = []

        for index, count in enumerate(self._ghost_count):
            rcount = self._remote_count[index]
            if count > 0 or rcount > 0 or index == self._mpi_rank:
                self._neighbors.append(index)
                self._send_count.append(count)

        neighbor_size = len(self._neighbors)

        self._neighbors_comm = comm.Create_dist_graph_adjacent(
            self._neighbors, self._neighbors
        )

        self._recv_count = np.zeros(neighbor_size, dtype=np.int64)
        if neighbor_size == 0:
            self._send_count = [0]
            self._recv_count = [0]

        self._send_count = np.array(self._send_count, dtype=np.int64)
        self._recv_count = np.array(self._recv_count, dtype=np.int64)

        self._neighbors_comm.Neighbor_alltoall(self._send_count, self._recv_count)

        self._send_offset = [0]
        self._recv_offset = [0]
        for c in self._send_count[:-1]:
            self._send_offset.append(self._send_offset[-1] + c)
        for c in self._recv_count[:-1]:
            self._recv_offset.append(self._recv_offset[-1] + c)

        count = sum(self._recv_count)

        self._send_count = np.array(self._send_count, dtype=np.int64)
        self._recv_count = np.array(self._recv_count, dtype=np.int64)
        self._send_offset = np.array(self._send_offset, dtype=np.int64)
        self._recv_offset = np.array(self._recv_offset, dtype=np.int64)

        self._ghost_local = np.array(self._ghost_local, dtype=np.int64)

        self._indexbuffer = np.zeros(count, dtype=np.int64)

        self._neighbors_comm.Alltoallv(
            [self._ghost_local, (self._send_count, self._send_offset)],
            [self._indexbuffer, (self._recv_count, self._recv_offset)],
        )

        self._send_offset += local_size

    def update(self, vec_data):
        """Send data from local indices to ghost region of other processes."""

        num_indices = len(self._indexbuffer)

        databuf = np.zeros(num_indices, dtype=vec_data.dtype)
        databuf[:] = vec_data[self._indexbuffer]
        self._neighbors_comm.Alltoallv(
            [databuf, (self._recv_count, self._recv_offset)],
            [vec_data, (self._send_count, self._send_offset)],
        )

    def reverse_update(self, vec_data):
        """Send values from ghost vector to remotes."""

        num_indices = len(self._indexbuffer)

        databuf = np.zeros(num_indices, dtype=vec_data.dtype)
        self._neighbors_comm.Alltoallv(
            [vec_data, (self._send_count, self._send_offset)],
            [databuf, (self._recv_count, self._recv_offset)],
        )

        vec_data[self._indexbuffer] += databuf

    def global_to_local(self, global_index):
        """Global to local mapping. Works only for local and ghost indices."""

        r0 = self._ranges[self._mpi_rank]
        r1 = self._ranges[1 + self._mpi_rank]

        if r0 <= global_index and global_index < r1:
            return global_index - r0
        else:
            return self._global_to_local[global_index]

    @property
    def local_size(self):
        """Return local size."""
        return self._local_size

    @property
    def num_ghosts(self):
        """Return number of gosts."""
        return len(self._ghosts)
