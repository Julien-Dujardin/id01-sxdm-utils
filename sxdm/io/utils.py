import h5py
import os

from ._bliss import ioh5


@ioh5
def list_available_counters(h5f, scan_no):
    return list(h5f[f"{scan_no}/measurement/"].keys())


def _split_range(start, stop, n_chunks):
    """Split an interval into at most n_chunks nonempty, contiguous ranges."""
    if n_chunks is None:
        n_chunks = os.cpu_count() or 1
    if isinstance(n_chunks, bool) or not isinstance(n_chunks, int) or n_chunks < 1:
        raise ValueError("Number of chunks must be a positive integer")
    size = stop - start
    if size == 0:
        return [(start, stop)]
    count = min(size, n_chunks)
    width, extra = divmod(size, count)
    indexes = []
    for i in range(count):
        end = start + width + (i < extra)
        indexes.append((start, end))
        start = end
    return indexes


def _get_chunk_indexes(path_h5, path_in_h5, n_proc=None):
    """Partition the dataset's first dimension into contiguous process ranges."""
    with h5py.File(path_h5, "r") as h5f:
        size = h5f[path_in_h5].shape[0]
    return _split_range(0, size, n_proc)


def _get_chunk_indexes_detector(path_h5, path_in_h5, n_chunks=3, roi=None):
    """Return row and column chunk ranges within the detector ROI."""
    with h5py.File(path_h5, "r") as h5f:
        rows, cols = h5f[path_in_h5].shape[1:]
    if roi is None:
        roi = (0, rows, 0, cols)
    if len(roi) != 4 or not (
        0 <= roi[0] <= roi[1] <= rows and 0 <= roi[2] <= roi[3] <= cols
    ):
        raise ValueError("ROI must lie within the detector bounds")
    return [
        _split_range(roi[0], roi[1], n_chunks),
        _split_range(roi[2], roi[3], n_chunks),
    ]


def _get_qspace_avg_chunk(path_h5, path_in_h5, idx_mask, idx_range):
    """
    Return the q-space intensity array summed over the (flattened) sample positons
    given by `idx_range`, which is a list of tuples.
    """

    idx_list = [x for x in range(*idx_range, 1) if idx_mask[x].astype("bool")]
    with h5py.File(path_h5, "r") as h5f:
        chunk = h5f[path_in_h5][idx_list, ...]
        chunk = chunk.sum(0)

    return chunk
