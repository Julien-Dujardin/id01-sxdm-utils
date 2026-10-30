import h5py
import pytest

from sxdm.io.utils import _get_chunk_indexes, _get_chunk_indexes_detector


@pytest.mark.parametrize("size", [0, 1, 2, 10, 11])
@pytest.mark.parametrize("workers", [1, 2, 3, 32])
def test_frame_chunks_cover_once(tmp_path, size, workers):
    path = tmp_path / "frames.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("data", shape=(size, 4, 5), dtype="f4")
    chunks = _get_chunk_indexes(path, "data", workers)
    assert [i for start, stop in chunks for i in range(start, stop)] == list(
        range(size)
    )
    assert len(chunks) <= workers


@pytest.mark.parametrize("workers", [1, 2, 32])
def test_detector_chunks_preserve_roi(tmp_path, workers):
    path = tmp_path / "frames.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("data", shape=(2, 8, 10), dtype="f4")
    rows, cols = _get_chunk_indexes_detector(path, "data", workers, [2, 7, 3, 9])
    assert [i for a, b in rows for i in range(a, b)] == list(range(2, 7))
    assert [i for a, b in cols for i in range(a, b)] == list(range(3, 9))


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_invalid_workers(tmp_path, workers):
    path = tmp_path / "frames.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("data", shape=(2, 4, 5), dtype="f4")
    with pytest.raises(ValueError, match="positive integer"):
        _get_chunk_indexes(path, "data", workers)
