import numpy as np
import multiprocessing as mp
import os
import h5py

from tqdm import tqdm
from functools import partial
from datetime import datetime

from ._bliss import (
    ioh5,
    get_command,
    get_counter,
    get_positioner,
    get_detector_aliases,
    get_scan_shape,
    get_roi_names,
    _check_detector,
)

from .utils import _get_chunk_indexes
from .scan_positions import get_scan_positions


def get_scan_map(path_dset, **kwargs):
    """Return a browser-native 2D scan overview. See sxdm.plot.get_scan_map."""
    from ..plot.interactive import get_scan_map as plot_scans

    return plot_scans(path_dset, **kwargs)


def display_scan_positions(path_dset, **kwargs):
    """Map scans onto a normalized rectangle; see sxdm.plot.display_scan_positions."""
    from ..plot.interactive import display_scan_positions as display

    return display(path_dset, **kwargs)


@ioh5
def get_sxdm_scan_numbers(h5f, interrupted_scans=False):
    """
    Extracts scan numbers corresponding to SXDM scans from an input BLISS HDF5 dataset.

    Parameters
    ----------
    h5f : h5py.File
        The BLISS HDF5 dataset from which to extract SXDM scan numbers.
    interrupted_scans : bool, optional
        A flag indicating whether to include interrupted scans. Default is False.

    Returns
    -------
    scan_nos : list
        A list of scan numbers corresponding to SXDM scans.
    """

    scan_nos = []
    for entry in list(h5f.keys()):
        command = h5f[f"{entry}/title"][()].decode()
        if any([s in command for s in ("sxdm", "mesh", "kmap")]):
            if not interrupted_scans:
                try:
                    h5f[entry]["measurement"]
                    scan_nos.append(entry)
                except KeyError:
                    pass
            else:
                scan_nos.append(entry)

    scan_nos.sort(key=lambda s: int(s.split(".")[0]))
    scan_nos = [s for s in scan_nos if not s.endswith(".2")]

    return scan_nos


@ioh5
def get_datetime(h5f, scan_no):
    """
    Retrieves and formats the start time of a specific scan from a BLSS HDF5 dataset.

    Parameters
    ----------
    h5f : h5py.File
        The BLISS HDF5 dataset containing the target scan.
    scan_no : str
        The scan number for which the start time is to be retrieved.

    Returns
    -------
    str
        The formatted start time of the scan.
    """
    scan_no = scan_no
    dtime = h5f[f"{scan_no}/start_time"][()].decode()
    dtime = datetime.fromisoformat(dtime).strftime("%b %d | %H:%M:%S")

    return dtime


@ioh5
def get_det_params(h5f, scan_no):
    """
    Retrieves detector calibration parameters from a BLISS HDF5 dataset for a
    specific scan.

    Parameters
    ----------
    h5f : h5py.File
        The BLISS HDF5 dataset from which to extract the parameters.
    scan_no : str
        The scan number for which the detector parameters are to be retrieved.

    Returns
    -------
    dict
        A dictionary containing the detector calibration parameters.
    """
    det_alias = get_detector_aliases(h5f, scan_no)

    params = dict(beam_energy=0, center_chan=[], chan_per_deg=[])
    calib = h5f[f"{scan_no}/instrument/{det_alias}"]

    for key in params.keys():
        if key == "beam_energy":
            params[key] = calib[key][()]
        else:
            params[key] = [calib[f"{key}_dim{i}"][()] for i in (0, 1)]

    return params


def get_piezo_motor_names(h5f, scan_no):
    """
    Retrieve the names of the piezo motors used in the specified scan.

    Parameters:
    -----------
    h5f : h5py.File
        The BLISS HDF5 dataset containing the scan data.
    scan_no : str
        The scan number for which to retrieve the piezo motor names.

    Returns:
    --------
    tuple
        A tuple containing the names of the two piezo motors (m1_name, m2_name).

    Raises:
    -------
    Exception
        If the scan is not a mesh or an sxdm scan, an exception is raised with
        an appropriate message.
    """
    command = get_command(h5f, scan_no)
    if any([x in command for x in ("sxdm", "kmap", "mesh")]):
        m1_name, m2_name = [command.split(" ")[x] for x in (1, 5)]
        if "," in m1_name:
            m1_name, m2_name = m1_name[:-1], m2_name[:-1]
    else:
        fname = os.path.basename(h5f)
        msg = f"Scan {scan_no} in {fname} is not a mesh or an sxdm scan!"
        raise Exception(msg)

    return m1_name, m2_name


def get_piezo_motor_positions(h5f, scan_no):
    """
    Retrieve the sample coordinates of an SXDM scan.

    Parameters
    ----------
    h5f : str
        Path to the .hdf5 BLISS dataset.
    scan_no : str
        Number of the SXDM scan, e.g. 4.1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The coordinates of the two motors of the piezoelectric stage as 2D NumPy
        arrays.
    """
    sh = get_scan_shape(h5f, scan_no)
    m1n, m2n = get_piezo_motor_names(h5f, scan_no)

    try:  # sxdm
        m1, m2 = [get_positioner(h5f, scan_no, f"{m}_position") for m in (m1n, m2n)]
    except KeyError:
        try:  # sxdm new
            m1, m2 = [get_positioner(h5f, scan_no, f"{m}") for m in (m1n, m2n)]
        except KeyError:  # mesh
            m1, m2 = [get_positioner(h5f, scan_no, m) for m in (m1n, m2n)]
    m1, m2 = [m.reshape(*sh) for m in (m1, m2)]

    return m1, m2


def get_roidata(h5f, scan_no, roi_name, return_pi_motors=False):
    """
    DEPRECATED. Use get_counter_sxdm instead.
    """
    return get_counter_sxdm(h5f, scan_no, roi_name, return_pi_motors=return_pi_motors)


def get_counter_sxdm(h5f, scan_no, counter, return_pi_motors=False):
    """
    Retrieve counter data for a specific counter from an SXDM scan.

    Parameters
    ----------
    h5f : str
        Path to the HDF5 BLISS dataset containing the SXDM data.
    scan_no : str
        The scan number for the SXDM scan, e.g. 1.1.
    counter : str
        The name of the counter for which the data is retrieved.
    return_pi_motors : bool, optional
        If True, also return the positions of the piezo motors (m1, m2) associated
        with the scan. Default is False.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, float, float]
        If return_pi_motors is False:
            The counter data for the specified counter, reshaped to match the
            scan shape.
        If return_pi_motors is True:
            A tuple containing:
            - The counter data for the specified counter, reshaped to match the
            scan shape.
            - The position of piezo motor m1.
            - The position of piezo motor m2.
    """

    sh = get_scan_shape(h5f, scan_no)
    data = get_counter(h5f, scan_no, counter)

    if data.size == sh[0] * sh[1]:
        data = data.reshape(*sh)
    else:
        empty = np.zeros(sh).flatten()
        empty[: data.size] = data
        data = empty.reshape(*sh)

    if return_pi_motors:
        m1, m2 = get_piezo_motor_positions(h5f, scan_no)
        return data, m1, m2
    else:
        return data


def _batch_size(dataset, roi_shape):
    # Bound each worker's input to approximately 32 MiB (at least one frame).
    frame_bytes = max(1, int(np.prod(roi_shape)) * dataset.dtype.itemsize)
    return min(1024, max(1, (32 * 1024**2) // frame_bytes))


def _map_chunks(func, indexes, n_proc):
    """Run small/default jobs in-process; limit pools to actual work."""
    if n_proc == 1 or len(indexes) == 1:
        yield from map(func, indexes)
    else:
        with mp.Pool(processes=min(n_proc, len(indexes))) as pool:
            yield from pool.imap(func, indexes)


def _get_frames_chunk(path_dset, path_in_h5, roi_dir_idxs, roi, idx_range):
    """Sum selected frames in bounded batches, with ROI applied during reading."""
    selected = np.asarray(roi_dir_idxs)
    selected = selected[(selected >= idx_range[0]) & (selected < idx_range[1])]
    with h5py.File(path_dset, "r") as h5f:
        dataset = h5f[path_in_h5]
        rows, cols = dataset.shape[1:]
        if roi is None:
            roi = (0, rows, 0, cols)
        if len(roi) != 4 or not (
            0 <= roi[0] <= roi[1] <= rows and 0 <= roi[2] <= roi[3] <= cols
        ):
            raise ValueError("ROI must lie within the detector bounds")
        shape = (roi[1] - roi[0], roi[3] - roi[2])
        dtype = np.sum(np.empty(0, dtype=dataset.dtype)).dtype
        result = np.zeros(shape, dtype=dtype)
        batch = _batch_size(dataset, shape)
        for start in range(0, selected.size, batch):
            indexes = selected[start : start + batch]
            selection = (
                slice(int(indexes[0]), int(indexes[-1]) + 1)
                if np.all(np.diff(indexes) == 1)
                else indexes.tolist()
            )
            frames = dataset[selection, roi[0] : roi[1], roi[2] : roi[3]]
            result += frames.sum(axis=0)
    return result


def get_sxdm_frame_sum(
    path_dset,
    scan_no,
    mask_sample=None,
    detector=None,
    n_proc=None,
    pbar=True,
    path_data_h5="/{scan_no}/instrument/{detector}/data",
    roi=None,
):
    """Return the sum of all detector frames collected within an SXDM scan.


    Parameters
    ----------
    path_dset : str
        Path to the .hdf5 BLISS dataset.
    scan_no : str
        Number of the SXDM scan, e.g. 4.1.
    mask_sample : np.ndarray, optional
        Array of the same shape as the SXDM scan whose True values indicate
        where to exclude frames from the sum, by default None (all the scanned area is
        considered in the computation)
    detector : str, optional
        The detector used for the SXDM scan, by default None
    n_proc : int, optional
        Number of processes. None uses one process to avoid notebook startup overhead
    pbar : bool, optional
        Spawn a process bar, by default True
    path_data_h5 : str, optional
        Path within the .hdf5 BLISS dataset where to look for the raw data,
        by default "/{scan_no}/instrument/{detector}/data"
    roi : list, optional
        List of [row_min, row_max, col_min, col_max], by default None

    Returns
    -------
    np.ndarray
        Sum of all detector frames collected as part of a single SXDM scan.

    Raises
    ------
    ValueError
        The specified detector is not contained in the .hdf5 dataset.
    """

    detlist = get_detector_aliases(path_dset, scan_no)
    if detector is None:
        detector = detlist[0]
    if detector not in detlist:
        raise ValueError(
            f"Detector {detector} not in data file. Available detectors are: {detlist}."
        )
    else:
        path_data_h5 = path_data_h5.format(scan_no=scan_no, detector=detector)

        if n_proc is None:
            n_proc = 1

        # list of idx ranges [(i0, i1), (i0, i1), ...]
        indexes = _get_chunk_indexes(path_dset, path_data_h5, n_proc)

        # direct space shape (1D)
        with h5py.File(path_dset, "r") as h5f:
            sh = h5f[path_data_h5].shape[0]

        # direct space slice from mask
        if mask_sample is not None:
            if np.asarray(mask_sample).size != sh:
                raise ValueError("Sample mask must contain one value per frame")
            mask_sample = np.asarray(mask_sample, dtype=bool)
            roi_dir_idxs = np.where(np.invert(mask_sample).flatten())[0]
        else:
            roi_dir_idxs = np.indices((sh,))[0]

        pfun = partial(_get_frames_chunk, path_dset, path_data_h5, roi_dir_idxs, roi)

        # set progress bar
        if pbar is True:
            pbar = tqdm()
        elif isinstance(pbar, tqdm):
            pbar.total = len(indexes)
            pbar.refresh()

        # apply partial function multi process, with an index range per process
        frame_sum_list = []
        try:
            pbar.reset(total=len(indexes))
        except AttributeError:
            pass
        for res in _map_chunks(pfun, indexes, n_proc):
            frame_sum_list.append(res)
            try:
                pbar.update()
                pbar.refresh()
            except AttributeError:  # pbar=False
                pass

        if pbar:
            pbar.close()

        return np.stack(frame_sum_list).sum(0)


def _calc_pos_sum_chunk(
    path_dset, path_in_h5, roi_rec_sl, idx_range, mask_detector=None
):
    """Integrate detector ROIs in bounded batches, honoring irregular masks."""
    i0, i1 = idx_range
    with h5py.File(path_dset, "r") as h5f:
        dataset = h5f[path_in_h5]
        roi_shape = tuple(
            len(range(*sl.indices(n))) for sl, n in zip(roi_rec_sl, dataset.shape[1:])
        )
        batch = _batch_size(dataset, roi_shape)
        dtype = np.sum(np.empty(0, dtype=dataset.dtype)).dtype
        result = np.empty(i1 - i0, dtype=dtype)
        for start in range(i0, i1, batch):
            stop = min(start + batch, i1)
            frames = dataset[(slice(start, stop), *roi_rec_sl)]
            if mask_detector is not None:
                frames = np.where(mask_detector[roi_rec_sl], 0, frames)
            result[start - i0 : stop - i0] = frames.sum(axis=(1, 2))
    return result


def get_sxdm_pos_sum(
    path_dset,
    scan_no,
    mask_detector=None,
    detector=None,
    n_proc=None,
    pbar=True,
    path_data_h5="/{scan_no}/instrument/{detector}/data",
):
    """Obtain the sum of scattered intensity integrated over the detector space for
    an SXDM scan.

    Parameters
    ----------
    path_dset : str
        Path to the .hdf5 BLISS dataset file
    scan_no : str
        Number of the SXDM scan, e.g. 4.1.
    mask_detector : np.ndarray, optional
        Array of the same shape as a detector frame whose True values indicate where
        to exclude from the sum, by default None (the full detector area is considered for
        the computation)
    detector : str, optional
        Alias of the detector used for the SXDM scan, by default None
    n_proc : int, optional
        Number of processes. None uses one process to avoid notebook startup overhead
    pbar : bool, optional
        Spawn a process bar, by default True
    path_data_h5 : str, optional
        Path within the .hdf5 BLISS dataset where to look for the raw data,
        by default "/{scan_no}/instrument/{detector}/data"

    Returns
    -------
    np.ndarray
        Sum of scattered intensity over the detector dimensions.

    Raises
    ------
    ValueError
        The specified detector alias does not exist.
    """

    detlist = get_detector_aliases(path_dset, scan_no)
    if detector is None:
        detector = detlist[0]
    if detector not in detlist:
        raise ValueError(
            f"Detector {detector} not in data file. Available detectors are: {detlist}."
        )
    else:
        path_data_h5 = path_data_h5.format(scan_no=scan_no, detector=detector)

    if n_proc is None:
        n_proc = 1

    # list of idx ranges [(i0, i1), (i0, i1), ...]
    idxs_list = _get_chunk_indexes(path_dset, path_data_h5, n_proc=n_proc)

    # recipocal space slice from mask
    if mask_detector is not None:
        with h5py.File(path_dset, "r") as h5f:
            if np.shape(mask_detector) != h5f[path_data_h5].shape[1:]:
                raise ValueError("Detector mask must match the detector frame shape")
        mask_detector = np.asarray(mask_detector, dtype=bool)
        roi_rec = np.where(np.invert(mask_detector.astype("bool")))
        mask_detector = np.asarray(mask_detector, dtype=bool)
        if roi_rec[0].size:
            roi_rec_sl = tuple(slice(x.min(), x.max() + 1) for x in roi_rec)
        else:
            roi_rec_sl = np.s_[0:0, 0:0]
    else:
        roi_rec_sl = np.s_[:, :]

    # call function with everything except the index ranges
    # the function returns a direct space map
    pfun = partial(
        _calc_pos_sum_chunk,
        path_dset,
        path_data_h5,
        roi_rec_sl,
        mask_detector=mask_detector,
    )

    # set progress bar
    if pbar is True:
        pbar = tqdm()
    elif isinstance(pbar, tqdm):
        pbar.total = len(idxs_list)
        pbar.refresh()

    # apply partial function multi process, with an index range per process
    roi_sum_list = []
    try:
        pbar.reset(total=len(idxs_list))
    except AttributeError:
        pass
    for res in _map_chunks(pfun, idxs_list, n_proc):
        roi_sum_list.append(res)
        try:
            pbar.update()
            pbar.refresh()
        except AttributeError:  # pbar=False
            pass

    if pbar:
        pbar.close()

    return np.ma.concatenate(roi_sum_list)


# TODO
# replace with new funtion
@ioh5
def get_roi_pos(h5f, scan_no, roi_names_list, detector="mpx1x4"):
    """
    Only works for ROIs starting with `detector_`
    x,y,width,height
    """

    roi_params = {key: None for key in roi_names_list}
    badrois = []
    for r in roi_params.keys():
        try:
            roi_params[r] = [
                h5f[f"/{scan_no}/instrument/{detector}_{r}/selection/{m}"][()]
                for m in "x,y,width,height".split(",")
            ]
        except KeyError:
            badrois.append(r)

    for r in badrois:
        del roi_params[r]

    return roi_params


@ioh5
def _get_roi_pos_new(h5f, scan_no, detector=None):
    """
    TBD
    """

    detector = _check_detector(h5f.filename, scan_no, detector)
    roi_names = get_roi_names(h5f.filename, scan_no, only_sum=True)

    roi_params = {key: None for key in roi_names}
    badrois = []
    for r in roi_params.keys():
        try:
            roi_params[r] = [
                h5f[f"/{scan_no}/instrument/{r}/selection/{m}"][()]
                for m in "x,y,width,height".split(",")
            ]
        except KeyError:
            badrois.append(r)

    for r in badrois:
        del roi_params[r]

    return roi_params


def get_scan_table(path_dset, *, as_widget=True, scan_type=None):
    """Get an HTML table of scans contained within a BLISS dataset.

    Parameters
    ----------
    path_dset : str
        Path to the .hdf5 BLISS dataset.

    as_widget : bool, optional
        Keep the legacy ipywidgets return by default. Set False to get an HTML
        string suitable for ``mo.Html(get_scan_table(path, as_widget=False))``.

    scan_type : str or sequence of str, optional
        Filter by scan family (sxdm, mesh, ascan) or exact supported command.
        None includes all scans. Sequences combine types.

    Returns
    -------
    ipywidgets.HTML or str
        Scan table, with escaped metadata.
    """

    table = [
        "<div>",
        '<table class="specs rendered_html output_html" style="font-size: small">',
        "  <tbody>",
        "    <tr>",
        "      <th>Scan</th>",
        "      <th>Command</th>",
        "      <th>Date | Time</th>",
        "    </tr>",
    ]

    from html import escape
    from .scan_positions import (
        _scan_entries,
        _scalar_text,
        _scan_type_commands,
        _scan_command,
    )

    commands = _scan_type_commands(scan_type)
    with h5py.File(path_dset, "r") as h5f:
        for scan_no in _scan_entries(h5f):
            scan = h5f[scan_no]
            title = _scalar_text(scan, "title")
            if commands is not None and _scan_command(title) not in commands:
                continue
            dtime = _scalar_text(scan, "start_time")
            if dtime:
                try:
                    dtime = datetime.fromisoformat(dtime).strftime("%b %d | %H:%M:%S")
                except ValueError:
                    pass  # Show non-ISO timestamps verbatim instead of hiding a scan.
            table.extend(
                [
                    "    <tr>",
                    f"      <th>{escape(scan_no)}</th>",
                    f"      <td>{escape(title)}</td>",
                    f"      <td>{escape(dtime)}</td>",
                    "    </tr>",
                ]
            )

    table += ["  </tbody>", "</table>", "</div>"]
    table = "\n".join(table)
    if not as_widget:
        return table

    try:
        from ipywidgets import HTML
    except ImportError as exc:
        raise ImportError(
            "The legacy table needs id01-sxdm[jupyter]. "
            "Use get_scan_map() for a marimo-compatible scan overview."
        ) from exc
    return HTML(table)


def get_sxdm_frame_sum_multi(
    path_dset,
    path_save_framesum=None,
    scan_nums=None,
    detector=None,
    path_data_h5="/{scan_no}/instrument/{detector}/data",
):
    if scan_nums is None:
        scan_nums = get_sxdm_scan_numbers(path_dset)

    detlist = get_detector_aliases(path_dset, scan_nums[0])
    if detector is None:
        detector = detlist[0]
    if detector not in detlist:
        raise ValueError(
            f"Detector {detector} not in data file. Available detectors are: {detlist}."
        )
    else:
        first_data_path = path_data_h5.format(scan_no=scan_nums[0], detector=detector)

    with h5py.File(path_dset) as h5f:
        frame_shape = h5f[first_data_path].shape[1:3]

    # if no fname specified or if fname does not exist
    if path_save_framesum is None or not os.path.isfile(path_save_framesum):
        fint_tot = np.zeros((*frame_shape, len(scan_nums)))
        for i, scan_no in tqdm(enumerate(scan_nums), total=len(scan_nums)):
            fint_tot[..., i] = get_sxdm_frame_sum(
                path_dset,
                scan_no,
                detector=detector,
                path_data_h5=path_data_h5,
                pbar=False,
            )
        fint_tot = fint_tot.transpose(2, 0, 1)
        # if fname
        if path_save_framesum is not None:
            np.save(path_save_framesum, fint_tot)
    # if fname specified and exists
    else:
        print(f"Loading: \n\t{path_save_framesum}\n")
        fint_tot = np.load(path_save_framesum)

    return fint_tot


def get_detector_frame(path_dset, scan_no, frame_index=0, *, detector=None, roi=None):
    """Read just one detector image, optionally cropped before loading.

    ``frame_index`` is the zero-based acquisition index. ``roi`` is
    ``(row_start, row_stop, col_start, col_stop)`` with exclusive stop bounds.
    No file handle is retained, making this safe for reactive marimo cells.
    """
    from numbers import Integral
    from ._bliss import _scan_no

    scan_no = _scan_no(scan_no)
    if isinstance(frame_index, bool) or not isinstance(frame_index, Integral):
        raise ValueError("frame_index must be a nonnegative integer")
    with h5py.File(path_dset, "r") as h5f:
        detector = _check_detector(h5f, scan_no, detector)
        data = h5f[f"{scan_no}/instrument/{detector}/data"]
        if data.ndim != 3:
            raise ValueError("Detector data must have shape (frames, rows, columns)")
        if not 0 <= frame_index < data.shape[0]:
            raise IndexError(f"frame_index must be between 0 and {data.shape[0] - 1}")
        rows, cols = data.shape[1:]
        if roi is None:
            roi = (0, rows, 0, cols)
        if len(roi) != 4 or not all(
            isinstance(i, Integral) and not isinstance(i, bool) for i in roi
        ):
            raise ValueError("ROI must contain four integer bounds")
        if not (0 <= roi[0] < roi[1] <= rows and 0 <= roi[2] < roi[3] <= cols):
            raise ValueError("ROI must lie within the detector bounds and be nonempty")
        return data[frame_index, roi[0] : roi[1], roi[2] : roi[3]]
