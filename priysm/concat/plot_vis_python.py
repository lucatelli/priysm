#!/usr/bin/env python3
"""
plot_vis_python.py
==================
Pure-Python / matplotlib replacements for plotms-based visibility plotting
in ph4ser.  Designed to be correct, memory-safe, and fast on large MSes.

Public API
----------
plot_visibilities(self, g_vis, name, ...)
    Amp vs UV-wave and amp vs frequency for DATA / CORRECTED_DATA /
    MODEL_DATA / CORRECTED-MODEL.  Matches the plotms calls in ph4ser.

plot_uvwave(self, g_vis, name)
    u-wave vs v-wave [klambda] and u vs v [metres], coloured by SPW.

plot_uv_coverage(vis_list, concat_ms=None, output_dir='.', ...)
    Multi-MS UV coverage comparison (adapted from concat_pipeline).

Design notes
------------
Averaging order (matches plotms):
    1.  Select single correlation (RR or fallback XX).
    2.  Mask flagged cells using FLAG and FLAG_ROW.
    3a. UV-wave plots:  nanmean unflagged channels (complex) per row
                        -> one complex per row; then accumulate per baseline
                        across all time; mean complex; |.|.
    3b. Freq plots:     accumulate complex per (baseline, channel) across all
                        time; mean complex; |.|.
    For 'corrected_data/model_data' both columns are accumulated separately;
    the ratio/difference is formed from their complex means - not row-by-row.

Memory
------
  - complex64 throughout (CASA native precision).
  - Chunked row reads via ms.iterinit() / ms.iternext().
  - Chunk size auto-tuned to stay within mem_fraction * available RAM.

Performance
-----------
  - Baseline accumulation is vectorised using sort+slice (no per-baseline
    Python loop scanning the full chunk).
  - Correlation axis is sliced immediately after reading, halving array size.

CASA tool singletons
--------------------
Functions have 'self' as first argument so they can be used as ph4ser
Pipeline methods directly.  They resolve tb / msmd / ms from the calling
namespace (ph4ser module scope) and fall back to module-level singletons
created at import time for standalone use.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import logging
import warnings
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

log = logging.getLogger(__name__)

_LIGHT_SPEED = 299792458.0   # m s-1
_NAN_C64     = np.complex64(complex(float('nan'), 0.0))

# ---------------------------------------------------------------------------
# CASA Stokes / correlation integers  (AIPS convention)
# ---------------------------------------------------------------------------
_CORR_STR_INT = {
    'RR':  5, 'RL':  6, 'LR':  7, 'LL':  8,
    'XX':  9, 'XY': 10, 'YX': 11, 'YY': 12,
}
_CORR_FALLBACK = {'RR': 'XX', 'LL': 'YY', 'XX': 'RR', 'YY': 'LL'}
_CORR_INT_STR  = {v: k for k, v in _CORR_STR_INT.items()}


# ===========================================================================
# Section 1 – Memory helpers
# ===========================================================================

def _get_available_memory_gb():
    """Return available system RAM in GB; None on failure."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except Exception:
        pass
    return None


def _chunk_rows(n_chan, mem_budget_gb, n_datacols=1, safety=3.0):
    """
    Estimate a safe number of MS rows per iteration chunk.

    Memory model per row (complex64 = 8 bytes, flag bool = 1 byte):
        n_chan * (8 + 1) bytes * n_datacols * safety

    safety=3.0 accounts for intermediate arrays created during processing.
    Returns at least 500 rows.
    """
    bytes_per_row = n_chan * 9.0 * n_datacols * safety
    budget        = max(0.5, float(mem_budget_gb)) * 1e9
    return max(500, int(budget / bytes_per_row))


# ===========================================================================
# Section 2 – MS metadata helpers
# ===========================================================================

def _existing_cols(vis, tb_tool):
    """Return set of column names present in the main MS table."""
    tb_tool.open(vis)
    cols = set(tb_tool.colnames())
    tb_tool.close()
    return cols


def _corr_idx(vis, corr_str, tb_tool):
    """
    Return the axis-0 index for *corr_str* in the DATA array.

    Falls back to the circular/linear counterpart (RR↔XX, LL↔YY),
    then to index 0 with a warning.
    """
    tb_tool.open(vis + '/POLARIZATION')
    try:
        corr_type = tb_tool.getcol('CORR_TYPE')   # (n_corr, n_pol_setups)
        tb_tool.close()
        ctypes = corr_type[:, 0].astype(int)
    except RuntimeError:
        # CORR_TYPE rows have different lengths (e.g. one row is empty);
        # getcol cannot pack them into a uniform array.  Read cell-by-cell.
        nrows = tb_tool.nrows()
        rows = [np.array(tb_tool.getcell('CORR_TYPE', r), dtype=int)
                for r in range(nrows)]
        tb_tool.close()
        non_empty = [r for r in rows if len(r) > 0]
        ctypes = non_empty[0] if non_empty else np.array([], dtype=int)
        # Prefer a setup that actually contains the requested correlation.
        target_int = _CORR_STR_INT.get(corr_str.upper())
        if target_int is not None:
            for row_ct in non_empty:
                if target_int in row_ct:
                    ctypes = row_ct
                    break

    target = _CORR_STR_INT.get(corr_str.upper())
    if target is not None and target in ctypes:
        return int(np.where(ctypes == target)[0][0])

    fb = _CORR_FALLBACK.get(corr_str.upper())
    if fb:
        fb_int = _CORR_STR_INT.get(fb)
        if fb_int is not None and fb_int in ctypes:
            idx = int(np.where(ctypes == fb_int)[0][0])
            log.warning("  Corr '%s' not found; using '%s' (idx=%d).",
                        corr_str, fb, idx)
            return idx

    log.warning("  Corr '%s' not found; defaulting to index 0.", corr_str)
    return 0


def _spw_meta(vis, tb_tool, msmd_tool):
    """
    Returns dd_info: {dd_id: {spw_id, n_chan, chan_freqs, mean_freq}}
    """
    tb_tool.open(vis + '/DATA_DESCRIPTION')
    spw_per_dd = list(tb_tool.getcol('SPECTRAL_WINDOW_ID'))
    tb_tool.close()

    msmd_tool.open(vis)
    dd_info = {}
    for dd_id, spw_id in enumerate(spw_per_dd):
        freqs = msmd_tool.chanfreqs(int(spw_id))
        dd_info[int(dd_id)] = {
            'spw_id':     int(spw_id),
            'n_chan':     int(len(freqs)),
            'chan_freqs': freqs,
            'mean_freq':  float(freqs.mean()),
        }
    msmd_tool.done()
    return dd_info


def _ant_names(vis, tb_tool):
    """Return list of antenna names."""
    tb_tool.open(vis + '/ANTENNA')
    names = list(tb_tool.getcol('NAME'))
    tb_tool.close()
    return names



# ===========================================================================
# Section 3 – Colour helpers
# ===========================================================================

def _ant1_colormap(ant1_arr, ant_names, cmap_name='twilight_shifted'):
    """
    Build colour dict keyed by antenna1 index.
    Much more readable than per-baseline colouring for large arrays.
    """
    unique_ants = sorted(set(int(a) for a in ant1_arr))
    try:
        cmap = plt.colormaps[cmap_name]
    except AttributeError:
        cmap = plt.cm.get_cmap(cmap_name)

    colors  = {}
    handles = []
    for i, ant in enumerate(unique_ants):
        col = cmap(i % cmap.N)
        colors[ant] = col
        label = ant_names[ant] if (ant_names and ant < len(ant_names)) else str(ant)
        handles.append(mpatches.Patch(color=col, label=label))
    return colors, handles, unique_ants


def _spw_colormap(dd_ids, dd_info, cmap_name='twilight_shifted'):
    """Colour dict keyed by dd_id."""
    try:
        cmap = plt.colormaps[cmap_name]
    except AttributeError:
        cmap = plt.cm.get_cmap(cmap_name)
    colors  = {dd: cmap(i % cmap.N) for i, dd in enumerate(dd_ids)}
    handles = [mpatches.Patch(color=colors[dd],
                               label='SPW {}'.format(dd_info[dd]['spw_id']))
               for dd in dd_ids]
    return colors, handles


# ===========================================================================
# Section 4 – SPW-by-SPW row iterator  (ms tool)
# ===========================================================================

# ms.getdata() key names (CASA modular, verified):
#   'data'           -> DATA column
#   'corrected_data' -> CORRECTED_DATA column
#   'model_data'     -> MODEL_DATA column
#   'flag'           -> FLAG
#   'flag_row'       -> FLAG_ROW
#   'uvw'            -> UVW
#   'antenna1'       -> ANTENNA1
#   'antenna2'       -> ANTENNA2
#   'scan_number'    -> SCAN_NUMBER
_DATA_KEYS = frozenset(['data', 'corrected_data', 'model_data'])


def _iter_spw_chunks(vis, ms_tool, dd_ids, chunk_rows, request_cols):
    """
    Iterate over *vis* rows one DATA_DESC_ID at a time in row chunks.

    Yields dict from ms.getdata() plus key 'dd_id'.

    CASA silently drops all-but-one data column when several _DATA_KEYS are
    requested together.  We work around this by requesting each data column
    in a separate getdata() call and merging results.
    """
    data_cols  = [c for c in request_cols if c in _DATA_KEYS]
    other_cols = [c for c in request_cols if c not in _DATA_KEYS]

    ms_tool.open(vis)
    try:
        for dd_id in dd_ids:
            # selectinit(datadescid=n) alone does NOT filter rows when all DDs
            # share the same shape (CASA 6.7 behaviour), so we use selecttaql
            # for the explicit row filter.  selectinit(datadescid=n, reset=False
            # [default]) is called afterwards to set the correct data shape for
            # this DD without clearing the TaQL selection - required for MSes
            # where SPWs have different channel counts.
            ms_tool.selectinit(reset=True)
            ms_tool.selecttaql(f'DATA_DESC_ID=={dd_id}')
            ms_tool.selectinit(datadescid=int(dd_id))   # shape only, no reset
            ms_tool.iterinit(maxrows=int(chunk_rows))
            more = ms_tool.iterorigin()
            while more:
                chunk = ms_tool.getdata(other_cols) if other_cols else {}
                for dc in data_cols:
                    chunk.update(ms_tool.getdata([dc]))
                chunk['dd_id'] = int(dd_id)
                yield chunk
                more = ms_tool.iternext()
    finally:
        ms_tool.close()


# ===========================================================================
# Section 5 – Vectorised baseline accumulator helper
# ===========================================================================

def _bl_sort_index(ant1, ant2, n_ant):
    """
    Encode (ant1, ant2) pairs as a single integer and return the sort order
    that groups rows by baseline.  Used to replace the O(n_bl × n_row) loop
    with an O(n_row log n_row) sort + O(n_row) slice.
    """
    bl_idx   = ant1.astype(np.int32) * n_ant + ant2.astype(np.int32)
    sort_ord = np.argsort(bl_idx, kind='stable')
    sorted_bl = bl_idx[sort_ord]
    unique_bl, starts, cnts = np.unique(sorted_bl,
                                        return_index=True,
                                        return_counts=True)
    return sort_ord, unique_bl, starts, cnts, n_ant


# ===========================================================================
# Section 6 – UV-wave amplitude accumulator
# ===========================================================================

def _collect_uvwave_amp(vis, ms_tool, tb_tool, msmd_tool,
                        datacol, ci, dd_info, chunk_rows, avgscan=False):
    """
    Channel+time average -> amplitude vs UV-wave distance.

    Replicates plotms avgchannel='9999', avgtime='9999', avgscan=False.

    For tuple datacol (col1, op, col2) both columns are accumulated
    separately; ratio/difference is computed from their complex means.

    Returns
    -------
    uvwave    : ndarray [klambda]  or None
    amp       : ndarray [Jy]
    ant1_out  : ndarray int  (antenna1 index, for colouring)
    unique_a1 : set of int
    """
    dd_ids   = sorted(dd_info.keys())
    is_tuple = isinstance(datacol, tuple)

    if is_tuple:
        col1, op, col2 = datacol
        req_data = [col1, col2]
        n_dc     = 2
    else:
        col1, op, col2 = datacol, None, None
        req_data = [datacol]
        n_dc     = 1

    request_cols = req_data + ['flag', 'flag_row', 'uvw',
                               'antenna1', 'antenna2', 'scan_number']

    # accum key: (dd_id, a1, a2, scan)  — one entry per baseline per scan,
    # matching plotms avgtime='9999', avgscan=False behaviour.
    #   sum1, sum2 : complex64 scalar
    #   count      : int
    #   uvw_sum    : float64[3]
    accum = {}

    # Estimate n_ant from ANTENNA subtable
    tb_tool.open(vis + '/ANTENNA')
    n_ant = int(tb_tool.nrows())
    tb_tool.close()

    for chunk in _iter_spw_chunks(vis, ms_tool, dd_ids, chunk_rows, request_cols):
        dd_id    = chunk['dd_id']
        flag     = chunk['flag']      # (n_corr, n_chan, n_rows)
        flag_row = chunk['flag_row']  # (n_rows,)
        uvw      = chunk['uvw']       # (3, n_rows)
        ant1     = chunk['antenna1'].astype(np.int32)
        ant2     = chunk['antenna2'].astype(np.int32)
        scan_num = chunk['scan_number'].astype(np.int32)
        n_rows   = ant1.size

        # Select correlation, apply per-channel + row flags -> (n_chan, n_rows)
        f_sel = flag[ci, :, :] | flag_row[np.newaxis, :]

        # Channel-average one column -> (n_rows,) complex64
        # Uses explicit sum/count to stay in complex64 and avoid RuntimeWarning
        # for all-flagged rows (which nanmean emits via Python warnings module).
        def _chan_avg(col_key):
            d = chunk[col_key][ci, :, :].astype(np.complex64)
            n_valid = (~f_sel).sum(axis=0, dtype=np.int32)          # (n_rows,)
            d_sum   = np.where(f_sel, np.complex64(0j), d).sum(axis=0)
            avg     = (d_sum / np.maximum(n_valid, 1).astype(
                           np.float32)).astype(np.complex64)
            avg[n_valid == 0] = _NAN_C64
            return avg

        cavg1 = _chan_avg(col1)
        cavg2 = _chan_avg(col2) if is_tuple else None

        # Valid rows: cross-correlation, unflagged, finite channel average
        good = ((ant1 != ant2)
                & (~flag_row)
                & np.isfinite(cavg1.real))
        if is_tuple:
            good = good & np.isfinite(cavg2.real)
        if not good.any():
            continue

        # ── Vectorised (scan, baseline) accumulation ─────────────────────
        g_ant1  = ant1[good]
        g_ant2  = ant2[good]
        g_scan  = scan_num[good]
        g_cavg1 = cavg1[good]
        g_cavg2 = cavg2[good] if is_tuple else None
        g_uvw   = uvw[:, good]          # (3, n_good)

        # Encode sort key as int64.  When avgscan=False (default) include the
        # scan number so each scan produces its own point per baseline.
        # When avgscan=True all scans collapse into one point per baseline.
        n_ant64 = np.int64(n_ant)
        if avgscan:
            scan_bl = (g_ant1.astype(np.int64) * n_ant64
                       + g_ant2.astype(np.int64))
        else:
            scan_bl = (g_scan.astype(np.int64) * n_ant64 * n_ant64
                       + g_ant1.astype(np.int64) * n_ant64
                       + g_ant2.astype(np.int64))
        sort_ord   = np.argsort(scan_bl, kind='stable')
        sorted_sbl = scan_bl[sort_ord]
        _, starts, cnts = np.unique(sorted_sbl, return_index=True,
                                    return_counts=True)

        s_ant1  = g_ant1[sort_ord]
        s_ant2  = g_ant2[sort_ord]
        s_scan  = g_scan[sort_ord]
        s_cavg1 = g_cavg1[sort_ord]
        s_cavg2 = g_cavg2[sort_ord] if is_tuple else None
        s_uvw   = g_uvw[:, sort_ord]

        for start, cnt in zip(starts, cnts):
            sl   = slice(start, start + cnt)
            a1v  = int(s_ant1[start])
            a2v  = int(s_ant2[start])
            snv  = int(s_scan[start]) if not avgscan else 0
            key  = (dd_id, a1v, a2v, snv)
            if key not in accum:
                accum[key] = {
                    'sum1':    np.complex64(0j),
                    'count':   0,
                    'uvw_sum': np.zeros(3, dtype=np.float64),
                }
                if is_tuple:
                    accum[key]['sum2'] = np.complex64(0j)
            accum[key]['sum1']    += s_cavg1[sl].sum()
            accum[key]['count']   += cnt
            accum[key]['uvw_sum'] += s_uvw[:, sl].sum(axis=1)
            if is_tuple:
                accum[key]['sum2'] += s_cavg2[sl].sum()

    if not accum:
        return None, None, None, set()

    uvwave_list = []
    amp_list    = []
    ant1_list   = []
    ant2_list   = []
    unique_a1   = set()

    for (dd_id, a1v, a2v, _scan), v in accum.items():
        cnt = v['count']
        if cnt == 0:
            continue

        mean1 = v['sum1'] / cnt

        if is_tuple:
            mean2 = v['sum2'] / cnt
            if op == '/':
                if abs(mean2) == 0.0:
                    continue
                amp = abs(mean1 / mean2)
            else:
                amp = abs(mean1 - mean2)
        else:
            amp = abs(mean1)

        if not np.isfinite(amp):
            continue

        mean_uvw  = v['uvw_sum'] / cnt
        mean_freq = dd_info[dd_id]['mean_freq']
        uv_m      = np.sqrt(mean_uvw[0]**2 + mean_uvw[1]**2)
        uvw_kl    = uv_m * mean_freq / _LIGHT_SPEED / 1e3

        uvwave_list.append(float(uvw_kl))
        amp_list.append(float(amp))
        ant1_list.append(a1v)
        ant2_list.append(a2v)
        unique_a1.add(a1v)

    if not uvwave_list:
        return None, None, None, None, unique_a1

    return (np.array(uvwave_list, dtype=np.float64),
            np.array(amp_list,   dtype=np.float64),
            np.array(ant1_list,  dtype=np.int32),
            np.array(ant2_list,  dtype=np.int32),
            unique_a1)


# ===========================================================================
# Section 7 – Frequency-amplitude accumulator  (time-averaged)
# ===========================================================================

def _accumulate_freq_amp(vis, ms_tool, tb_tool, msmd_tool,
                         datacol, ci, dd_info, chunk_rows):
    """
    Time-average complex visibilities per (dd_id, ant1, ant2, channel).

    Replicates plotms avgtime='9999'.

    For tuple datacol both columns use the same per-channel flag mask so
    their counts are consistent (fixes the flag-mismatch bug in the old code).

    Returns
    -------
    result : dict {(dd_id, a1, a2): {'freqs': ndarray [Hz], 'amp': ndarray}}
    ant1_all : set of int  (all ant1 values seen, for colouring)
    """
    dd_ids   = sorted(dd_info.keys())
    is_tuple = isinstance(datacol, tuple)

    if is_tuple:
        col1, op, col2 = datacol
        req_data = [col1, col2]
    else:
        col1, op, col2 = datacol, None, None
        req_data = [datacol]

    request_cols = req_data + ['flag', 'flag_row', 'antenna1', 'antenna2']

    # accum key: (dd_id, a1, a2)
    #   sum1, sum2: complex64[n_chan]
    #   count:      int32[n_chan]
    accum   = {}
    ant1_all = set()

    tb_tool.open(vis + '/ANTENNA')
    n_ant = int(tb_tool.nrows())
    tb_tool.close()

    for chunk in _iter_spw_chunks(vis, ms_tool, dd_ids, chunk_rows, request_cols):
        dd_id  = chunk['dd_id']
        n_chan = dd_info[dd_id]['n_chan']
        flag     = chunk['flag']       # (n_corr, n_chan, n_rows)
        flag_row = chunk['flag_row']   # (n_rows,)
        ant1     = chunk['antenna1'].astype(np.int32)
        ant2     = chunk['antenna2'].astype(np.int32)

        # Per-channel flag for selected correlation: (n_chan, n_rows)
        f_sel = flag[ci, :, :] | flag_row[np.newaxis, :]
        valid = (~f_sel)   # bool (n_chan, n_rows)

        # Extract data columns, zero flagged cells (safe summation)
        def _get(col_key):
            d = chunk[col_key][ci, :, :].astype(np.complex64)
            return np.where(f_sel, np.complex64(0j), d)   # (n_chan, n_rows)

        d1 = _get(col1)
        d2 = _get(col2) if is_tuple else None

        # Cross-correlations only
        cross = (ant1 != ant2) & (~flag_row)
        if not cross.any():
            continue

        # ── Vectorised baseline accumulation ─────────────────────────────
        g_ant1 = ant1[cross]
        g_ant2 = ant2[cross]
        d1_c   = d1[:, cross]        # (n_chan, n_good_rows)
        valid_c = valid[:, cross]    # (n_chan, n_good_rows)
        d2_c   = d2[:, cross] if is_tuple else None

        sort_ord, unique_bl, starts, cnts, _ = _bl_sort_index(
            g_ant1, g_ant2, n_ant)

        for start, cnt in zip(starts, cnts):
            sl  = slice(start, start + cnt)
            a1v = int(g_ant1[sort_ord[start]])
            a2v = int(g_ant2[sort_ord[start]])
            key = (dd_id, a1v, a2v)
            ant1_all.add(a1v)

            # (n_chan, cnt) slices - count only unflagged rows per channel
            d1_sl    = d1_c[:,    sort_ord[sl]]
            valid_sl = valid_c[:, sort_ord[sl]]

            if key not in accum:
                accum[key] = {
                    'sum1':  np.zeros(n_chan, dtype=np.complex64),
                    'count': np.zeros(n_chan, dtype=np.int32),
                }
                if is_tuple:
                    accum[key]['sum2'] = np.zeros(n_chan, dtype=np.complex64)

            accum[key]['sum1']  += d1_sl.sum(axis=1)
            accum[key]['count'] += valid_sl.sum(axis=1).astype(np.int32)
            if is_tuple:
                accum[key]['sum2'] += d2_c[:, sort_ord[sl]].sum(axis=1)

    result = {}
    for (dd_id, a1v, a2v), v in accum.items():
        cnt      = v['count']               # int32[n_chan]
        has_data = cnt > 0
        denom    = np.maximum(cnt, 1).astype(np.float32)

        mean1 = np.where(has_data, v['sum1'] / denom, _NAN_C64)

        if is_tuple:
            mean2 = np.where(has_data, v['sum2'] / denom, _NAN_C64)
            if op == '/':
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(np.abs(mean2) > 0,
                                     mean1 / mean2, np.nan + 0j)
                amp = np.abs(ratio)
            else:   # '-'
                amp = np.abs(mean1 - mean2)
        else:
            amp = np.abs(mean1)

        amp = np.where(has_data, amp, np.nan).astype(np.float32)
        result[(dd_id, a1v, a2v)] = {
            'freqs': dd_info[dd_id]['chan_freqs'],
            'amp':   amp,
        }
    return result, ant1_all


# ===========================================================================
# Section 8 – Plot renderers
# ===========================================================================

def _apply_plotrange(ax, plotrange):
    if plotrange is None:
        return
    xmin, xmax, ymin, ymax = plotrange
    if xmin != 0 or xmax != 0:
        ax.set_xlim(xmin, xmax)
    if ymin != 0 or ymax != 0:
        ax.set_ylim(ymin, ymax)


def _robust_ylim(amp_flat, percentile=99.5, pad=1.3):
    """
    Compute a robust y-axis upper limit for amplitude plots.

    Uses the given percentile of finite values to clip RFI/outlier spikes.
    Returns (0.0, ymax) where ymax = percentile_value * pad.

    percentile=99.5 discards the top 0.5% of points, which covers typical
    RFI outliers that are 10-100x brighter than the source signal.
    """
    finite = amp_flat[np.isfinite(amp_flat)]
    if finite.size == 0:
        return (0.0, 1.0)
    ymax = float(np.nanpercentile(finite, percentile)) * pad
    if ymax <= 0:
        ymax = float(finite.max()) * pad if finite.max() > 0 else 1.0
    ylo = -0.03 * ymax
    return (ylo, ymax)


def _ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


# def _render_uvwave_amp(uvwave, amp, ant1_arr, ant_colors,
#                        plotfile, title, plotrange=None,
#                        figsize=(16, 6), dpi=150):
#     fig, ax = plt.subplots(figsize=figsize)
#     for ant in np.unique(ant1_arr):
#         mask  = ant1_arr == ant
#         color = ant_colors.get(int(ant), 'grey')
#         ax.plot(uvwave[mask], amp[mask], '.',
#                 markersize=2, color=color, alpha=0.8,
#                 rasterized=True, linewidth=0)
#     _apply_plotrange(ax, plotrange)
#     ax.set_xlabel(r"$|uv|\;[\mathrm{k}\lambda]$")
#     ax.set_ylabel("Amplitude [Jy]")
#     ax.set_title(title)
#     ax.grid(True, lw=0.5, alpha=0.8)
#     _ensure_dir(plotfile)
#     fig.savefig(plotfile, dpi=dpi, bbox_inches='tight')
#     plt.close(fig)
#     log.info("  Saved: %s", plotfile)


# def _render_freq_amp(freq_amp, ant1_all, ant_colors,
#                      plotfile, title, plotrange=None,
#                      figsize=(16, 6), dpi=150):
#     fig, ax = plt.subplots(figsize=figsize)
#     for (dd_id, a1v, a2v), v in sorted(freq_amp.items()):
#         freqs_ghz = v['freqs'] / 1e9
#         amp       = v['amp']
#         color     = ant_colors.get(a1v, 'grey')
#         valid     = np.isfinite(amp)
#         if not valid.any():
#             continue
#         ax.plot(freqs_ghz[valid], amp[valid], '.',
#                 markersize=2, color=color, alpha=0.8,
#                 rasterized=True, linewidth=0)
#     _apply_plotrange(ax, plotrange)
#     ax.set_xlabel("Frequency [GHz]")
#     ax.set_ylabel("Amplitude [Jy]")
#     ax.set_title(title)
#     ax.grid(True, lw=0.5, alpha=0.8)
#     _ensure_dir(plotfile)
#     fig.savefig(plotfile, dpi=dpi, bbox_inches='tight')
#     plt.close(fig)
#     log.info("  Saved: %s", plotfile)


def _render_uvwave_amp(uvwave, amp, ant1_arr, ant2_arr, ant_colors,
                       plotfile, title, plotrange=None,
                       figsize=(16, 6), dpi=150):
    style = _adaptive_style(len(uvwave))
    fig, ax = plt.subplots(figsize=figsize)
    for (a1, a2), color in ant_colors.items():
        mask = (ant1_arr == a1) & (ant2_arr == a2)
        if not mask.any():
            continue
        ax.plot(uvwave[mask], amp[mask], '.',
                markersize=style['markersize'],
                color=color, alpha=style['alpha'],
                rasterized=True, linewidth=0)
    # Robust percentile auto-scale — skipped when an explicit y-range is set
    # (e.g. ratio plots with plotrange=[0,0,0,10]).
    if plotrange is None or (plotrange[2] == 0 and plotrange[3] == 0):
        ylo, yhi = _robust_ylim(amp)
        ax.set_ylim(ylo, yhi)
    _apply_plotrange(ax, plotrange)
    ax.set_xlabel(r"$|uv|\;[\mathrm{k}\lambda]$")
    ax.set_ylabel("Amplitude [Jy]")
    ax.set_title(title)
    ax.grid(True, lw=0.3, alpha=0.4)
    _ensure_dir(plotfile)
    fig.savefig(plotfile, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log.info("  Saved: %s", plotfile)


def _render_freq_amp(freq_amp, ant_colors,
                     plotfile, title, plotrange=None,
                     figsize=(16, 6), dpi=150):
    n_total = sum(np.isfinite(v['amp']).sum() for v in freq_amp.values())
    style   = _adaptive_style(n_total)
    fig, ax = plt.subplots(figsize=figsize)
    for (dd_id, a1v, a2v), v in sorted(freq_amp.items()):
        freqs_ghz = v['freqs'] / 1e9
        amp       = v['amp']
        valid     = np.isfinite(amp)
        if not valid.any():
            continue
        color = ant_colors.get((a1v, a2v), 'grey')
        ax.plot(freqs_ghz[valid], amp[valid], '.',
                markersize=style['markersize'],
                color=color, alpha=style['alpha'],
                rasterized=True, linewidth=0)
    # Robust percentile auto-scale — skipped when an explicit y-range is set
    # (e.g. ratio plots with plotrange=[0,0,0,10]).
    if plotrange is None or (plotrange[2] == 0 and plotrange[3] == 0):
        all_amp = np.concatenate([v['amp'][np.isfinite(v['amp'])]
                                  for v in freq_amp.values()])
        if all_amp.size > 0:
            ylo, yhi = _robust_ylim(all_amp)
            ax.set_ylim(ylo, yhi)
    _apply_plotrange(ax, plotrange)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Amplitude [Jy]")
    ax.set_title(title)
    ax.grid(True, lw=0.3, alpha=0.4)
    _ensure_dir(plotfile)
    fig.savefig(plotfile, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log.info("  Saved: %s", plotfile)

def _baseline_colormap(ant1_arr, ant2_arr, ant_names, cmap_name='twilight_shifted'):
    """
    Colour dict keyed by (ant1, ant2) tuple - one colour per unique baseline.
    Uses {cmap_name} so even 351 VLA baselines get maximally distinct hues.
    """
    pairs = sorted(set(zip(ant1_arr.tolist(), ant2_arr.tolist())))
    n     = max(len(pairs), 1)
    try:
        cmap = plt.colormaps[cmap_name]
    except AttributeError:
        cmap = plt.cm.get_cmap(cmap_name)

    colors  = {}
    handles = []
    for i, (a1, a2) in enumerate(pairs):
        col = cmap(i / n)
        colors[(a1, a2)] = col
        n1 = ant_names[a1] if (ant_names and a1 < len(ant_names)) else str(a1)
        n2 = ant_names[a2] if (ant_names and a2 < len(ant_names)) else str(a2)
        handles.append(mpatches.Patch(color=col, label=f'{n1}-{n2}'))
    return colors, handles

def _adaptive_style(n_points):
    """Return markersize and alpha scaled to data density."""
    if n_points < 500:
        return dict(markersize=8, alpha=0.95)
    if n_points < 5_000:
        return dict(markersize=6, alpha=0.95)
    if n_points < 50_000:
        return dict(markersize=4, alpha=0.95)
    return dict(markersize=2, alpha=0.95)


# def _blen_colormap(uvwave_arr, cmap_name='tab20'):
#     """
#     Colour each point by its own UV-wave distance (baseline length).
#     Returns per-point RGBA array and a ScalarMappable for the colorbar.
#     """
#     import matplotlib.cm as mcm
#     import matplotlib.colors as mcolors
#     vmin, vmax = uvwave_arr.min(), uvwave_arr.max()
#     if vmax == vmin:
#         vmax = vmin + 1.0
#     norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
#     try:
#         cmap = plt.colormaps[cmap_name]
#     except AttributeError:
#         cmap = mcm.get_cmap(cmap_name)
#     rgba = cmap(norm(uvwave_arr))          # (N, 4)
#     sm   = mcm.ScalarMappable(norm=norm, cmap=cmap)
#     sm.set_array([])
#     return rgba, sm


# def _render_uvwave_amp(uvwave, amp, ant1_arr, ant_colors,
#                        plotfile, title, plotrange=None,
#                        figsize=(16, 6), dpi=150,
#                        coloraxis='baseline_length'):
#     style = _adaptive_style(len(uvwave))
#     fig, ax = plt.subplots(figsize=figsize)

#     if coloraxis == 'baseline_length':
#         rgba, sm = _blen_colormap(uvwave)
#         ax.scatter(uvwave, amp, c=rgba,
#                    s=style['markersize'] ** 2,
#                    alpha=style['alpha'],
#                    rasterized=True, linewidths=0)
#         cb = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.85)
#         cb.set_label(r"$|uv|\;[\mathrm{k}\lambda]$", fontsize=9)
#     else:
#         # antenna1 colouring
#         for ant in np.unique(ant1_arr):
#             mask  = ant1_arr == ant
#             color = ant_colors.get(int(ant), 'grey')
#             ax.plot(uvwave[mask], amp[mask], '.',
#                     markersize=style['markersize'],
#                     color=color, alpha=style['alpha'],
#                     rasterized=True, linewidth=0)

#     _apply_plotrange(ax, plotrange)
#     ax.set_xlabel(r"$|uv|\;[\mathrm{k}\lambda]$")
#     ax.set_ylabel("Amplitude [Jy]")
#     ax.set_title(title)
#     ax.grid(True, lw=0.3, alpha=0.4)
#     _ensure_dir(plotfile)
#     fig.savefig(plotfile, dpi=dpi, bbox_inches='tight')
#     plt.close(fig)
#     log.info("  Saved: %s", plotfile)
    
# def _render_freq_amp(freq_amp, ant1_all, ant_colors, dd_info,
#                      plotfile, title, plotrange=None,
#                      figsize=(16, 6), dpi=150,
#                      coloraxis='spw'):
#     # Count total valid points for adaptive style
#     n_total = sum(
#         np.isfinite(v['amp']).sum() for v in freq_amp.values()
#     )
#     style = _adaptive_style(n_total)

#     # SPW colourmap (tab10 - distinct and readable for ≤10 SPWs,
#     # wraps gracefully for more)
#     dd_ids = sorted({dd_id for (dd_id, _, _) in freq_amp.keys()})
#     spw_colors, spw_handles = _spw_colormap(dd_ids, dd_info)

#     fig, ax = plt.subplots(figsize=figsize)
#     for (dd_id, a1v, a2v), v in sorted(freq_amp.items()):
#         freqs_ghz = v['freqs'] / 1e9
#         amp       = v['amp']
#         valid     = np.isfinite(amp)
#         if not valid.any():
#             continue
#         color = spw_colors[dd_id] if coloraxis == 'spw' else ant_colors.get(a1v, 'grey')
#         ax.plot(freqs_ghz[valid], amp[valid], '.',
#                 markersize=style['markersize'],
#                 color=color, alpha=style['alpha'],
#                 rasterized=True, linewidth=0)

#     # Legend: SPW entries (compact) or skip if too many
#     if coloraxis == 'spw' and len(spw_handles) <= 16:
#         ax.legend(handles=spw_handles, fontsize=7,
#                   loc='upper right', framealpha=0.7,
#                   ncol=max(1, len(spw_handles) // 8))

#     _apply_plotrange(ax, plotrange)
#     ax.set_xlabel("Frequency [GHz]")
#     ax.set_ylabel("Amplitude [Jy]")
#     ax.set_title(title)
#     ax.grid(True, lw=0.3, alpha=0.4)
#     _ensure_dir(plotfile)
#     fig.savefig(plotfile, dpi=dpi, bbox_inches='tight')
#     plt.close(fig)
#     log.info("  Saved: %s", plotfile)


# ===========================================================================
# Section 9 – plot_visibilities()
# ===========================================================================

# def plot_visibilities(self, g_vis, name,
#                       with_DATA=True, with_MODEL=False,
#                       with_CORRECTED=False, with_RESIDUAL=False,
#                       correlation='RR', mem_fraction=0.25):
def plot_visibilities(self, g_vis, name,
                      with_DATA=True, with_MODEL=False,
                      with_CORRECTED=False, with_RESIDUAL=False,
                      correlation='RR', mem_fraction=0.25, avgscan=False):
    """
    Pure-Python replacement for the plotms-based plot_visibilities() method.

    Produces the same output files as the original plotms version:
      <name>_uvwave_amp_data.jpg
      <name>_freq_amp_data.jpg
      <name>_uvwave_amp_corrected.jpg          (with_CORRECTED)
      <name>_freq_amp_corrected.jpg            (with_CORRECTED)
      <name>_uvwave_amp_corrected_div_model.jpg (with_CORRECTED + MODEL)
      <name>_uvwave_amp_model.jpg              (with_MODEL)
      <name>_freq_amp_model.jpg                (with_MODEL)
      <name>_uvwave_amp_corrected-model.jpg    (with_RESIDUAL)

    Parameters
    ----------
    correlation : str
        Preferred correlation to plot ('RR', 'XX', etc.).  Falls back to
        the circular/linear counterpart if the requested one is absent.
    mem_fraction : float
        Fraction of available RAM to budget for data arrays (default 0.25).
    """
    try:
        _tb   = tb      # noqa: F821
        _msmd = msmd    # noqa: F821
        _ms   = ms      # noqa: F821
    except NameError:
        _tb   = _TB_STANDALONE
        _msmd = _MSMD_STANDALONE
        _ms   = _MS_STANDALONE

    plot_dir = os.path.join(os.path.dirname(g_vis), 'selfcal', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # ── Inspect available columns ─────────────────────────────────────────
    existing      = _existing_cols(g_vis, _tb)
    has_corrected = 'CORRECTED_DATA' in existing
    has_model     = 'MODEL_DATA'     in existing

    # ── Correlation index ─────────────────────────────────────────────────
    ci = _corr_idx(g_vis, correlation, _tb)

    # ── SPW metadata ──────────────────────────────────────────────────────
    dd_info = _spw_meta(g_vis, _tb, _msmd)
    dd_ids  = sorted(dd_info.keys())

    # ── Antenna names (for legend) ────────────────────────────────────────
    ant_names = _ant_names(g_vis, _tb)

    # ── Chunk size ────────────────────────────────────────────────────────
    # Use the widest SPW so the chunk budget is safe for non-homogeneous MSes
    # (e.g. mixed 64-chan + 2048-chan windows or combined-instrument datasets).
    avail_gb  = _get_available_memory_gb() or 2.0
    n_chan_max = max(v['n_chan'] for v in dd_info.values())
    cr_single  = _chunk_rows(n_chan_max, avail_gb * mem_fraction, n_datacols=1)
    cr_double  = _chunk_rows(n_chan_max, avail_gb * mem_fraction, n_datacols=2)
    log.info("  n_chan_max=%d  chunk_rows(1col)=%d  chunk_rows(2col)=%d"
             "  avail_ram=%.1f GB", n_chan_max, cr_single, cr_double, avail_gb)


    # # ── Inner helper ──────────────────────────────────────────────────────
    # def _make_plots(datacol,
    #                 pf_uvwave, pf_freq,
    #                 title_uvwave, title_freq,
    #                 plotrange_uvwave=None, plotrange_freq=None):
    #     need_uv   = pf_uvwave and not os.path.isfile(pf_uvwave)
    #     need_freq = (pf_freq is not None) and not os.path.isfile(pf_freq)
    #     if not need_uv and not need_freq:
    #         return

    #     cr = cr_double if isinstance(datacol, tuple) else cr_single

    #     if need_uv:
    #         uvwave, amp, ant1_arr, unique_a1 = _collect_uvwave_amp(
    #             g_vis, _ms, _tb, _msmd, datacol, ci, dd_info, cr)
    #         if uvwave is not None and len(uvwave):
    #             ant_colors, handles, _ = _ant1_colormap(
    #                 ant1_arr, ant_names)
    #             _render_uvwave_amp(
    #                 uvwave, amp, ant1_arr, ant_colors,
    #                 pf_uvwave, title_uvwave,
    #                 plotrange=plotrange_uvwave)
    #         else:
    #             log.warning("  No valid data for UV-wave plot: %s", pf_uvwave)
    #         del uvwave, amp, ant1_arr

    #     if need_freq:
    #         freq_amp, ant1_all = _accumulate_freq_amp(
    #             g_vis, _ms, _tb, _msmd, datacol, ci, dd_info, cr)
    #         if freq_amp:
    #             all_a1 = np.array(sorted(ant1_all), dtype=np.int32)
    #             ant_colors, handles, _ = _ant1_colormap(all_a1, ant_names)
    #             _render_freq_amp(
    #                 freq_amp, ant1_all, ant_colors,
    #                 pf_freq, title_freq,
    #                 plotrange=plotrange_freq)
    #         else:
    #             log.warning("  No valid data for freq plot: %s", pf_freq)
    #         del freq_amp

    def _make_plots(datacol,
                        pf_uvwave, pf_freq,
                        title_uvwave, title_freq,
                        plotrange_uvwave=None, plotrange_freq=None):
            need_uv   = pf_uvwave and not os.path.isfile(pf_uvwave)
            need_freq = (pf_freq is not None) and not os.path.isfile(pf_freq)
            if not need_uv and not need_freq:
                return

            cr = cr_double if isinstance(datacol, tuple) else cr_single

            if need_uv:
                uvwave, amp, ant1_arr, ant2_arr, unique_a1 = _collect_uvwave_amp(
                    g_vis, _ms, _tb, _msmd, datacol, ci, dd_info, cr,
                    avgscan=avgscan)
                if uvwave is not None and len(uvwave):
                    bl_colors, _ = _baseline_colormap(ant1_arr, ant2_arr, ant_names)
                    _render_uvwave_amp(
                        uvwave, amp, ant1_arr, ant2_arr, bl_colors,
                        pf_uvwave, title_uvwave,
                        plotrange=plotrange_uvwave)
                else:
                    log.warning("  No valid data for UV-wave plot: %s", pf_uvwave)
                del uvwave, amp, ant1_arr, ant2_arr

            if need_freq:
                freq_amp, ant1_all = _accumulate_freq_amp(
                    g_vis, _ms, _tb, _msmd, datacol, ci, dd_info, cr)
                if freq_amp:
                    # Build baseline pairs from freq_amp keys
                    a1s = np.array([a1 for (_, a1, _) in freq_amp.keys()], dtype=np.int32)
                    a2s = np.array([a2 for (_, _, a2) in freq_amp.keys()], dtype=np.int32)
                    bl_colors, _ = _baseline_colormap(a1s, a2s, ant_names)
                    _render_freq_amp(
                        freq_amp, bl_colors,
                        pf_freq, title_freq,
                        plotrange=plotrange_freq)
                else:
                    log.warning("  No valid data for freq plot: %s", pf_freq)
                del freq_amp

    # def _make_plots(datacol,
    #                     pf_uvwave, pf_freq,
    #                     title_uvwave, title_freq,
    #                     plotrange_uvwave=None, plotrange_freq=None):
    #         need_uv   = pf_uvwave and not os.path.isfile(pf_uvwave)
    #         need_freq = (pf_freq is not None) and not os.path.isfile(pf_freq)
    #         if not need_uv and not need_freq:
    #             return

    #         cr = cr_double if isinstance(datacol, tuple) else cr_single

    #         if need_uv:
    #             uvwave, amp, ant1_arr, unique_a1 = _collect_uvwave_amp(
    #                 g_vis, _ms, _tb, _msmd, datacol, ci, dd_info, cr)
    #             if uvwave is not None and len(uvwave):
    #                 ant_colors, handles, _ = _ant1_colormap(ant1_arr, ant_names)
    #                 _render_uvwave_amp(
    #                     uvwave, amp, ant1_arr, ant_colors,
    #                     pf_uvwave, title_uvwave,
    #                     plotrange=plotrange_uvwave,
    #                     coloraxis=coloraxis_uvwave)
    #             else:
    #                 log.warning("  No valid data for UV-wave plot: %s", pf_uvwave)
    #             del uvwave, amp, ant1_arr

    #         if need_freq:
    #             freq_amp, ant1_all = _accumulate_freq_amp(
    #                 g_vis, _ms, _tb, _msmd, datacol, ci, dd_info, cr)
    #             if freq_amp:
    #                 all_a1 = np.array(sorted(ant1_all), dtype=np.int32)
    #                 ant_colors, handles, _ = _ant1_colormap(all_a1, ant_names)
    #                 _render_freq_amp(
    #                     freq_amp, ant1_all, ant_colors, dd_info,
    #                     pf_freq, title_freq,
    #                     plotrange=plotrange_freq,
    #                     coloraxis=coloraxis_freq)
    #             else:
    #                 log.warning("  No valid data for freq plot: %s", pf_freq)
    #             del freq_amp

    # ── DATA ──────────────────────────────────────────────────────────────
    if with_DATA:
        _make_plots(
            datacol='data',
            pf_uvwave=os.path.join(plot_dir, name + '_uvwave_amp_data.jpg'),
            pf_freq=os.path.join(plot_dir,   name + '_freq_amp_data.jpg'),
            title_uvwave=name + '  |  DATA  |  amp vs UV-wave',
            title_freq=name   + '  |  DATA  |  amp vs frequency',
            plotrange_uvwave=[0, 0, 0, 0],
            plotrange_freq=[0, 0, 0, 0],
        )
        if has_model:
            _make_plots(
                datacol=('data', '/', 'model_data'),
                pf_uvwave=os.path.join(
                    plot_dir, name + '_uvwave_amp_data_div_model.jpg'),
                pf_freq=None,
                title_uvwave=name + '  |  DATA/MODEL  |  amp vs UV-wave',
                title_freq=None,
                plotrange_uvwave=[0, 0, 0, 10],
            )
        else:
            log.warning("  MODEL_DATA absent; skipping data/model plot.")

    # ── CORRECTED ─────────────────────────────────────────────────────────
    if with_CORRECTED:
        if not has_corrected:
            log.warning("  CORRECTED_DATA absent in %s; skipping.", g_vis)
        else:
            if has_model:
                _make_plots(
                    datacol=('corrected_data', '/', 'model_data'),
                    pf_uvwave=os.path.join(
                        plot_dir, name + '_uvwave_amp_corrected_div_model.jpg'),
                    pf_freq=None,
                    title_uvwave=name + '  |  CORRECTED/MODEL  |  amp vs UV-wave',
                    title_freq=None,
                    plotrange_uvwave=[0, 0, 0, 10],
                )
            else:
                log.warning("  MODEL_DATA absent; skipping corrected/model plot.")

            _make_plots(
                datacol='corrected_data',
                pf_uvwave=os.path.join(
                    plot_dir, name + '_uvwave_amp_corrected.jpg'),
                pf_freq=os.path.join(
                    plot_dir, name + '_freq_amp_corrected.jpg'),
                title_uvwave=name + '  |  CORRECTED  |  amp vs UV-wave',
                title_freq=name   + '  |  CORRECTED  |  amp vs frequency',
                plotrange_uvwave=[0, 0, 0, 0],
                plotrange_freq=[0, 0, 0, 0],
            )

    # ── MODEL ─────────────────────────────────────────────────────────────
    if with_MODEL:
        if not has_model:
            log.warning("  MODEL_DATA absent in %s; skipping.", g_vis)
        else:
            _make_plots(
                datacol='model_data',
                pf_uvwave=os.path.join(
                    plot_dir, name + '_uvwave_amp_model.jpg'),
                pf_freq=os.path.join(
                    plot_dir, name + '_freq_amp_model.jpg'),
                title_uvwave=name + '  |  MODEL  |  amp vs UV-wave',
                title_freq=name   + '  |  MODEL  |  amp vs frequency',
                plotrange_uvwave=[0, 0, 0, 0],
                plotrange_freq=[0, 0, 0, 0],
            )

    # ── RESIDUAL ──────────────────────────────────────────────────────────
    if with_RESIDUAL:
        if not has_corrected or not has_model:
            log.warning("  CORRECTED_DATA or MODEL_DATA absent; "
                        "skipping residual plot.")
        else:
            _make_plots(
                datacol=('corrected_data', '-', 'model_data'),
                pf_uvwave=os.path.join(
                    plot_dir, name + '_uvwave_amp_corrected-model.jpg'),
                pf_freq=None,
                title_uvwave=name + '  |  CORRECTED-MODEL  |  amp vs UV-wave',
                title_freq=None,
                plotrange_uvwave=[0, 0, 0, 0],
            )


# ===========================================================================
# Section 10 – plot_uvwave()   (SPW-coloured uv coverage)
# ===========================================================================

# def plot_uvwave(self, g_vis, name, mem_fraction=0.10):
#     """
#     u-wave vs v-wave [klambda] and u vs v [metres], coloured by SPW.

#     Matches the plotms avgchannel='16' representation by grouping
#     channels into 16-channel chunks for the frequency-scaled plot.
#     """
#     try:
#         _tb   = tb      # noqa: F821
#         _msmd = msmd    # noqa: F821
#         _ms   = ms      # noqa: F821
#     except NameError:
#         _tb   = _TB_STANDALONE
#         _msmd = _MSMD_STANDALONE
#         _ms   = _MS_STANDALONE

#     plot_dir = os.path.join(os.path.dirname(g_vis), 'selfcal', 'plots')
#     os.makedirs(plot_dir, exist_ok=True)

#     pf_wave = os.path.join(plot_dir, name + '_uvwave.jpg')
#     pf_uv   = os.path.join(plot_dir, name + '_uv.jpg')

#     if os.path.isfile(pf_wave) and os.path.isfile(pf_uv):
#         log.info("  Both uv plots already exist; skipping.")
#         return

#     dd_info = _spw_meta(g_vis, _tb, _msmd)
#     dd_ids  = sorted(dd_info.keys())

#     avail_gb   = _get_available_memory_gb() or 2.0
#     # UVW only - very light; generous chunk
#     chunk_rows = min(500_000, max(10_000,
#                                   int(avail_gb * mem_fraction * 1e9 / (3 * 8))))

#     spw_colors, spw_handles = _spw_colormap(dd_ids, dd_info)

#     uwave_parts = {dd: [] for dd in dd_ids}
#     vwave_parts = {dd: [] for dd in dd_ids}
#     u_m_parts   = {dd: [] for dd in dd_ids}
#     v_m_parts   = {dd: [] for dd in dd_ids}

#     for chunk in _iter_spw_chunks(g_vis, _ms, dd_ids, chunk_rows,
#                                    ['uvw', 'antenna1', 'antenna2', 'flag_row']):
#         dd_id    = chunk['dd_id']
#         uvw      = chunk['uvw']       # (3, n_rows)
#         ant1     = chunk['antenna1']
#         ant2     = chunk['antenna2']
#         flag_row = chunk['flag_row']

#         good = (ant1 != ant2) & (~flag_row)
#         if not good.any():
#             continue

#         u = uvw[0, good]
#         v = uvw[1, good]

#         u_m_parts[dd_id].append(u.copy())
#         v_m_parts[dd_id].append(v.copy())

#         # Freq-scaled: 16-channel chunks to represent bandwidth smearing
#         freqs   = dd_info[dd_id]['chan_freqs']
#         cs      = 16
#         n_freq  = len(freqs)
#         n_c     = max(1, n_freq // cs)
#         cf      = freqs[:n_c * cs].reshape(n_c, cs).mean(axis=1)   # (n_c,) Hz

#         wl      = (_LIGHT_SPEED / cf).reshape(-1, 1)   # (n_c, 1) m
#         u_kl    = np.tile(np.hstack([u, -u]), (n_c, 1)) / wl / 1e3
#         v_kl    = np.tile(np.hstack([v, -v]), (n_c, 1)) / wl / 1e3
#         uwave_parts[dd_id].append(u_kl.ravel())
#         vwave_parts[dd_id].append(v_kl.ravel())

#     # ── Plot 1: uwave vs vwave ────────────────────────────────────────────
#     if not os.path.isfile(pf_wave):
#         fig, ax = plt.subplots(figsize=(8, 8))
#         for dd_id in dd_ids:
#             parts = uwave_parts[dd_id]
#             if not parts:
#                 continue
#             uw = np.concatenate(parts)
#             vw = np.concatenate(vwave_parts[dd_id])
#             ax.plot(uw, vw, '.', markersize=0.5,
#                     color=spw_colors[dd_id], alpha=0.5,
#                     rasterized=True, linewidth=0)
#         ax.legend(handles=spw_handles, fontsize=7,
#                   loc='upper right', framealpha=0.7)
#         ax.set_xlabel(r"$u\;[\mathrm{k}\lambda]$")
#         ax.set_ylabel(r"$v\;[\mathrm{k}\lambda]$")
#         ax.set_aspect('equal', adjustable='datalim')
#         ax.set_title("{} | uv coverage (wavelength-scaled)".format(name))
#         ax.grid(True, lw=0.3, alpha=0.4)
#         fig.savefig(pf_wave, dpi=150, bbox_inches='tight')
#         plt.close(fig)
#         log.info("  Saved: %s", pf_wave)
#         for dd in dd_ids:
#             uwave_parts[dd].clear()
#             vwave_parts[dd].clear()

#     # ── Plot 2: u vs v [metres] ───────────────────────────────────────────
#     if not os.path.isfile(pf_uv):
#         fig, ax = plt.subplots(figsize=(8, 8))
#         for dd_id in dd_ids:
#             parts = u_m_parts[dd_id]
#             if not parts:
#                 continue
#             u = np.concatenate(parts)
#             v = np.concatenate(v_m_parts[dd_id])
#             ax.plot(np.hstack([u, -u]), np.hstack([v, -v]),
#                     '.', markersize=0.3,
#                     color=spw_colors[dd_id], alpha=0.5,
#                     rasterized=True, linewidth=0)
#         ax.legend(handles=spw_handles, fontsize=7,
#                   loc='upper right', framealpha=0.7)
#         ax.set_xlabel("u [m]")
#         ax.set_ylabel("v [m]")
#         ax.set_aspect('equal', adjustable='datalim')
#         ax.set_title("{} | uv coverage (metric)".format(name))
#         ax.grid(True, lw=0.3, alpha=0.4)
#         fig.savefig(pf_uv, dpi=150, bbox_inches='tight')
#         plt.close(fig)
#         log.info("  Saved: %s", pf_uv)


# ===========================================================================
# Section 10 – plot_uvwave()   (baseline-coloured uv coverage)
# ===========================================================================

def plot_uvwave(self, g_vis, name, mem_fraction=0.10, cmap_name='twilight_shifted'):
    """
    u-wave vs v-wave [klambda] and u vs v [metres], coloured by baseline.

    Baseline colour uses _baseline_colormap (gist_rainbow by default).
    The wavelength-scaled plot groups channels into 16-channel chunks to
    represent bandwidth smearing, matching plotms avgchannel='16'.
    No legend is shown (too many baselines); colour identity is by baseline.
    """
    try:
        _tb   = tb      # noqa: F821
        _msmd = msmd    # noqa: F821
        _ms   = ms      # noqa: F821
    except NameError:
        _tb   = _TB_STANDALONE
        _msmd = _MSMD_STANDALONE
        _ms   = _MS_STANDALONE

    plot_dir = os.path.join(os.path.dirname(g_vis), 'selfcal', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    pf_wave = os.path.join(plot_dir, name + '_uvwave.jpg')
    pf_uv   = os.path.join(plot_dir, name + '_uv.jpg')

    if os.path.isfile(pf_wave) and os.path.isfile(pf_uv):
        log.info("  Both uv plots already exist; skipping.")
        return

    dd_info   = _spw_meta(g_vis, _tb, _msmd)
    dd_ids    = sorted(dd_info.keys())
    ant_names = _ant_names(g_vis, _tb)

    _tb.open(g_vis + '/ANTENNA')
    n_ant = int(_tb.nrows())
    _tb.close()

    avail_gb   = _get_available_memory_gb() or 2.0
    chunk_rows = min(500_000, max(10_000,
                                  int(avail_gb * mem_fraction * 1e9 / (3 * 8))))

    # Per-baseline accumulators - keyed by (ant1, ant2)
    # bl_u / bl_v : lists of u/v arrays in metres  (for metric plot)
    # bl_ukl / bl_vkl : lists of u/v arrays in klambda (for wave-scaled plot)
    bl_u   = {}
    bl_v   = {}
    bl_ukl = {}
    bl_vkl = {}

    for chunk in _iter_spw_chunks(g_vis, _ms, dd_ids, chunk_rows,
                                   ['uvw', 'antenna1', 'antenna2', 'flag_row']):
        dd_id    = chunk['dd_id']
        uvw      = chunk['uvw']        # (3, n_rows)
        ant1     = chunk['antenna1'].astype(np.int32)
        ant2     = chunk['antenna2'].astype(np.int32)
        flag_row = chunk['flag_row']

        good = (ant1 != ant2) & (~flag_row)
        if not good.any():
            continue

        g_u   = uvw[0, good]
        g_v   = uvw[1, good]
        g_a1  = ant1[good]
        g_a2  = ant2[good]

        # Frequency chunks for wavelength-scaled plot
        freqs  = dd_info[dd_id]['chan_freqs']
        cs     = 16
        n_freq = len(freqs)
        n_c    = max(1, n_freq // cs)
        cf     = freqs[:n_c * cs].reshape(n_c, cs).mean(axis=1)   # (n_c,) Hz
        wl     = (_LIGHT_SPEED / cf).reshape(-1, 1)                # (n_c, 1) m

        # Group rows by baseline using vectorised sort
        sort_ord, _, starts, cnts, _ = _bl_sort_index(g_a1, g_a2, n_ant)

        s_u  = g_u[sort_ord]
        s_v  = g_v[sort_ord]
        s_a1 = g_a1[sort_ord]
        s_a2 = g_a2[sort_ord]

        for start, cnt in zip(starts, cnts):
            sl  = slice(start, start + cnt)
            key = (int(s_a1[start]), int(s_a2[start]))

            u_bl = s_u[sl]   # (cnt,)
            v_bl = s_v[sl]   # (cnt,)

            if key not in bl_u:
                bl_u[key]   = []
                bl_v[key]   = []
                bl_ukl[key] = []
                bl_vkl[key] = []

            bl_u[key].append(u_bl)
            bl_v[key].append(v_bl)

            # Wavelength-scaled: tile each baseline's rows across freq chunks
            # Result shape: (n_c, 2*cnt) - include conjugate
            uv_both = np.hstack([u_bl, -u_bl])   # (2*cnt,)
            vv_both = np.hstack([v_bl, -v_bl])
            bl_ukl[key].append((np.tile(uv_both, (n_c, 1)) / wl / 1e3).ravel())
            bl_vkl[key].append((np.tile(vv_both, (n_c, 1)) / wl / 1e3).ravel())

    if not bl_u:
        log.warning("  No unflagged cross-correlation rows found in %s", g_vis)
        return

    # Build baseline colourmap from all baselines seen
    all_a1 = np.array([a1 for (a1, _) in bl_u.keys()], dtype=np.int32)
    all_a2 = np.array([a2 for (_, a2) in bl_u.keys()], dtype=np.int32)
    bl_colors, _ = _baseline_colormap(all_a1, all_a2, ant_names,
                                       cmap_name=cmap_name)

    n_bl    = len(bl_u)
    n_total = sum(sum(len(a) for a in arrs) for arrs in bl_u.values())
    # style   = _adaptive_style(n_total)
    style = dict(markersize=0.1, alpha=0.3)
    ms_uv   = max(0.1, style['markersize'] * 0.5)   # slightly smaller for coverage plots
    al_uv   = min(0.6, style['alpha'] + 0.1)

    # ── Plot 1: wavelength-scaled ─────────────────────────────────────────
    if not os.path.isfile(pf_wave):
        fig, ax = plt.subplots(figsize=(8, 8))
        for key, color in bl_colors.items():
            if key not in bl_ukl or not bl_ukl[key]:
                continue
            uw = np.concatenate(bl_ukl[key])
            vw = np.concatenate(bl_vkl[key])
            ax.plot(uw, vw, '.', markersize=ms_uv,
                    color=color, alpha=al_uv,
                    rasterized=True, linewidth=0)
        ax.set_xlabel(r"$u\;[\mathrm{k}\lambda]$")
        ax.set_ylabel(r"$v\;[\mathrm{k}\lambda]$")
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title("{} | uv coverage (wavelength-scaled) - {} baselines".format(
            name, n_bl))
        ax.grid(True, lw=0.3, alpha=0.4)
        fig.savefig(pf_wave, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log.info("  Saved: %s", pf_wave)

    # ── Plot 2: metric ────────────────────────────────────────────────────
    if not os.path.isfile(pf_uv):
        fig, ax = plt.subplots(figsize=(8, 8))
        for key, color in bl_colors.items():
            if key not in bl_u or not bl_u[key]:
                continue
            u = np.concatenate(bl_u[key])
            v = np.concatenate(bl_v[key])
            ax.plot(np.hstack([u, -u]), np.hstack([v, -v]),
                    '.', markersize=ms_uv,
                    color=color, alpha=al_uv,
                    rasterized=True, linewidth=0)
        ax.set_xlabel("u [m]")
        ax.set_ylabel("v [m]")
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title("{} | uv coverage (metric) - {} baselines".format(
            name, n_bl))
        ax.grid(True, lw=0.3, alpha=0.4)
        fig.savefig(pf_uv, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log.info("  Saved: %s", pf_uv)

# ===========================================================================
# Section 11 – plot_uv_coverage()   (multi-MS comparison)
# ===========================================================================
# Adapted directly from concat_vis.py, with the auto-sizing logic
# from _auto_plot_params and the _plot_one inner function.

def plot_uv_coverage(vis_list, concat_ms=None, output_dir='.',
                     chunk_size=None, downsample_factor=None,
                     mem_fraction=0.5):
    """
    UV coverage comparison for a list of MSes, optionally with the
    concatenated MS overlaid in the background.

    Adapted from concat_pipeline.plot_uv_comparison().

    Parameters
    ----------
    vis_list         : list of str   individual MS paths
    concat_ms        : str or None   combined MS (plotted in background)
    output_dir       : str           where to save the PNG
    chunk_size       : int or None   frequency channels per chunk (auto)
    downsample_factor: int or None   point thinning factor (auto)
    mem_fraction     : float         fraction of available RAM to budget
    """
    try:
        _tb   = tb      # noqa: F821
        _msmd = msmd    # noqa: F821
    except NameError:
        _tb   = _TB_STANDALONE
        _msmd = _MSMD_STANDALONE

    os.makedirs(output_dir, exist_ok=True)

    avail_gb = _get_available_memory_gb() or 4.0
    target_gb = mem_fraction * avail_gb

    def _auto_params(vis):
        """Estimate chunk_size and downsample_factor for one MS."""
        try:
            _tb.open(vis)
            n_rows = _tb.nrows()
            _tb.close()
            _msmd.open(vis)
            nspw   = _msmd.nspw()
            n_freq = sum(len(_msmd.chanfreqs(s)) for s in range(nspw))
            _msmd.done()
        except Exception:
            try:
                _tb.close()
            except Exception:
                pass
            try:
                _msmd.done()
            except Exception:
                pass
            return 4, 150

        # Memory model: tile = n_chunks × 2×n_rows × 2 × 8 bytes
        tile_budget = target_gb * 1e9 / 2.0
        bpr         = 2 * 2 * 8
        max_chunks  = max(1, int(tile_budget / (n_rows * bpr)))
        cs          = max(1, int(np.ceil(n_freq / max_chunks)))
        n_chunks    = max(1, n_freq // cs)
        raw_points  = n_chunks * 2 * n_rows
        ds          = max(1, int(np.ceil(raw_points / 2_000_000)))
        return cs, ds

    all_vis = ([concat_ms] if (concat_ms and os.path.exists(concat_ms)) else []) \
              + list(vis_list)

    auto_cs, auto_ds = chunk_size, downsample_factor
    if auto_cs is None or auto_ds is None:
        worst_cs, worst_ds = 4, 150
        for v in all_vis:
            cs_v, ds_v = _auto_params(v)
            worst_cs = max(worst_cs, cs_v)
            worst_ds = max(worst_ds, ds_v)
        if auto_cs is None:
            auto_cs = worst_cs
        if auto_ds is None:
            auto_ds = worst_ds

    try:
        cmap = matplotlib.colormaps['twilight_shifted']
    except AttributeError:
        cmap = matplotlib.cm.get_cmap('twilight_shifted')
    colors = [cmap(i % 10) for i in range(len(vis_list))]

    fig, ax = plt.subplots(figsize=(8, 8))

    def _plot_one(vis, color, alpha=0.65):
        try:
            _msmd.open(vis)
            nspw      = _msmd.nspw()
            all_freqs = np.concatenate([_msmd.chanfreqs(s) for s in range(nspw)])
            _msmd.done()
        except Exception:
            try:
                _msmd.done()
            except Exception:
                pass
            return
        try:
            _tb.open(vis)
            uvw  = _tb.getcol('UVW')
            ant1 = _tb.getcol('ANTENNA1')
            ant2 = _tb.getcol('ANTENNA2')
            _tb.close()
        except Exception:
            try:
                _tb.close()
            except Exception:
                pass
            return

        n_freq    = len(all_freqs)
        cs        = max(1, auto_cs)
        n_chunks  = max(1, n_freq // cs)
        freq_used = all_freqs[:n_chunks * cs].reshape(n_chunks, cs)
        chunk_frq = freq_used.mean(axis=1)
        wavelens  = (_LIGHT_SPEED / chunk_frq).reshape(-1, 1)

        ant_uniq = np.unique(np.concatenate([ant1, ant2]))
        for a in ant_uniq:
            for b in ant_uniq:
                if b <= a:
                    continue
                idx = np.where((ant1 == a) & (ant2 == b))[0]
                if len(idx) == 0:
                    continue
                u = uvw[0, idx]
                v = uvw[1, idx]
                u_kl = (np.tile(np.hstack([u, -u]), (n_chunks, 1))
                        / wavelens / 1e3)
                v_kl = (np.tile(np.hstack([v, -v]), (n_chunks, 1))
                        / wavelens / 1e3)
                ax.plot(u_kl[:, ::auto_ds], v_kl[:, ::auto_ds],
                        '.', markersize=0.2, color=color, alpha=alpha,
                        rasterized=True)

    if concat_ms and os.path.exists(concat_ms):
        log.info("  Plotting concat MS (background): %s",
                 os.path.basename(concat_ms))
        _plot_one(concat_ms, color='silver', alpha=0.35)

    for vis, col in zip(vis_list, colors):
        log.info("  Plotting: %s", os.path.basename(vis))
        _plot_one(vis, col)

    handles = []
    if concat_ms and os.path.exists(concat_ms):
        handles.append(
            mlines.Line2D([], [], color='silver', marker='.', linestyle='None',
                          markersize=6,
                          label='concat: ' + os.path.basename(concat_ms)))
    handles += [
        mlines.Line2D([], [], color=colors[i], marker='.', linestyle='None',
                      markersize=6, label=os.path.basename(vis_list[i]))
        for i in range(len(vis_list))
    ]
    ax.legend(handles=handles, fontsize=7, loc='upper right', framealpha=0.7)
    ax.set_xlabel(r"$u\;[\mathrm{k}\lambda]$")
    ax.set_ylabel(r"$v\;[\mathrm{k}\lambda]$")
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_title(r"Total $uv$ coverage")

    base = (os.path.basename(concat_ms).replace('.ms', '')
            if concat_ms else 'uv_coverage')
    fig_path = os.path.join(output_dir,
                            base + '_uv_coverage_total.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("  Saved: %s", fig_path)
    return fig_path


# ===========================================================================
# Section 12 – Standalone CASA tool singletons (outside ph4ser namespace)
# ===========================================================================

try:
    import casatools as _ct
    _TB_STANDALONE   = _ct.table()
    _MSMD_STANDALONE = _ct.msmetadata()
    _MS_STANDALONE   = _ct.ms()
except Exception:
    _TB_STANDALONE   = None
    _MSMD_STANDALONE = None
    _MS_STANDALONE   = None
