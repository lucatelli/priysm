#!/usr/bin/env python3
"""
vis_imaging.py
===============
Pipeline for multi-instrument interferometric visibility processing.

Stages
------
  1. Phase-shift visibilities to a common phase centre  [--do-phaseshift]
  2. Match UV coverage and split to common range         [--do-uv-match]
  3. Native-resolution imaging  (wsclean via mlibs)      [--do-native-imaging]
  4. UV-matched imaging         (wsclean via mlibs)      [--do-uv-matched-imaging]

Each stage is optional and independently switchable.  The pipeline tracks a
single "active" visibility list that is updated after every stage that runs,
so downstream stages always operate on the most-processed data available.

Usage examples
--------------
  # Phaseshift + UV match, all outputs in a dedicated directory
  python vis_imaging.py \
      --vis A.ms B.ms C.ms \
      --ref-vis ref.ms \
      --do-phaseshift --do-uv-match \
      --output-dir ./processed

  # UV match + native imaging, explicit cell size and two robust values
  python vis_imaging.py \
      --vis A.ms B.ms C.ms \
      --ref-vis ref.ms \
      --do-uv-match --do-native-imaging \
      --cell 0.04arcsec --robust -0.5 0.5 \
      --output-dir ./images

  # Full run, auto cell size, overwrite previous outputs
  python vis_imaging.py \
      --vis A.ms B.ms C.ms \
      --ref-vis ref.ms \
      --do-phaseshift --do-uv-match \
      --do-native-imaging --do-uv-matched-imaging \
      --output-dir ./run01 --force-overwrite \
      --mlibs-path /path/to/morphen/morphen

Author: vis_imaging
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import logging
import os
import shutil
import sys
from matplotlib import use as mpluse
mpluse('Agg')

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Third-party / science
# ---------------------------------------------------------------------------
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CASA (modular installation assumed)
# ---------------------------------------------------------------------------
try:
    import casatools
    import casatasks
    from casaplotms import plotms
    msmd = casatools.msmetadata()
    ms   = casatools.ms()
    tb   = casatools.table()
except ImportError as _casa_err:
    print(
        "[ERROR] Could not import casatools / casatasks / casaplotms.\n"
        "        Make sure you are running inside a modular CASA environment.\n"
        "        Details: {}".format(_casa_err)
    )
    sys.exit(1)


# ===========================================================================
# Logging setup
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vis_imaging")


def separator(char="=", width=72):
    log.info(char * width)


def section(title, char="-", width=72):
    separator(char, width)
    log.info("  %s", title)
    separator(char, width)


# ===========================================================================
# Helpers: spectral window and UV utilities
# ===========================================================================

def get_spw_freq_map(vis):
    """
    Return a dict keyed by SPW id with frequency metadata for every SPW in *vis*.

    Each entry contains:
        freqs  : np.ndarray  channel centre frequencies  [Hz]
        fmin   : float       minimum channel frequency   [Hz]
        fmax   : float       maximum channel frequency   [Hz]
        fmean  : float       mean channel frequency      [Hz]
        nchan  : int         number of channels
        bw     : float       total bandwidth             [Hz]
    """
    msmd.open(vis)
    try:
        nspw   = msmd.nspw()
        result = {}
        for spw_id in range(nspw):
            freqs = msmd.chanfreqs(spw_id)
            bws   = msmd.chanwidths(spw_id)
            result[spw_id] = {
                "freqs" : freqs,
                "fmin"  : float(np.min(freqs)),
                "fmax"  : float(np.max(freqs)),
                "fmean" : float(np.mean(freqs)),
                "nchan" : len(freqs),
                "bw"    : float(np.sum(np.abs(bws))),
            }
    finally:
        msmd.done()
    return result


def log_spw_info(vis):
    """
    Log a per-SPW frequency table for *vis*, including gap detection.
    Returns the spw_map dict.
    """
    spw_map = get_spw_freq_map(vis)
    log.info("  SPW info for: %s", os.path.basename(vis))

    header = (
        "    {:<5}  {:<7}  {:<14}  {:<14}  {:<14}  {:<10}  {:<10}"
    ).format("SPW", "Nchan", "Fmin [GHz]", "Fmax [GHz]", "Fmean [GHz]",
             "BW [MHz]", "ChanW [kHz]")
    log.info(header)
    log.info("    " + "-" * 68)

    for spw_id, info in sorted(spw_map.items()):
        chan_width_khz = info["bw"] / info["nchan"] / 1e3
        log.info(
            "    {:<5d}  {:<7d}  {:<14.6f}  {:<14.6f}  {:<14.6f}"
            "  {:<10.3f}  {:<10.3f}".format(
                spw_id,
                info["nchan"],
                info["fmin"]  / 1e9,
                info["fmax"]  / 1e9,
                info["fmean"] / 1e9,
                info["bw"]    / 1e6,
                chan_width_khz,
            )
        )

    total_bw = sum(v["bw"] for v in spw_map.values())
    fmin_all = min(v["fmin"] for v in spw_map.values())
    fmax_all = max(v["fmax"] for v in spw_map.values())
    log.info(
        "    Total SPWs : %d  |  Total BW : %.3f MHz  |  "
        "Full range : %.6f -- %.6f GHz",
        len(spw_map), total_bw / 1e6, fmin_all / 1e9, fmax_all / 1e9,
    )

    sorted_spws = sorted(spw_map.items())
    gap_found   = False
    for i in range(len(sorted_spws) - 1):
        _, info_a = sorted_spws[i]
        _, info_b = sorted_spws[i + 1]
        gap_hz = info_b["fmin"] - info_a["fmax"]
        if gap_hz > 0:
            log.info(
                "    [GAP] SPW %d / SPW %d : %.3f MHz gap "
                "at %.6f -- %.6f GHz",
                sorted_spws[i][0], sorted_spws[i + 1][0],
                gap_hz / 1e6,
                info_a["fmax"] / 1e9,
                info_b["fmin"] / 1e9,
            )
            gap_found = True
    if not gap_found:
        log.info("    No gaps detected between SPWs.")

    return spw_map


def get_uvrange_stats(vis):
    """
    Compute UV distance statistics for *vis* in klambda.

    Autocorrelations (ANTENNA1 == ANTENNA2) and fully flagged rows are
    always excluded.  UV distance is evaluated at the mean frequency of
    each SPW so that multi-SPW datasets are handled correctly.

    Returns a dict with keys:
        uvmin, uvmax, uvmean, uvmedian  -- floats, klambda
        n_rows_total, n_rows_used       -- ints
        n_autocorr, n_flagged_rows      -- ints
        per_spw                         -- list of per-SPW sub-dicts
    """
    lightspeed = 299792458.0  # m/s

    tb.open(vis)
    uvw      = tb.getcol("UVW")
    ant1     = tb.getcol("ANTENNA1")
    ant2     = tb.getcol("ANTENNA2")
    flag_row = tb.getcol("FLAG_ROW")
    dd_ids   = tb.getcol("DATA_DESC_ID")
    tb.close()

    n_rows_total = uvw.shape[1]
    n_autocorr   = int(np.sum(ant1 == ant2))
    n_flagged    = int(np.sum(flag_row))

    good_mask   = (ant1 != ant2) & (~flag_row.astype(bool))
    n_rows_used = int(np.sum(good_mask))

    if n_rows_used == 0:
        log.warning("  No usable rows after excluding autocorr and flagged rows!")
        return None

    tb.open(vis + "/DATA_DESCRIPTION")
    spw_for_dd = tb.getcol("SPECTRAL_WINDOW_ID")
    tb.close()

    spw_map = get_spw_freq_map(vis)

    row_wavelen = np.zeros(n_rows_total, dtype=float)
    for dd_id, spw_id in enumerate(spw_for_dd):
        mask_dd = dd_ids == dd_id
        row_wavelen[mask_dd] = lightspeed / spw_map[spw_id]["fmean"]

    u_good  = uvw[0, good_mask]
    v_good  = uvw[1, good_mask]
    wl_good = row_wavelen[good_mask]
    uvdist  = np.sqrt(u_good ** 2 + v_good ** 2) / wl_good * 1e-3

    per_spw = []
    dd_good = dd_ids[good_mask]
    for dd_id, spw_id in enumerate(spw_for_dd):
        mask_spw = dd_good == dd_id
        if not np.any(mask_spw):
            continue
        uv_spw = uvdist[mask_spw]
        per_spw.append({
            "dd_id"  : dd_id,
            "spw_id" : spw_id,
            "fmean"  : spw_map[spw_id]["fmean"],
            "uvmin"  : float(np.nanmin(uv_spw)),
            "uvmax"  : float(np.nanmax(uv_spw)),
            "uvmean" : float(np.nanmean(uv_spw)),
            "nrows"  : int(np.sum(mask_spw)),
        })

    return {
        "uvmin"          : float(np.nanmin(uvdist)),
        "uvmax"          : float(np.nanmax(uvdist)),
        "uvmean"         : float(np.nanmean(uvdist)),
        "uvmedian"       : float(np.nanmedian(uvdist)),
        "n_rows_total"   : n_rows_total,
        "n_rows_used"    : n_rows_used,
        "n_autocorr"     : n_autocorr,
        "n_flagged_rows" : n_flagged,
        "per_spw"        : per_spw,
    }


def log_uvrange_stats(vis):
    """
    Log a compact UV range summary for *vis*.
    Returns the stats dict from get_uvrange_stats().
    """
    stats = get_uvrange_stats(vis)
    if stats is None:
        return None
    log.info("  UV stats for: %s", os.path.basename(vis))
    log.info(
        "    Rows  total / autocorr / flagged / used : "
        "%d / %d / %d / %d",
        stats["n_rows_total"], stats["n_autocorr"],
        stats["n_flagged_rows"], stats["n_rows_used"],
    )
    log.info(
        "    UV min / max / mean / median (klambda)  : "
        "%.4f / %.4f / %.4f / %.4f",
        stats["uvmin"], stats["uvmax"],
        stats["uvmean"], stats["uvmedian"],
    )
    return stats


# ---------------------------------------------------------------------------
# Remaining lower-level visibility helpers
# ---------------------------------------------------------------------------

def read_spws(vis):
    """Return the SPECTRAL_WINDOW_ID array from the DATA_DESCRIPTION sub-table."""
    tb.open(vis + "/DATA_DESCRIPTION")
    spw_ids = tb.getcol("SPECTRAL_WINDOW_ID")
    tb.close()
    return spw_ids


def read_col_names(vis):
    """Return the list of column names in the main table of *vis*."""
    tb.open(vis)
    col_names = tb.colnames()
    tb.close()
    return col_names


def query_ms(vis, spw_id, query_args=None):
    """Return a dict of requested data items for a given SPW in *vis*."""
    if query_args is None:
        query_args = ["UVW", "FLAG"]
    ms.open(vis)
    ms.selectinit(datadescid=spw_id)
    result = ms.getdata(query_args)
    ms.selectinit(reset=True)
    ms.close()
    return result


def get_uvwave_tab(vis, index_freq=None):
    """
    Compute per-baseline UV distance (in klambda) for *vis*.

    NOTE: Retained for backward compatibility with the original notebook.
    For UV range computation in the pipeline, prefer get_uvrange_stats()
    which correctly excludes autocorrelations and uses per-SPW mean frequencies.
    """
    lightspeed = 299792458.0

    ms.open(vis)
    mydata    = ms.getdata(["axis_info"])
    chan_freq  = mydata["axis_info"]["freq_axis"]["chan_freq"].flatten()
    ms.close()

    tb.open(vis)
    uvw      = tb.getcol("UVW")
    flag_row = tb.getcol("FLAG_ROW")
    tb.close()

    if index_freq is not None:
        wavelen = (lightspeed / chan_freq[index_freq]) * 1e3
    else:
        wavelen = (lightspeed / chan_freq.mean()) * 1e3

    uwave  = uvw[0] / wavelen
    vwave  = uvw[1] / wavelen
    uvwave = np.sqrt(uwave ** 2.0 + vwave ** 2.0)
    return uvw, uvwave, wavelen


def get_phase_centre(vis):
    """
    Return the phase centre of *vis* as a CASA-style J2000 string.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u_astropy

    msmd.open(vis)
    ra_rad  = msmd.phasecenter()["m0"]["value"]
    dec_rad = msmd.phasecenter()["m1"]["value"]
    msmd.close()

    coord     = SkyCoord(ra=ra_rad * u_astropy.radian,
                         dec=dec_rad * u_astropy.radian, frame="icrs")
    formatted = coord.to_string("hmsdms")
    fmt_ra, fmt_dec = formatted.split()

    fmt_ra  = fmt_ra.replace("h", ":").replace("m", ":").replace("s", "")
    fmt_dec = fmt_dec.replace("d", ".").replace("m", ".").replace("s", "")

    return "J2000 {} {}".format(fmt_ra, fmt_dec)


def plot_uwave_vwave(vis, color="black", fig=None, ax=None,
                     chunk_size=8, downsample_factor=200):
    """
    Plot the UV coverage of *vis* using matplotlib (pure Python, no plotms).
    Multi-SPW aware; downsamples for speed.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import matplotlib.pyplot as plt

    lightspeed = 299792458.0

    def _get_blinfo(ant1_arr, ant2_arr):
        ant_uniq = np.unique(np.hstack((ant1_arr, ant2_arr)))
        return dict(
            [((x, y), np.where((ant1_arr == x) & (ant2_arr == y))[0])
             for x in ant_uniq for y in ant_uniq if y > x]
        )

    msmd.open(vis)
    nspw = len(msmd.bandwidths())
    chan_freqs_all = np.empty(nspw, dtype=object)
    for nch in range(nspw):
        chan_freqs_all[nch] = msmd.chanfreqs(nch)
    msmd.done()
    chan_freq = np.concatenate(chan_freqs_all)

    tb.open(vis)
    uvw  = tb.getcol("UVW")
    ant1 = tb.getcol("ANTENNA1")
    ant2 = tb.getcol("ANTENNA2")
    tb.close()

    bldict        = _get_blinfo(ant1, ant2)
    reshaped_size = int(len(chan_freq) / chunk_size)
    avg_chan       = chan_freq[: reshaped_size * chunk_size].reshape(
                        reshaped_size, chunk_size)
    chunk_mean     = avg_chan.mean(axis=1).reshape(-1, 1)

    if fig is None:
        fig = plt.figure()
        ax  = fig.add_subplot(111)

    def _downsample(arr, factor):
        return arr[:, ::factor]

    for bl in list(bldict.keys()):
        wavelen = (lightspeed / chunk_mean) * 1e3
        u_bl    = uvw[0, bldict[bl]]
        v_bl    = uvw[1, bldict[bl]]
        uuu     = np.tile(np.hstack([u_bl, np.nan, -u_bl]),
                          (reshaped_size, 1)) / wavelen
        vvv     = np.tile(np.hstack([v_bl, np.nan, -v_bl]),
                          (reshaped_size, 1)) / wavelen
        ax.plot(_downsample(uuu, downsample_factor).T,
                _downsample(vvv, downsample_factor).T,
                ".", markersize=0.5, color=color, alpha=0.7)

    return fig, ax


# ===========================================================================
# Helpers: cell size estimation
# ===========================================================================

def compute_cell_size(vis, oversample=5.0):
    """
    Estimate an appropriate pixel (cell) size for *vis* in arcseconds.

    Method
    ------
    The synthesized beam FWHM is approximated as:

        theta_beam = lambda / B_max   [radians]

    where B_max is the maximum baseline length (m) over all cross-correlation,
    unflagged rows, and lambda is computed at the mean frequency of the
    dataset.  The cell size is then:

        cell = theta_beam / oversample

    with oversample=5 (default) giving good Nyquist sampling.

    NOTE: This is an analytical estimate.  For a more accurate beam derived
    directly from the uv coverage (L80 method), use compute_vis_restoring_beam()
    which calls CASA's analysisUtils.estimateSynthesizedBeam via a subprocess.

    Parameters
    ----------
    vis        : str    Path to the Measurement Set.
    oversample : float  Number of pixels across the synthesized beam (default 5).

    Returns
    -------
    cell_str : str   Cell size string suitable for wsclean, e.g. '0.042arcsec'.
    cell_asec: float Cell size in arcseconds.
    """
    lightspeed = 299792458.0  # m/s

    # Max baseline from cross-correlation, unflagged rows only
    tb.open(vis)
    uvw      = tb.getcol("UVW")
    ant1     = tb.getcol("ANTENNA1")
    ant2     = tb.getcol("ANTENNA2")
    flag_row = tb.getcol("FLAG_ROW")
    tb.close()

    good     = (ant1 != ant2) & (~flag_row.astype(bool))
    u_good   = uvw[0, good]
    v_good   = uvw[1, good]
    b_max_m  = float(np.max(np.sqrt(u_good ** 2 + v_good ** 2)))

    # Mean frequency across all SPWs
    spw_map   = get_spw_freq_map(vis)
    freq_mean = float(np.mean([info["fmean"] for info in spw_map.values()]))

    wavelength_m  = lightspeed / freq_mean
    beam_rad      = wavelength_m / b_max_m
    beam_arcsec   = beam_rad * (180.0 / np.pi) * 3600.0
    cell_arcsec   = beam_arcsec / oversample

    cell_str = "{:.4f}arcsec".format(cell_arcsec)
    log.info(
        "  Cell size estimate for %-45s  Bmax=%.1f m  "
        "nu_mean=%.3f GHz  beam=%.4f arcsec  cell=%s",
        os.path.basename(vis),
        b_max_m,
        freq_mean / 1e9,
        beam_arcsec,
        cell_str,
    )
    return cell_str, cell_arcsec


def compute_vis_restoring_beam(vis, casa_path="casa",
                               analysis_scripts_path=None):
    """
    Compute the effective restoring beam using analysisUtils.estimateSynthesizedBeam.

    This spawns a separate CASA subprocess, so it requires a monolithic CASA
    installation at *casa_path* and the analysisUtils scripts at
    *analysis_scripts_path*.

    Returns the effective_beam string as returned by au.estimateSynthesizedBeam.
    """
    import subprocess

    os.environ["MPLBACKEND"] = "Agg"

    scripts_path = analysis_scripts_path or ""
    script_content = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import sys\n"
        "sys.path.append('{scripts_path}')\n"
        "import analysisUtils as au\n"
        "effective_beam = au.estimateSynthesizedBeam("
        "'{vis}', useL80method=True, field='')\n"
        "print(effective_beam)\n"
    ).format(scripts_path=scripts_path, vis=vis)

    script_file = "compute_beam_tmp.py"
    with open(script_file, "w") as f:
        f.write(script_content)

    result = subprocess.run(
        [casa_path, "-nologger", "-c", script_file],
        capture_output=True, text=True,
    )

    try:
        os.remove(script_file)
    except OSError:
        pass

    if result.returncode != 0:
        log.warning("  compute_vis_restoring_beam failed: %s", result.stderr)
        return None

    effective_beam = result.stdout.splitlines()[-1]
    log.info("  Effective beam (L80): %s", effective_beam)
    return effective_beam


# ===========================================================================
# Helpers: mlibs / morphen import
# ===========================================================================

def import_mlibs(mlibs_path=None):
    """
    Import the mlibs module from the morphen package.

    Parameters
    ----------
    mlibs_path : str or None
        Path to the directory containing mlibs.py (usually morphen/morphen/).
        If None, the existing sys.path is used.

    Returns
    -------
    mlibs module, or None if the import fails.
    """
    if mlibs_path:
        if mlibs_path not in sys.path:
            sys.path.insert(0, mlibs_path)
    for _attempt in range(2):
        try:
            import mlibs as _mlibs
            log.info("  mlibs imported from: %s",
                     getattr(_mlibs, "__file__", "unknown location"))
            return _mlibs
        except ImportError as exc:
            if _attempt == 0:
                continue
            log.error(
                "  Could not import mlibs: %s\n"
                "  Specify the path to morphen/morphen/ with --mlibs-path.",
                exc,
            )
            return None


# ===========================================================================
# I/O helpers
# ===========================================================================

def _ensure_dir(path):
    """Create *path* (and any parents) if it does not already exist."""
    if path and not os.path.exists(path):
        os.makedirs(path)
        log.info("  Created output directory: %s", path)


def _make_output_path(input_vis, suffix, output_dir=None):
    """
    Build an output MS path from *input_vis* by appending *suffix*.

    If *output_dir* is given, the output is placed there regardless of
    where the input lives.  Otherwise the output lands in the same directory
    as the input (original behaviour).

    Example
    -------
    _make_output_path("/data/A.ms", "_ps", output_dir="/out")
    -> "/out/A_ps.ms"

    _make_output_path("/data/A.ms", "_ps")
    -> "/data/A_ps.ms"
    """
    base = os.path.splitext(os.path.basename(input_vis))[0]
    fname = base + suffix + ".ms"
    if output_dir:
        return os.path.join(output_dir, fname)
    else:
        parent = os.path.dirname(input_vis) or "."
        return os.path.join(parent, fname)


def _remove_if_exists(path, force_overwrite):
    """
    If *path* exists:
      - force_overwrite=True  : remove it, return True
      - force_overwrite=False : log a skip notice, return False
    If *path* does not exist, return True unconditionally.
    """
    if os.path.exists(path):
        if force_overwrite:
            log.info("  Removing existing output: %s", os.path.basename(path))
            shutil.rmtree(path)
            return True
        else:
            log.info(
                "  Output already exists, skipping "
                "(use --force-overwrite to reprocess): %s",
                os.path.basename(path),
            )
            return False
    return True


# ===========================================================================
# Helpers: instrument / band identification and output directory layout
# ===========================================================================

# ---------------------------------------------------------------------------
# Frequency -> band letter mapping
# Boundaries follow the IEEE standard radio band definitions.
# ---------------------------------------------------------------------------
_BAND_EDGES_GHZ = [
    (0.1,   0.3,  "P"),
    (0.3,   1.0,  "L"),   # Note: L conventionally 1-2 GHz; P covers below
    (1.0,   2.0,  "L"),
    (2.0,   4.0,  "S"),
    (4.0,   8.0,  "C"),
    (8.0,  12.0,  "X"),
    (12.0, 18.0,  "Ku"),
    (18.0, 26.5,  "K"),
    (26.5, 40.0,  "Ka"),
    (40.0, 75.0,  "Q"),
    (75.0, 110.0, "W"),
]


def freq_to_band(freq_hz):
    """
    Map a frequency in Hz to the standard IEEE radio band letter.

    Returns the band string (e.g. 'C', 'X', 'Ka') or 'UNK' if the frequency
    falls outside all known band edges.
    """
    freq_ghz = freq_hz / 1e9
    for lo, hi, label in _BAND_EDGES_GHZ:
        if lo <= freq_ghz < hi:
            return label
    return "UNK"


# ---------------------------------------------------------------------------
# Telescope name normalisation
# ---------------------------------------------------------------------------
_TELESCOPE_ALIASES = {
    "EVLA"     : "VLA",
    "VLA"      : "VLA",
    "E-MERLIN" : "eM",
    "EMERLIN"  : "eM",
    "E_MERLIN" : "eM",
    "eMERLIN"  : "eM",
    "MERLIN"   : "eM",
}


def get_telescope_name(vis):
    """
    Read the telescope name from the OBSERVATION sub-table of *vis* and
    return a short normalised label (e.g. 'VLA', 'eM').

    Falls back to 'UNK' if the sub-table is missing or unreadable.
    """
    try:
        tb.open(vis + "/OBSERVATION")
        raw = tb.getcol("TELESCOPE_NAME")
        tb.close()
        name = str(raw[0]).strip()
        # Try exact match first, then case-insensitive
        if name in _TELESCOPE_ALIASES:
            return _TELESCOPE_ALIASES[name]
        upper = name.upper().replace("-", "").replace("_", "")
        for key, val in _TELESCOPE_ALIASES.items():
            if key.upper().replace("-", "").replace("_", "") == upper:
                return val
        # Unknown telescope: return the raw name truncated to 6 chars
        return name[:6]
    except Exception as exc:
        log.warning("  Could not read telescope name from %s: %s",
                    os.path.basename(vis), exc)
        return "UNK"


def get_vis_label(vis):
    """
    Build a short instrument-band label for *vis*, e.g. 'VLA_C' or 'eM_Ka'.

    The band is derived from the mean frequency across all SPWs.
    """
    telescope = get_telescope_name(vis)
    spw_map   = get_spw_freq_map(vis)
    freq_mean = float(np.mean([info["fmean"] for info in spw_map.values()]))
    band      = freq_to_band(freq_mean)
    return "{}_{}".format(telescope, band)


def get_group_label(visibilities):
    """
    Build a combined label from the instrument-band labels of all input
    visibilities, joined by underscores.

    Labels are sorted by mean frequency (ascending, lowest band first) and
    deduplicated so that multiple MSs of the same telescope+band contribute
    only one entry.

    Example: ['VLA_C', 'VLA_C', 'VLA_C', 'VLA_Ka', 'VLA_S']
             -> 'VLA_S_VLA_C_VLA_Ka'
    """
    entries = []
    for vis in visibilities:
        spw_map   = get_spw_freq_map(vis)
        freq_mean = float(np.mean([info["fmean"] for info in spw_map.values()]))
        label     = get_vis_label(vis)
        entries.append((freq_mean, vis, label))

    entries.sort(key=lambda x: x[0])

    seen   = set()
    unique = []
    for _, _, lbl in entries:
        if lbl not in seen:
            seen.add(lbl)
            unique.append(lbl)

    group = "_".join(unique)
    log.info("  Visibility group label: %s", group)
    for _, vis, lbl in entries:
        log.info("    %-55s  %s", os.path.basename(vis), lbl)
    return group


def _build_output_dirs(visibilities, output_dir):
    """
    Resolve the concrete output directories for each pipeline stage.

    If *output_dir* is None, both values are also None (outputs land next
    to their respective input files, existing behaviour).

    Otherwise:

        output_dir/
          native/               <- phase-shifted MSs + native images
          uv_matched/<label>/   <- UV-matched MSs + UV-matched images
                                   where <label> is e.g. 'eM_C_VLA_C_VLA_Ka'

    Parameters
    ----------
    visibilities : list of str
        The ORIGINAL input visibilities (before any processing) -- used to
        derive the group label from telescope names and bands.
    output_dir : str or None
        Root output directory from the CLI / config.

    Returns
    -------
    dict with keys:
        "native"     : str or None
        "uv_matched" : str or None
        "group_label": str  (empty string when output_dir is None)
    """
    if not output_dir:
        return {"native": None, "uv_matched": None, "group_label": ""}

    separator(" ")
    log.info("  Resolving output directory layout ...")
    group_label  = get_group_label(visibilities)
    native_dir   = os.path.join(output_dir, "native")
    uvmatch_dir  = os.path.join(output_dir, "uv_matched", group_label)

    _ensure_dir(native_dir)
    _ensure_dir(uvmatch_dir)

    log.info("  native dir   : %s", native_dir)
    log.info("  uv_match dir : %s", uvmatch_dir)
    separator(" ")

    return {
        "native"     : native_dir,
        "uv_matched" : uvmatch_dir,
        "group_label": group_label,
    }


# ===========================================================================
# Pipeline stages
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage 1 -- Phase shift
# ---------------------------------------------------------------------------

def run_phaseshift(visibilities, ref_vis=None, ref_phasecenter=None,
                   force_overwrite=False, output_dir=None):
    """
    Phase-shift every MS in *visibilities* to a common phase centre.

    The target phase centre is resolved in order of priority:
      1. *ref_vis*         -- read from this MS (highest priority).
      2. *ref_phasecenter* -- use this CASA J2000 string directly.

    Output files are named  <basename>_ps.ms  and placed in *output_dir*
    (or in the same directory as the input if output_dir is None).

    Returns the list of shifted MS paths (same order as input).
    """
    section("STAGE 1 -- Phase shift visibilities")
    _ensure_dir(output_dir)

    if ref_vis is not None:
        log.info("Reading reference phase centre from: %s", ref_vis)
        ref_phasecenter = get_phase_centre(ref_vis)
    else:
        log.info("Using supplied reference phase centre.")
    log.info("  Reference phase centre : %s", ref_phasecenter)
    separator(" ")

    log.info("Input phase centres:")
    for vis in visibilities:
        pc = get_phase_centre(vis)
        log.info("  %-60s  %s", os.path.basename(vis), pc)
    separator(" ")

    shifted = []
    for vis in tqdm(visibilities, desc="  phaseshift"):
        out_vis = _make_output_path(vis, "_ps", output_dir)
        shifted.append(out_vis)
        if not _remove_if_exists(out_vis, force_overwrite):
            continue
        try:
            casatasks.phaseshift(
                vis=vis,
                phasecenter=ref_phasecenter,
                outputvis=out_vis,
            )
            log.info("  OK  %s", os.path.basename(out_vis))
        except Exception as exc:
            log.error("  FAILED  %s  --  %s", os.path.basename(vis), exc)

    separator()
    log.info("Phase-shifted MSs:")
    for v in shifted:
        log.info("  %s", v)
    separator()
    return shifted


# ---------------------------------------------------------------------------
# Stage 2 -- UV matching
# ---------------------------------------------------------------------------

def _compute_common_uvrange(visibilities):
    """
    Compute the common UV-wave range (klambda) across *visibilities*.

    Autocorrelations and flagged rows are excluded.  UV distance is
    evaluated at the mean frequency of each SPW.

    Intersection strategy:
        common_min = max of all per-MS UV minimums
        common_max = min of all per-MS UV maximums
    """
    log.info(
        "  Computing UV range for each visibility "
        "(autocorr and flagged rows excluded) ..."
    )

    uv_mins = []
    uv_maxs = []

    for vis in tqdm(visibilities, desc="  UV stats"):
        stats = get_uvrange_stats(vis)
        if stats is None:
            raise RuntimeError(
                "No usable rows in {}.  "
                "Cannot compute UV range.".format(vis)
            )
        uv_mins.append(stats["uvmin"])
        uv_maxs.append(stats["uvmax"])
        log.info(
            "    %-55s  %.4f -- %.4f klambda",
            os.path.basename(vis), stats["uvmin"], stats["uvmax"],
        )

    common_min = float(np.max(uv_mins))
    common_max = float(np.min(uv_maxs))

    if common_min > common_max:
        raise RuntimeError(
            "No common UV-wave range found across input visibilities. "
            "Check that all datasets share overlapping baselines."
        )

    log.info("  Common UV range : %.4f -- %.4f klambda", common_min, common_max)
    uvrange = "{:.4f}~{:.4f} klambda".format(common_min, common_max)
    return uvrange, common_min, common_max


def _plot_uvmatch_overlay(vis_list, output_dir, mem_fraction=0.25):
    """
    Save a combined UV-wave coverage PNG for a list of UV-matched MSs.

    Each MS is plotted in a distinct colour, all overlaid on the same axes.
    Uses SPW-by-SPW row iteration (via plot_vis_python helpers) so that
    bandwidth-smearing arcs are rendered correctly for every instrument
    (VLA, e-MERLIN, etc.) -- identical methodology to plot_uvwave().

    Parameters
    ----------
    vis_list     : list of str   UV-matched MS paths.
    output_dir   : str           Directory where the PNG is saved.
    mem_fraction : float         Fraction of RAM for row-chunk budget.

    Returns
    -------
    str  Path to the saved PNG, or None on failure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    # Shared plotting module in ../plotting/
    _plotting_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "plotting"))
    if _plotting_dir not in sys.path:
        sys.path.insert(0, _plotting_dir)
    try:
        from plot_vis_python import (
            _spw_meta, _iter_spw_chunks, _bl_sort_index,
            _get_available_memory_gb,
        )
    except ImportError as exc:
        log.warning("  _plot_uvmatch_overlay: cannot import plot_vis_python: %s", exc)
        return None

    LIGHT_SPEED = 299792458.0
    avail_gb   = _get_available_memory_gb() or 4.0
    chunk_rows = min(500_000, max(10_000,
                                  int(avail_gb * mem_fraction * 1e9 / (3 * 8))))

    try:
        cmap = matplotlib.colormaps['tab20']
    except AttributeError:
        cmap = matplotlib.cm.get_cmap('tab20')

    n_ms   = max(len(vis_list), 1)
    colors = [cmap(i / n_ms) for i in range(len(vis_list))]

    fig, ax = plt.subplots(figsize=(8, 8))

    for vis, color in zip(vis_list, colors):
        log.info("    UV coverage: %s", os.path.basename(vis))
        try:
            dd_info = _spw_meta(vis, tb, msmd)
        except Exception as exc:
            log.warning("    Could not read SPW info from %s: %s",
                        os.path.basename(vis), exc)
            continue

        # Drop dd_ids that have no rows in the MAIN table after UV filtering.
        # _iter_spw_chunks calls selecttaql(DATA_DESC_ID==n); if the SPW is
        # absent from the data CASA raises "zero selected rows".
        try:
            tb.open(vis)
            dd_ids_present = set(int(d) for d in np.unique(tb.getcol('DATA_DESC_ID')))
            tb.close()
        except Exception:
            dd_ids_present = set(dd_info.keys())
        dd_ids = sorted(d for d in dd_info if d in dd_ids_present)
        if not dd_ids:
            log.warning("    No populated SPWs in %s -- skipping.", os.path.basename(vis))
            continue
        n_skipped = len(dd_info) - len(dd_ids)
        if n_skipped:
            log.info("    Skipped %d empty SPW(s) in %s",
                     n_skipped, os.path.basename(vis))

        try:
            tb.open(vis + '/ANTENNA')
            n_ant = int(tb.nrows())
            tb.close()
        except Exception as exc:
            log.warning("    Could not read ANTENNA table from %s: %s",
                        os.path.basename(vis), exc)
            continue

        bl_ukl = {}
        bl_vkl = {}

        try:
            for chunk in _iter_spw_chunks(vis, ms, dd_ids, chunk_rows,
                                           ['uvw', 'antenna1', 'antenna2',
                                            'flag_row']):
                dd_id    = chunk['dd_id']
                uvw      = chunk['uvw']
                ant1     = chunk['antenna1'].astype(np.int32)
                ant2     = chunk['antenna2'].astype(np.int32)
                flag_row = chunk['flag_row']

                good = (ant1 != ant2) & (~flag_row)
                if not good.any():
                    continue

                g_u  = uvw[0, good]
                g_v  = uvw[1, good]
                g_a1 = ant1[good]
                g_a2 = ant2[good]

                # Per-SPW frequency chunks (cs=16, matching plot_uvwave)
                freqs  = dd_info[dd_id]['chan_freqs']
                n_freq = len(freqs)
                if n_freq == 0:
                    continue
                cs  = min(16, n_freq)
                n_c = max(1, n_freq // cs)
                cf  = freqs[:n_c * cs].reshape(n_c, cs).mean(axis=1)
                wl  = (LIGHT_SPEED / cf).reshape(-1, 1)  # (n_c, 1) metres

                sort_ord, _, starts, cnts, _ = _bl_sort_index(g_a1, g_a2, n_ant)
                s_u  = g_u[sort_ord]
                s_v  = g_v[sort_ord]
                s_a1 = g_a1[sort_ord]
                s_a2 = g_a2[sort_ord]

                for start, cnt in zip(starts, cnts):
                    sl  = slice(start, start + cnt)
                    key = (int(s_a1[start]), int(s_a2[start]))

                    u_bl = s_u[sl]
                    v_bl = s_v[sl]

                    if key not in bl_ukl:
                        bl_ukl[key] = []
                        bl_vkl[key] = []

                    uv_both = np.hstack([u_bl, -u_bl])
                    vv_both = np.hstack([v_bl, -v_bl])
                    bl_ukl[key].append(
                        (np.tile(uv_both, (n_c, 1)) / wl / 1e3).ravel())
                    bl_vkl[key].append(
                        (np.tile(vv_both, (n_c, 1)) / wl / 1e3).ravel())

        except Exception as exc:
            log.warning("    Error reading %s: %s", os.path.basename(vis), exc)
            continue

        for key in bl_ukl:
            if not bl_ukl[key]:
                continue
            uw = np.concatenate(bl_ukl[key])
            vw = np.concatenate(bl_vkl[key])
            ax.plot(uw, vw, '.', markersize=0.05,
                    color=color, alpha=0.4, rasterized=True, linewidth=0)

    handles = [
        mlines.Line2D([], [], color=colors[i], marker='.', linestyle='None',
                      markersize=6, label=os.path.basename(vis_list[i]))
        for i in range(len(vis_list))
    ]
    ax.legend(handles=handles, fontsize=7, loc='upper right', framealpha=0.7)
    ax.set_xlabel(r"$u\;[\mathrm{k}\lambda]$")
    ax.set_ylabel(r"$v\;[\mathrm{k}\lambda]$")
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_title(r"UV-matched $uv$ coverage")

    fig_path = os.path.join(output_dir, 'uvmatched_uv_coverage.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("  Saved UV coverage plot: %s", fig_path)
    return fig_path


def run_uv_match(visibilities, plot_vis=False,
                 force_overwrite=False, output_dir=None):
    """
    Compute the common UV-wave range and split each MS to that range.

    Output files are named  <basename>_uvmatch.ms  and placed in *output_dir*
    (or in the same directory as the input if output_dir is None).

    Returns the list of UV-matched MS paths (same order as input).
    """
    section("STAGE 2 -- UV matching")
    _ensure_dir(output_dir)

    log.info("Input visibilities (%d) -- SPW and UV summary:", len(visibilities))
    separator(" ")
    for vis in visibilities:
        log_spw_info(vis)
        log_uvrange_stats(vis)
        separator(" ")

    uvrange, common_min, common_max = _compute_common_uvrange(visibilities)
    log.info("  CASA uvrange string: '%s'", uvrange)

    separator(" ")
    log.info("  Splitting to common UV range ...")

    matched = []
    for vis in tqdm(visibilities, desc="  split"):
        out_vis = _make_output_path(vis, "_uvmatch", output_dir)
        matched.append(out_vis)
        if not _remove_if_exists(out_vis, force_overwrite):
            continue
        try:
            tb.open(vis)
            n_in = int(tb.nrows())
            tb.close()
        except Exception:
            n_in = None
        try:
            casatasks.split(
                vis=vis,
                outputvis=out_vis,
                datacolumn="data",
                uvrange=uvrange,
                keepflags=False,
            )
            try:
                tb.open(out_vis)
                n_out = int(tb.nrows())
                tb.close()
            except Exception:
                n_out = None
            if n_in and n_out is not None:
                frac_kept     = n_out / n_in
                frac_discard  = 1.0 - frac_kept
                log.info(
                    "  OK  %-45s  rows: %d -> %d  kept %.1f%%  discarded %.1f%%",
                    os.path.basename(out_vis),
                    n_in, n_out,
                    100.0 * frac_kept,
                    100.0 * frac_discard,
                )
            else:
                log.info("  OK  %s", os.path.basename(out_vis))
        except Exception as exc:
            log.error("  FAILED  %s  --  %s", os.path.basename(vis), exc)

    separator(" ")
    log.info("UV-matched MSs:")
    for v in matched:
        log.info("  %s", v)

    if plot_vis:
        separator(" ")
        log.info("  Plotting combined UV coverage ...")
        plot_dir = output_dir or os.path.dirname(matched[0]) or "."
        _plot_uvmatch_overlay(matched, plot_dir)

    separator()
    return matched


# ---------------------------------------------------------------------------
# Stage 3 -- Native imaging
# ---------------------------------------------------------------------------

def run_native_imaging(visibilities, imaging_params, output_dir=None):
    """
    Image each MS in *visibilities* at its native UV coverage using wsclean
    (called via mlibs.run_wsclean).

    For each visibility, the cell size is either taken from *imaging_params*
    or estimated analytically from the maximum baseline length.
    Each visibility is imaged independently at every robust value in
    imaging_params['robust_list'].

    Output images land in *output_dir* (or next to the visibility if None).

    Parameters
    ----------
    visibilities   : list of str   Paths to the input Measurement Sets.
    imaging_params : dict          See build_imaging_params() for keys.
    output_dir     : str or None   Directory for wsclean output images.
    """
    section("STAGE 3 -- Native imaging")
    _ensure_dir(output_dir)

    mlibs = import_mlibs(imaging_params.get("mlibs_path"))
    if mlibs is None:
        log.error("  Cannot proceed without mlibs -- aborting Stage 3.")
        separator()
        return

    robust_list          = imaging_params["robust_list"]
    n_subimages          = imaging_params["n_subimages"]
    imsizex              = imaging_params["imsizex"]
    imsizey              = imaging_params["imsizey"]
    opt_args             = imaging_params["opt_args"]
    nsigma_automask      = imaging_params["nsigma_automask"]
    nsigma_autothreshold = imaging_params["nsigma_autothreshold"]
    uvtaper              = imaging_params["uvtaper"]
    niter                = imaging_params["niter"]
    with_multiscale      = imaging_params["with_multiscale"]
    scales               = imaging_params["scales"]
    cell_override        = imaging_params.get("cell")   # None means auto
    native_beam_size     = imaging_params.get("native_beam_size")
    native_uvtaper       = imaging_params.get("native_uvtaper")

    # Build vis_list dict: short key -> full path
    # Key is the MS basename without extension, which uniquely identifies
    # each dataset and is used as the image name prefix.
    vis_list = {
        os.path.splitext(os.path.basename(v))[0]: v
        for v in visibilities
    }

    log.info("Visibilities to image (%d):", len(vis_list))
    for name, path in vis_list.items():
        log.info("  %-40s  %s", name, path)

    # Run a listobs for each vis as a sanity check
    separator(" ")
    log.info("  Writing listobs files ...")
    for name, vis_path in vis_list.items():
        listobs_file = os.path.join(
            output_dir or os.path.dirname(vis_path) or ".",
            name + ".listobs",
        )
        try:
            casatasks.listobs(
                vis=vis_path,
                listfile=listobs_file,
                overwrite=True,
            )
            log.info("  listobs -> %s", listobs_file)
        except Exception as exc:
            log.warning("  listobs failed for %s: %s", name, exc)

    separator(" ")
    log.info("  Imaging parameters:")
    log.info("    n_subimages      : %d", n_subimages)
    log.info("    imsize (x, y)    : %d x %d", imsizex, imsizey)
    log.info("    robust values    : %s", robust_list)
    cell_mode = (
        "fixed ({})".format(cell_override)
        if cell_override and cell_override.lower() != "min"
        else ("min across all MSs" if cell_override == "min" else "auto (per MS)")
    )
    log.info("    cell mode        : %s", cell_mode)
    log.info("    nsigma_automask  : %s", nsigma_automask)
    log.info("    nsigma_autothres : %s", nsigma_autothreshold)
    log.info("    uvtaper          : %s", uvtaper)
    log.info("    native_beam_size : %s", native_beam_size or "(wsclean default)")
    log.info("    native_uvtaper   : %s", native_uvtaper or "(uses --uvtaper)")
    log.info("    niter            : %d", niter)
    log.info("    with_multiscale  : %s", with_multiscale)
    log.info("    scales           : %s", scales)
    log.info("    opt_args         : %s", opt_args)
    separator(" ")

    # -------------------------------------------------------------------
    # Cell size pre-pass
    # Resolve the cell size for every visibility BEFORE entering the main
    # imaging loop so we can implement the "min" mode cleanly.
    #
    # Modes:
    #   None        -> compute analytically per MS independently
    #   "min"       -> compute analytically per MS, then use the minimum
    #                  so all images share the same pixel scale
    #   "<value>"   -> use this fixed string for every MS (no computation)
    # -------------------------------------------------------------------
    if cell_override and cell_override.lower() != "min":
        # Fixed value: same cell for every MS, no computation needed
        vis_cells = {name: cell_override for name in vis_list}
        log.info("  Cell size (fixed): %s", cell_override)
    else:
        # Compute analytically for every MS
        log.info("  Computing cell sizes analytically ...")
        log.info(
            "    %-45s  %-15s  %s",
            "MS", "Cell [arcsec]", "Bmax [m]"
        )
        log.info("    " + "-" * 60)
        vis_cells = {}
        for name, vis_path in vis_list.items():
            cell_str, cell_asec = compute_cell_size(vis_path)
            vis_cells[name] = cell_str
            log.info("    %-45s  %-15s", name, cell_str)

        if cell_override and cell_override.lower() == "min":
            # Pick the minimum cell size (smallest pixels = highest resolution)
            # so that all images share a common pixel scale.
            # NOTE: compute_cell_size returns strings like "0.0423arcsec";
            # we call it again here to get the numeric value for comparison.
            min_cell_asec = min(
                compute_cell_size(v)[1] for v in vis_list.values()
            )
            min_cell_str = "{:.4f}arcsec".format(min_cell_asec)
            log.info(
                "  Cell size mode 'min': using %s for all MSs", min_cell_str
            )
            vis_cells = {name: min_cell_str for name in vis_list}

    separator(" ")

    # Main imaging loop
    #
    # IMPORTANT: base_name must be a plain filename prefix with NO directory
    # component.  run_wsclean constructs the full image path internally by
    # binding the MS directory (or cwd) as /mnt inside Singularity and
    # prepending /mnt/ to the name.  Injecting an absolute path here would
    # produce a broken double-path like /mnt//absolute/path/... and wsclean
    # would crash trying to create that file.
    #
    # If output_dir is set we os.chdir into it before calling run_wsclean so
    # that wsclean's --bind ./:/mnt lands in the right place, then restore cwd.

    original_cwd = os.getcwd()

    for vis_name, vis_path in vis_list.items():
        log.info("  Imaging: %s", vis_name)
        log.info("    Path: %s", vis_path)

        cell = vis_cells[vis_name]
        log.info("    Cell size: %s", cell)

        # base_name is ONLY the short prefix; run_wsclean appends MS name,
        # imsize, thresholds, cell size, robust, etc. automatically.
        if native_beam_size:
            used_opt_args = "' -circular-beam -beam-size {} '".format(native_beam_size)
            prefix = "im_nc{}_cb_bs_{}".format(n_subimages, native_beam_size)
        else:
            used_opt_args = opt_args
            prefix = "im_nc{}_cb".format(n_subimages)

        used_uvtaper = native_uvtaper if native_uvtaper is not None else uvtaper

        # If an output directory is requested, switch into it so wsclean
        # writes images there.  The MS path is kept absolute so CASA/wsclean
        # can still find the data regardless of cwd.
        if output_dir:
            os.chdir(output_dir)
            log.info("    Working directory set to: %s", output_dir)

        for robust in robust_list:
            log.info("    robust=%-5.1f  base_name=%s", robust, prefix)
            try:
                image_list, image_statistics, Omaj = mlibs.run_wsclean(
                    vis_path,
                    robust=robust,
                    imsize=imsizex,
                    imsizey=imsizey,
                    opt_args=used_opt_args,
                    cell=cell,
                    base_name=prefix,
                    nsigma_automask=nsigma_automask,
                    nsigma_autothreshold=nsigma_autothreshold,
                    quiet=True,
                    nc=n_subimages,
                    with_multiscale=with_multiscale,
                    scales=scales,
                    datacolumn="DATA",
                    uvtaper=used_uvtaper,
                    calculate_subband_fluxes=True,
                    niter=niter,
                )
                log.info(
                    "    OK  robust=%.1f  images: %d",
                    robust, len(image_list) if image_list else 0,
                )
            except Exception as exc:
                log.error(
                    "    FAILED  %s  robust=%.1f  --  %s",
                    vis_name, robust, exc,
                )

        # Always restore cwd after each visibility, whether or not output_dir
        # was set, so subsequent stages are not affected.
        os.chdir(original_cwd)

    separator()


# ---------------------------------------------------------------------------
# Stage 4 -- UV-matched imaging
# ---------------------------------------------------------------------------

def _split_vis_by_band(vis_list, bands_plus, bands_minus):
    """
    Split *vis_list* into three sub-dicts based on band membership.

    Uses get_vis_label() (which reads the OBSERVATION table and SPW frequencies)
    to determine the band of each MS, then assigns it to one of:

        wt_plus  -- MSs whose band is in *bands_plus*  (e.g. ["X", "Ka"])
                    -> imaged with extra positive-robust values to weight
                       shorter baselines more (compensates for high resolution)
        wt_minus -- MSs whose band is in *bands_minus* (e.g. ["L", "S"])
                    -> imaged with extra negative-robust values to weight
                       longer baselines more (compensates for low resolution)
        main     -- ALL MSs regardless of group (always imaged with robust_main)

    Parameters
    ----------
    vis_list    : dict  {name: path}
    bands_plus  : list of str  band letters for the wt_plus group (may be empty)
    bands_minus : list of str  band letters for the wt_minus group (may be empty)

    Returns
    -------
    main, wt_plus, wt_minus : three dicts {name: path}
    """
    wt_plus  = {}
    wt_minus = {}
    for name, path in vis_list.items():
        label = get_vis_label(path)          # e.g. "VLA_Ka"
        band  = label.split("_")[-1]        # e.g. "Ka"
        if band in bands_plus:
            wt_plus[name] = path
        if band in bands_minus:
            wt_minus[name] = path

    log.info("  Band group assignment:")
    log.info("    %-45s  %-10s  %s", "MS", "Band", "Groups")
    log.info("    " + "-" * 65)
    for name, path in vis_list.items():
        label = get_vis_label(path)
        band  = label.split("_")[-1]
        groups = []
        if name in wt_plus:
            groups.append("wt_plus")
        if name in wt_minus:
            groups.append("wt_minus")
        groups.append("main")
        log.info("    %-45s  %-10s  %s", name, band, ", ".join(groups))

    return dict(vis_list), wt_plus, wt_minus


def _wsclean_probe(vis_path, vis_name, imaging_params, mlibs, prefix,
                   output_dir, original_cwd):
    """
    Run a single wsclean probe imaging call (Pass 1) to extract the
    natural restoring beam size (Omaj) for *vis_path*.

    The probe uses the shared imaging parameters with no forced beam size.

    Returns Omaj (float, arcsec) or None on failure.
    """
    # TODO: Replace this brute-force wsclean probe with an analytical
    # beam estimator once a reliable method (beyond analysisUtils
    # estimateSynthesizedBeam, which is known to return incorrect values
    # for some datasets) is available.  The analytical estimate should
    # derive the synthesised beam from the UV coverage geometry (e.g.
    # lambda / Bmax for major axis, appropriate for circular beams) without
    # requiring a full deconvolution run.

    n_subimages          = imaging_params["n_subimages"]
    imsizex              = imaging_params["imsizex"]
    imsizey              = imaging_params["imsizey"]
    nsigma_automask      = imaging_params["nsigma_automask"]
    nsigma_autothreshold = imaging_params["nsigma_autothreshold"]
    uvtaper              = imaging_params["uvtaper"]
    niter                = imaging_params["niter"]
    with_multiscale      = imaging_params["with_multiscale"]
    scales               = imaging_params["scales"]
    cell_override        = imaging_params.get("cell")
    robust_probe         = imaging_params["robust_probe"]

    cell = cell_override if (cell_override and cell_override.lower() != "min") \
        else compute_cell_size(vis_path)[0]

    if output_dir:
        os.chdir(output_dir)

    try:
        _, _, Omaj = mlibs.run_wsclean(
            vis_path,
            robust=robust_probe,
            imsize=imsizex,
            imsizey=imsizey,
            opt_args="' -circular-beam '",   # no forced beam in probe
            cell=cell,
            base_name=prefix,
            nsigma_automask=nsigma_automask,
            nsigma_autothreshold=nsigma_autothreshold,
            quiet=True,
            nc=n_subimages,
            with_multiscale=with_multiscale,
            scales=scales,
            datacolumn="DATA",
            uvtaper=uvtaper,
            calculate_subband_fluxes=True,
            niter=niter,
        )
        log.info("    Omaj (probe) = %.4f arcsec  [%s]", Omaj, vis_name)
        os.chdir(original_cwd)
        return float(Omaj)
    except Exception as exc:
        log.error("    Probe FAILED for %s: %s", vis_name, exc)
        os.chdir(original_cwd)
        return None


def _wsclean_final(vis_path, vis_name, imaging_params, mlibs, prefix,
                   forced_opt_args, robust_list, output_dir, original_cwd):
    """
    Run wsclean final imaging (Pass 2) for *vis_path* at every value in
    *robust_list*, using *forced_opt_args* (which includes -beam-size).
    """
    n_subimages          = imaging_params["n_subimages"]
    imsizex              = imaging_params["imsizex"]
    imsizey              = imaging_params["imsizey"]
    nsigma_automask      = imaging_params["nsigma_automask"]
    nsigma_autothreshold = imaging_params["nsigma_autothreshold"]
    uvtaper              = imaging_params["uvtaper"]
    niter                = imaging_params["niter"]
    with_multiscale      = imaging_params["with_multiscale"]
    scales               = imaging_params["scales"]
    cell_override        = imaging_params.get("cell")

    cell = cell_override if (cell_override and cell_override.lower() != "min") \
        else compute_cell_size(vis_path)[0]

    if output_dir:
        os.chdir(output_dir)

    for robust in robust_list:
        log.info("    robust=%-5.1f  base_name=%s", robust, prefix)
        try:
            image_list, image_statistics, Omaj = mlibs.run_wsclean(
                vis_path,
                robust=robust,
                imsize=imsizex,
                imsizey=imsizey,
                opt_args=forced_opt_args,
                cell=cell,
                base_name=prefix,
                nsigma_automask=nsigma_automask,
                nsigma_autothreshold=nsigma_autothreshold,
                quiet=True,
                nc=n_subimages,
                with_multiscale=with_multiscale,
                scales=scales,
                datacolumn="DATA",
                uvtaper=uvtaper,
                calculate_subband_fluxes=True,
                niter=niter,
            )
            log.info(
                "    OK  robust=%.1f  images: %d",
                robust, len(image_list) if image_list else 0,
            )
        except Exception as exc:
            log.error("    FAILED  %s  robust=%.1f  --  %s",
                      vis_name, robust, exc)

    os.chdir(original_cwd)


def run_uvmatched_imaging(visibilities, imaging_params, output_dir=None):
    """
    Image each MS in *visibilities* at the matched UV coverage using wsclean,
    enforcing a common restoring beam across all datasets.

    Two-pass process
    ----------------
    Pass 1 -- Probe imaging.
        Every MS is imaged once at *robust_probe* with no forced beam size
        (just -circular-beam) to let wsclean converge to the natural beam.
        The major axis Omaj returned by mlibs is recorded for each MS.

    Beam statistics.
        mean, median, min and max of the collected Omaj values are logged.
        The *beam_stat* parameter accepts one or more statistics; Pass 2 runs
        once per value, each time forcing a different common restoring beam.
        The sky_taper is always set to the maximum beam and may optionally be
        applied as a uvtaper (Pass 3) after each Pass 2.

    Pass 2 -- Final imaging with forced common beam (one run per beam_stat).
        opt_args is updated to include -beam-size {restoring_beam_size}.
        Three imaging sub-groups are run:
          main     -- ALL MSs at robust_main (= imaging_params["robust_list"])
          wt_plus  -- high-freq MSs additionally at robust_plus
          wt_minus -- low-freq MSs additionally at robust_minus
        Group membership is determined by the band of each MS (read from the
        OBSERVATION table) vs the band lists in imaging_params.

    Parameters
    ----------
    visibilities   : list of str   UV-matched MS paths.
    imaging_params : dict          See build_imaging_params().
    output_dir     : str or None   Directory for wsclean images.
    """
    section("STAGE 4 -- UV-matched imaging")
    _ensure_dir(output_dir)

    mlibs_mod = import_mlibs(imaging_params.get("mlibs_path"))
    if mlibs_mod is None:
        log.error("  Cannot proceed without mlibs -- aborting Stage 4.")
        separator()
        return

    n_subimages  = imaging_params["n_subimages"]
    robust_main  = imaging_params["robust_list"]
    robust_probe = imaging_params["robust_probe"]
    robust_plus  = imaging_params["robust_plus"]
    robust_minus = imaging_params["robust_minus"]
    bands_plus   = imaging_params["bands_plus"]
    bands_minus  = imaging_params["bands_minus"]
    beam_stat         = imaging_params["beam_stat"]
    apply_sky_taper   = imaging_params["apply_sky_taper"]
    matched_beam_size = imaging_params.get("matched_beam_size")

    vis_list = {
        os.path.splitext(os.path.basename(v))[0]: v
        for v in visibilities
    }

    log.info("Visibilities to image (%d):", len(vis_list))
    for name, path in vis_list.items():
        log.info("  %-45s  %s", name, path)

    separator(" ")
    main_group, wt_plus_group, wt_minus_group = _split_vis_by_band(
        vis_list, bands_plus, bands_minus
    )

    separator(" ")
    log.info("  Stage 4 parameters:")
    log.info("    robust_main  : %s", robust_main)
    log.info("    robust_probe : %s", robust_probe)
    log.info("    robust_plus  : %s  (bands: %s)", robust_plus, bands_plus)
    log.info("    robust_minus : %s  (bands: %s)", robust_minus, bands_minus)
    log.info("    beam_stat         : %s", beam_stat)
    log.info("    apply_sky_taper   : %s", apply_sky_taper)
    log.info("    matched_beam_size : %s", matched_beam_size or "(probe-derived)")
    separator(" ")

    # Write listobs for each MS
    log.info("  Writing listobs files ...")
    for name, vis_path in vis_list.items():
        listobs_file = os.path.join(
            output_dir or os.path.dirname(vis_path) or ".",
            name + ".listobs",
        )
        try:
            casatasks.listobs(vis=vis_path, listfile=listobs_file,
                              overwrite=True)
            log.info("  listobs -> %s", listobs_file)
        except Exception as exc:
            log.warning("  listobs failed for %s: %s", name, exc)

    original_cwd = os.getcwd()

    if matched_beam_size is None:
        # -------------------------------------------------------------------
        # PASS 1 -- Probe imaging: collect natural beam sizes
        # -------------------------------------------------------------------
        separator(" ")
        log.info("  PASS 1 -- Probe imaging (robust=%.1f, no forced beam) ...",
                 robust_probe)

        probe_prefix = "probe_nc{}_cb".format(n_subimages)
        beam_sizes   = {}   # {vis_name: Omaj_arcsec}

        for vis_name, vis_path in vis_list.items():
            log.info("  Probing: %s", vis_name)
            Omaj = _wsclean_probe(
                vis_path, vis_name, imaging_params, mlibs_mod,
                probe_prefix, output_dir, original_cwd,
            )
            if Omaj is not None:
                beam_sizes[vis_name] = Omaj

        if not beam_sizes:
            log.error("  No beam sizes collected from probe imaging -- aborting.")
            separator()
            return

        omaj_values = list(beam_sizes.values())
        beam_stats  = {
            "mean"   : float(np.mean(omaj_values)),
            "median" : float(np.median(omaj_values)),
            "min"    : float(np.min(omaj_values)),
            "max"    : float(np.max(omaj_values)),
        }

        separator(" ")
        log.info("  Beam sizes from probe imaging:")
        for name, omaj in beam_sizes.items():
            log.info("    %-45s  %.4f arcsec", name, omaj)
        log.info("  Statistics:")
        for stat, val in beam_stats.items():
            log.info("    %-8s  %.4f arcsec", stat, val)

        sky_taper_arcsec = beam_stats["max"]
        sky_taper_str    = "{:.2f}arcsec".format(sky_taper_arcsec)
        log.info("  Sky taper (max beam) = %s  (apply_sky_taper=%s)",
                 sky_taper_str, apply_sky_taper)
    else:
        # User supplied beam -- skip probe entirely
        separator(" ")
        log.info("  Skipping PASS 1 (probe) -- using matched_beam_size=%s",
                 matched_beam_size)
        _val = float(matched_beam_size.strip().rstrip("abcdefghijklmnopqrstuvwxyz "))
        beam_stats = {
            "mean"   : _val,
            "median" : _val,
            "min"    : _val,
            "max"    : _val,
        }
        sky_taper_arcsec = _val
        sky_taper_str    = "{:.2f}arcsec".format(sky_taper_arcsec)
        log.info("  Sky taper = %s  (apply_sky_taper=%s)",
                 sky_taper_str, apply_sky_taper)

    valid_stats = [s for s in beam_stat if s in beam_stats]
    invalid     = [s for s in beam_stat if s not in beam_stats]
    for s in invalid:
        log.warning("  Unknown beam_stat '%s' -- skipping.", s)

    for stat in valid_stats:
        restoring_beam_arcsec = beam_stats[stat]
        restoring_beam_str    = "{:.2f}arcsec".format(restoring_beam_arcsec)

        # -------------------------------------------------------------------
        # PASS 2 -- Final imaging with forced common beam
        # -------------------------------------------------------------------
        separator(" ")
        log.info("  PASS 2 [beam_stat=%s] -- forced beam %s ...",
                 stat, restoring_beam_str)

        forced_opt_args = "' -circular-beam -beam-size {} '".format(
            restoring_beam_str
        )
        final_params = dict(imaging_params)
        final_params["uvtaper"] = imaging_params["uvtaper"]

        final_prefix = "im_nc{}_cb_bs_{}".format(n_subimages, restoring_beam_str)
        log.info("  Final image prefix: %s", final_prefix)

        # Sub-group 1: ALL MSs at robust_main
        separator(" ")
        log.info("  Sub-group MAIN (all MSs, robust=%s):", robust_main)
        for vis_name, vis_path in main_group.items():
            log.info("    Imaging: %s", vis_name)
            _wsclean_final(
                vis_path, vis_name, final_params, mlibs_mod,
                final_prefix, forced_opt_args, robust_main,
                output_dir, original_cwd,
            )

        # Sub-group 2: wt_plus MSs at robust_plus (extra robusts only)
        if wt_plus_group and robust_plus:
            separator(" ")
            log.info("  Sub-group WT_PLUS (bands %s, robust=%s):",
                     bands_plus, robust_plus)
            for vis_name, vis_path in wt_plus_group.items():
                log.info("    Imaging: %s", vis_name)
                _wsclean_final(
                    vis_path, vis_name, final_params, mlibs_mod,
                    final_prefix, forced_opt_args, robust_plus,
                    output_dir, original_cwd,
                )
        elif wt_plus_group:
            log.info("  WT_PLUS group has %d MSs but robust_plus is empty -- skipped.",
                     len(wt_plus_group))

        # Sub-group 3: wt_minus MSs at robust_minus (extra robusts only)
        if wt_minus_group and robust_minus:
            separator(" ")
            log.info("  Sub-group WT_MINUS (bands %s, robust=%s):",
                     bands_minus, robust_minus)
            for vis_name, vis_path in wt_minus_group.items():
                log.info("    Imaging: %s", vis_name)
                _wsclean_final(
                    vis_path, vis_name, final_params, mlibs_mod,
                    final_prefix, forced_opt_args, robust_minus,
                    output_dir, original_cwd,
                )
        elif wt_minus_group:
            log.info("  WT_MINUS group has %d MSs but robust_minus is empty -- skipped.",
                     len(wt_minus_group))

        # -------------------------------------------------------------------
        # PASS 3 (optional) -- Sky-tapered imaging of WT_PLUS group only
        # -------------------------------------------------------------------
        if apply_sky_taper:
            separator(" ")
            log.info("  PASS 3 [beam_stat=%s] -- sky-tapered imaging "
                     "(uvtaper=%s, WT_PLUS group only) ...", stat, sky_taper_str)
            if wt_plus_group and robust_plus:
                tapered_params = dict(imaging_params)
                tapered_params["uvtaper"] = [sky_taper_str]
                for vis_name, vis_path in wt_plus_group.items():
                    log.info("    Imaging (tapered): %s", vis_name)
                    _wsclean_final(
                        vis_path, vis_name, tapered_params, mlibs_mod,
                        final_prefix, forced_opt_args, robust_plus,
                        output_dir, original_cwd,
                    )
            else:
                log.warning(
                    "  apply_sky_taper=True but WT_PLUS group is empty or "
                    "robust_plus is not set -- Pass 3 skipped."
                )

    separator()
    log.info("  Stage 4 complete.  beam_stat(s) used: %s", valid_stats)


# ===========================================================================
# Configuration file loader
# ===========================================================================

#: Name of the config file that is auto-loaded from the current working
#: directory when no --config argument is given.
DEFAULT_CONFIG_NAME = "vis_imaging.cfg.yaml"


def load_config(config_path=None):
    """
    Load a YAML configuration file and return its contents as a flat dict
    suitable for use with argparse.set_defaults().

    Search order
    ------------
    1. *config_path* if explicitly given (e.g. via --config on the CLI).
    2. DEFAULT_CONFIG_NAME (vis_imaging.cfg.yaml) in the current working
       directory.
    3. Nothing found -> return empty dict (all defaults come from argparse).

    Key mapping
    -----------
    YAML keys use underscores (matching argparse dest names after hyphens
    are converted), e.g. the CLI flag --do-phaseshift maps to the YAML key
    do_phaseshift.

    Null values in the YAML file are treated as "not set" and are dropped
    so they do not override argparse defaults.

    Parameters
    ----------
    config_path : str or None
        Explicit path to a YAML config file.  When None the default name is
        tried in the current working directory.

    Returns
    -------
    dict  (may be empty if no config file is found or yaml is unavailable)
    """
    if not _YAML_AVAILABLE:
        log.warning(
            "PyYAML is not installed -- config file support disabled.  "
            "Install with:  pip install pyyaml"
        )
        return {}

    # Resolve which file to open
    candidate = config_path or os.path.join(os.getcwd(), DEFAULT_CONFIG_NAME)

    if not os.path.exists(candidate):
        if config_path:
            # User explicitly asked for a file that does not exist -- hard error
            log.error("Config file not found: %s", config_path)
            sys.exit(1)
        else:
            # Auto-detection: no file present, that is fine
            return {}

    log.info("Loading config file: %s", candidate)
    with open(candidate) as fh:
        raw = yaml.safe_load(fh) or {}

    # Drop null values so argparse defaults are not silently clobbered
    cfg = {k: v for k, v in raw.items() if v is not None}
    log.info("  Config keys loaded: %s", sorted(cfg.keys()))
    return cfg


# ===========================================================================
# Imaging parameter dict builder
# ===========================================================================

def build_imaging_params(args):
    """
    Assemble the imaging parameter dict from parsed CLI arguments.
    This dict is passed unchanged to run_native_imaging and run_uvmatched_imaging.
    """
    imsizex = args.imsizex if args.imsizex is not None else args.imsize
    imsizey = args.imsizey if args.imsizey is not None else imsizex
    return {
        "robust_list"          : args.robust,
        "n_subimages"          : args.n_subimages,
        "imsizex"              : imsizex,
        "imsizey"              : imsizey,
        "opt_args"             : args.opt_args,
        "nsigma_automask"      : args.nsigma_automask,
        "nsigma_autothreshold" : args.nsigma_autothreshold,
        "uvtaper"              : args.uvtaper,
        "niter"                : args.niter,
        "with_multiscale"      : args.with_multiscale,
        "scales"               : args.scales,
        "cell"                 : args.cell,
        "mlibs_path"           : args.mlibs_path,
        # Stage 3 -- native imaging specific
        "native_beam_size"     : args.native_beam_size,
        "native_uvtaper"       : args.native_uvtaper,
        # Stage 4 -- UV-matched imaging specific
        "matched_beam_size"    : args.matched_beam_size,
        "robust_probe"         : args.robust_probe,
        "robust_plus"          : args.robust_plus,
        "robust_minus"         : args.robust_minus,
        "bands_plus"           : args.bands_plus,
        "bands_minus"          : args.bands_minus,
        "beam_stat"            : ([args.beam_stat] if isinstance(args.beam_stat, str)
                                   else list(args.beam_stat)),
        "apply_sky_taper"      : args.apply_sky_taper,
    }


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args(argv=None):
    # ------------------------------------------------------------------
    # Two-pass config loading:
    #   Pass 1 -- extract --config (if given) using a minimal pre-parser.
    #             add_help=False so the real parser still owns -h/--help.
    #   Pass 2 -- inject config values as defaults, then do the full parse.
    # This guarantees CLI always beats config, config beats hardcoded defaults.
    # ------------------------------------------------------------------
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(argv)

    cfg = load_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description=(
            "Multi-instrument interferometric visibility pipeline.\n"
            "Stages: phaseshift -> uv-match -> native imaging -> uv-matched imaging.\n"
            "Each stage is independently optional.\n\n"
            "A YAML config file can supply any of the arguments below.\n"
            "Auto-loaded from 'vis_imaging.cfg.yaml' in the current directory,\n"
            "or specify an explicit path with --config."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Config file ---
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help=(
            "Path to a YAML config file.  If not given, "
            "'vis_imaging.cfg.yaml' in the current directory is tried.  "
            "CLI arguments always override config file values."
        ),
    )

    # --- Input data ---
    inp = parser.add_argument_group("Input data")
    inp.add_argument(
        "--vis",
        nargs="+",
        required=False,
        default=None,
        metavar="PATH",
        help="One or more input Measurement Sets to process.",
    )
    inp.add_argument(
        "--ref-vis",
        metavar="PATH",
        default=None,
        help=(
            "Reference MS from which the target phase centre is read for "
            "--do-phaseshift.  Takes priority over --ref-phasecentre if "
            "both are provided."
        ),
    )
    inp.add_argument(
        "--ref-phasecentre",
        metavar="J2000STR",
        default=None,
        help=(
            "Target phase centre as a CASA J2000 string, e.g. "
            "'J2000 12:34:56.7 +12.34.56.7'.  Alternative to --ref-vis "
            "when no reference MS is available.  --ref-vis takes priority "
            "if both are supplied."
        ),
    )

    # --- Stage switches ---
    stages = parser.add_argument_group("Pipeline stage switches")
    stages.add_argument(
        "--do-phaseshift",
        action="store_true", default=False,
        help="Stage 1: phase-shift all input MSs to the phase centre of --ref-vis.",
    )
    stages.add_argument(
        "--do-uv-match",
        action="store_true", default=False,
        help="Stage 2: compute common UV range and split all MSs to that range.",
    )
    stages.add_argument(
        "--do-native-imaging",
        action="store_true", default=False,
        help="Stage 3: image each MS at its native UV coverage (wsclean via mlibs).",
    )
    stages.add_argument(
        "--do-uv-matched-imaging",
        action="store_true", default=False,
        help="Stage 4: image the UV-matched MSs  [placeholder].",
    )

    # --- Output control ---
    out = parser.add_argument_group("Output control")
    out.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help=(
            "Directory where all output MSs and images will be written. "
            "Created if it does not exist.  "
            "Default: outputs land next to their respective input files."
        ),
    )
    out.add_argument(
        "--force-overwrite",
        action="store_true", default=False,
        help=(
            "Delete and reprocess output MSs that already exist on disk. "
            "Default: skip existing outputs and continue."
        ),
    )

    # --- UV-plot extra ---
    extras = parser.add_argument_group("Optional extras")
    extras.add_argument(
        "--plot-vis",
        action="store_true", default=False,
        help=(
            "After UV matching, save a combined UV-coverage PNG with all "
            "UV-matched MSs overlaid in distinct colours "
            "(uses the helpers in ../plotting/plot_vis_python.py)."
        ),
    )

    # --- Imaging parameters ---
    img = parser.add_argument_group(
        "Imaging parameters (Stages 3 and 4, all optional)"
    )
    img.add_argument(
        "--mlibs-path",
        metavar="PATH",
        default=None,
        help="Path to morphen/morphen/ directory containing mlibs.py.",
    )
    img.add_argument(
        "--robust",
        nargs="+",
        type=float,
        default=[-0.5, 0.5],
        metavar="FLOAT",
        help="Briggs robust weighting values to image at. Default: -0.5 0.5",
    )
    img.add_argument(
        "--n-subimages",
        type=int,
        default=3,
        metavar="N",
        help="Number of wsclean sub-band images (spectral windows). Default: 3",
    )
    img.add_argument(
        "--imsize",
        type=int,
        default=2048,
        metavar="PIX",
        help="Image size in pixels (x axis, and y if --imsizex/--imsizey not set). Default: 2048",
    )
    img.add_argument(
        "--imsizex",
        type=int,
        default=None,
        metavar="PIX",
        help="Image size in pixels (x axis). Overrides --imsize for x. Defaults to --imsize if not set.",
    )
    img.add_argument(
        "--imsizey",
        type=int,
        default=None,
        metavar="PIX",
        help="Image size in pixels (y axis). Defaults to --imsize if not set.",
    )
    img.add_argument(
        "--cell",
        default=None,
        metavar="STR|min",
        help=(
            "Pixel (cell) size.  Three modes:\n"
            "  <value>  e.g. '0.04arcsec' -- fixed size used for all MSs.\n"
            "  min      -- compute size analytically per MS (lambda/Bmax/5),\n"
            "              then use the MINIMUM across all MSs so every image\n"
            "              shares the same pixel scale.\n"
            "  (omit)   -- compute analytically per MS independently.\n"
            "Default: per-MS auto."
        ),
    )
    img.add_argument(
        "--nsigma-automask",
        default="4.0",
        metavar="FLOAT",
        help="wsclean auto-mask threshold in sigma. Default: 4.0",
    )
    img.add_argument(
        "--nsigma-autothreshold",
        default="2.0",
        metavar="FLOAT",
        help="wsclean auto-threshold in sigma. Default: 2.0",
    )
    img.add_argument(
        "--uvtaper",
        nargs="+",
        default=[""],
        metavar="STR",
        help="wsclean UV taper(s). Default: none (empty string).",
    )
    img.add_argument(
        "--niter",
        type=int,
        default=100000,
        metavar="N",
        help="Maximum number of wsclean CLEAN iterations. Default: 100000",
    )
    img.add_argument(
        "--with-multiscale",
        action="store_true", default=False,
        help="Enable wsclean multi-scale CLEAN.",
    )
    img.add_argument(
        "--scales",
        default="None",
        metavar="STR",
        help=(
            "Multi-scale CLEAN scales as a comma-separated string, "
            "e.g. '0,40,120,240'. Default: 'None' (auto)."
        ),
    )
    img.add_argument(
        "--opt-args",
        default="' -circular-beam '",
        metavar="STR",
        help=(
            "Extra wsclean arguments passed through mlibs.run_wsclean opt_args. "
            "Default: ' -circular-beam '"
        ),
    )
    img.add_argument(
        "--native-beam-size",
        default=None,
        metavar="ARCSEC",
        help=(
            "Force a fixed circular restoring beam in Stage 3 native imaging, "
            "e.g. '0.5arcsec'. Overrides --opt-args with "
            "'-circular-beam -beam-size {value}'. Default: None (wsclean chooses)."
        ),
    )
    img.add_argument(
        "--native-uvtaper",
        nargs="+",
        default=None,
        metavar="STR",
        help=(
            "UV taper(s) applied only in Stage 3 native imaging. "
            "When set, overrides --uvtaper for Stage 3. "
            "Default: None (uses --uvtaper)."
        ),
    )

    # --- Stage 4 specific: UV-matched imaging ---
    uvm = parser.add_argument_group(
        "Stage 4 -- UV-matched imaging parameters"
    )
    uvm.add_argument(
        "--robust-probe",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help=(
            "Robust value used in the Pass 1 probe imaging to extract "
            "natural beam sizes.  Default: 0.0"
        ),
    )
    uvm.add_argument(
        "--matched-beam-size",
        default=None,
        metavar="ARCSEC",
        help=(
            "Skip the Pass 1 probe and force this beam size directly in "
            "Pass 2 UV-matched imaging, e.g. '0.85arcsec'. "
            "beam_stat entries still control output naming but all resolve "
            "to this value.  Default: None (probe is run normally)."
        ),
    )
    uvm.add_argument(
        "--robust-plus",
        nargs="+",
        type=float,
        default=[1.0, 2.0],
        metavar="FLOAT",
        help=(
            "Additional robust values for the wt_plus band group "
            "(high-frequency MSs that need more short-baseline weight). "
            "Default: 1.0 2.0"
        ),
    )
    uvm.add_argument(
        "--robust-minus",
        nargs="+",
        type=float,
        default=[-0.5],
        metavar="FLOAT",
        help=(
            "Additional robust values for the wt_minus band group "
            "(low-frequency MSs that need more long-baseline weight). "
            "Default: -0.5"
        ),
    )
    uvm.add_argument(
        "--bands-plus",
        nargs="*",
        default=[],
        metavar="BAND",
        help=(
            "Band letters (e.g. X Ka) assigned to the wt_plus group. "
            "MSs in this group are imaged additionally with --robust-plus. "
            "Default: none"
        ),
    )
    uvm.add_argument(
        "--bands-minus",
        nargs="*",
        default=[],
        metavar="BAND",
        help=(
            "Band letters (e.g. L S) assigned to the wt_minus group. "
            "MSs in this group are imaged additionally with --robust-minus. "
            "Default: none"
        ),
    )
    uvm.add_argument(
        "--beam-stat",
        nargs="+",
        default=["mean"],
        choices=["mean", "median", "min", "max"],
        metavar="STAT",
        help=(
            "Statistic(s) used to select the common restoring beam from the "
            "probe imaging beam sizes.  One or more of: mean, median, min, max. "
            "Pass 2 runs once per value.  Default: mean"
        ),
    )
    uvm.add_argument(
        "--apply-sky-taper",
        action="store_true",
        default=False,
        help=(
            "Apply the maximum probe beam size as a UV taper in Pass 2 "
            "to smooth all images to a common angular resolution. "
            "Default: False"
        ),
    )

    # Inject config file values as defaults BEFORE final parse so that
    # any explicit CLI flag still wins.
    if cfg:
        # store_true flags need special treatment: a True value in the config
        # must be set as a default, but argparse will still honour an explicit
        # CLI flag.  False values in the config are left as argparse defaults
        # (already False) so we only inject True overrides.
        bool_flags = {
            "do_phaseshift", "do_uv_match", "do_native_imaging",
            "do_uv_matched_imaging", "force_overwrite", "plot_vis",
            "with_multiscale", "apply_sky_taper",
        }
        clean_cfg = {}
        for k, v in cfg.items():
            if k in bool_flags:
                if v is True:
                    clean_cfg[k] = True
            else:
                clean_cfg[k] = v
        parser.set_defaults(**clean_cfg)

    args = parser.parse_args(argv)

    # --- Validation ---
    if args.vis is None:
        parser.error(
            "--vis is required (either on the CLI or via the config file)."
        )

    if args.do_phaseshift and args.ref_vis is None and not args.ref_phasecentre:
        parser.error(
            "--do-phaseshift requires either --ref-vis or --ref-phasecentre."
        )

    if not any([
        args.do_phaseshift,
        args.do_uv_match,
        args.do_native_imaging,
        args.do_uv_matched_imaging,
    ]):
        parser.error(
            "No stage selected.  Use at least one of: "
            "--do-phaseshift, --do-uv-match, "
            "--do-native-imaging, --do-uv-matched-imaging."
        )

    return args


# ===========================================================================
# Main entry point
# ===========================================================================

def _attach_log_file(output_dir):
    """Add a FileHandler to the vis_imaging logger, writing to output_dir."""
    import datetime
    log_dir = output_dir or "."
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, "vis_imaging_{}.log".format(ts))
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(fh)
    log.info("  Log file: %s", log_path)
    return log_path


def main(argv=None):
    args = parse_args(argv)

    _attach_log_file(args.output_dir)

    separator()
    log.info("  vis_imaging.py  --  visibility processing pipeline")
    separator()

    log.info("Run configuration:")
    log.info("  Config file           : %s",
             args.config or "(auto: {})".format(DEFAULT_CONFIG_NAME))
    log.info("  Input visibilities (%d):", len(args.vis))
    for v in args.vis:
        log.info("    %s", v)
    if args.ref_vis:
        log.info("  Reference visibility  : %s", args.ref_vis)
    if args.ref_phasecentre:
        note = "  (overridden by --ref-vis)" if args.ref_vis else ""
        log.info("  Reference phase centre: %s%s", args.ref_phasecentre, note)
    log.info("  output-dir            : %s", args.output_dir or "(same as input)")
    log.info("  output layout         : native/ + uv_matched/<label>/  (when --output-dir set)")
    log.info("  do-phaseshift         : %s", args.do_phaseshift)
    log.info("  do-uv-match           : %s", args.do_uv_match)
    log.info("  do-native-imaging     : %s", args.do_native_imaging)
    log.info("  do-uv-matched-imaging : %s", args.do_uv_matched_imaging)
    log.info("  plot-vis              : %s", args.plot_vis)
    log.info("  force-overwrite       : %s", args.force_overwrite)
    if args.do_native_imaging or args.do_uv_matched_imaging:
        log.info("  robust values         : %s", args.robust)
        _ix = args.imsizex if args.imsizex is not None else args.imsize
        _iy = args.imsizey if args.imsizey is not None else _ix
        log.info("  imsize                : %d x %d", _ix, _iy)
        log.info("  cell                  : %s",
                 args.cell if args.cell else "auto (per MS)")
        log.info("  n_subimages           : %d", args.n_subimages)
        log.info("  mlibs-path            : %s", args.mlibs_path or "(sys.path)")
    separator()

    # Build imaging params once upfront; passed to both imaging stages
    imaging_params = build_imaging_params(args)

    # -----------------------------------------------------------------------
    # Resolve the sub-directory layout under output_dir BEFORE any stage runs
    # so every stage receives its concrete target directory.
    #
    # out_dirs["native"]     -> e.g. /run01/native/
    # out_dirs["uv_matched"] -> e.g. /run01/uv_matched/eM_C_VLA_Ka/
    # -----------------------------------------------------------------------
    separator(" ")
    log.info("Resolving output directories ...")
    out_dirs = _build_output_dirs(list(args.vis), args.output_dir)

    # -----------------------------------------------------------------------
    # active_vis: tracks the most-processed visibility list at every point.
    # -----------------------------------------------------------------------
    active_vis = list(args.vis)

    # -----------------------------------------------------------------------
    # Stage 1 -- Phaseshift
    # -----------------------------------------------------------------------
    if args.do_phaseshift:
        # --ref-vis takes priority; --ref-phasecentre is used only when
        # --ref-vis is not given.
        phasecenter_arg = None if args.ref_vis else args.ref_phasecentre
        active_vis = run_phaseshift(
            active_vis,
            ref_vis=args.ref_vis,
            ref_phasecenter=phasecenter_arg,
            force_overwrite=args.force_overwrite,
            output_dir=out_dirs["native"],
        )
    else:
        log.info("Skipping Stage 1 (phaseshift).")

    # -----------------------------------------------------------------------
    # Stage 2 -- UV matching
    # -----------------------------------------------------------------------
    if args.do_uv_match:
        active_vis = run_uv_match(
            active_vis,
            plot_vis=args.plot_vis,
            force_overwrite=args.force_overwrite,
            output_dir=out_dirs["uv_matched"],
        )
    else:
        log.info("Skipping Stage 2 (uv-match).")

    # -----------------------------------------------------------------------
    # Determine the correct lists for the two imaging stages.
    #
    #   native_vis    -> _ps.ms files (if phaseshift ran) or the original
    #                    inputs (if phaseshift did NOT run)
    #   uvmatched_vis -> _uvmatch.ms files (active_vis after Stage 2)
    #
    # When phaseshift ran AND output_dir is set, _ps.ms files live in
    # native/ and _uvmatch.ms files live in uv_matched/<label>/ -- two
    # separate directories, so the directory component must also be swapped.
    #
    # When phaseshift did NOT run, native_vis = args.vis regardless of
    # output_dir, because the originals were never copied into native_dir.
    # -----------------------------------------------------------------------
    if args.do_uv_match:
        if args.do_phaseshift:
            # Phase-shifted MSs live in native_dir; reconstruct their paths.
            native_dir = out_dirs["native"]
            if native_dir:
                native_vis = [
                    os.path.join(
                        native_dir,
                        os.path.basename(v).replace("_uvmatch.ms", ".ms"),
                    )
                    for v in active_vis
                ]
            else:
                # _ps.ms and _uvmatch.ms are side by side; strip suffix only.
                native_vis = [v.replace("_uvmatch.ms", ".ms") for v in active_vis]
        else:
            # No phaseshift ran: native vis are the original inputs (they
            # live wherever the user pointed --vis, not in native_dir).
            native_vis = list(args.vis)
        uvmatched_vis = active_vis
    else:
        native_vis    = active_vis
        uvmatched_vis = active_vis

    log.info("Native vis list:")
    for v in native_vis:
        log.info("  %s", v)
    log.info("UV-matched vis list:")
    for v in uvmatched_vis:
        log.info("  %s", v)

    # -----------------------------------------------------------------------
    # Stage 3 -- Native imaging
    # -----------------------------------------------------------------------
    if args.do_native_imaging:
        run_native_imaging(
            native_vis,
            imaging_params,
            output_dir=out_dirs["native"],
        )
    else:
        log.info("Skipping Stage 3 (native imaging).")

    # -----------------------------------------------------------------------
    # Stage 4 -- UV-matched imaging
    # -----------------------------------------------------------------------
    if args.do_uv_matched_imaging:
        if not args.do_uv_match:
            log.warning(
                "Stage 4 (UV-matched imaging) requested but Stage 2 (uv-match) "
                "was not run in this session.  Operating on the current active "
                "visibility list -- make sure UV matching was done in a prior run."
            )
        run_uvmatched_imaging(
            uvmatched_vis,
            imaging_params,
            output_dir=out_dirs["uv_matched"],
        )
    else:
        log.info("Skipping Stage 4 (uv-matched imaging).")

    separator()
    log.info("Pipeline complete.")
    separator()


if __name__ == "__main__":
    main()
