#!/usr/bin/env python3
"""
concat_vis.py
==================
Pipeline for preparing and concatenating multiple interferometric
measurement sets (MS) into a single combined visibility.

Works with data from the same instrument (e.g. multi-epoch VLA) or
from different instruments (e.g. VLA + e-MERLIN).  The two modes
differ in how spectral coverage is handled:

  same-instrument  [--mode same]
      Frequencies are NOT matched across MSs; the goal is to accumulate
      total bandwidth.  Concatenation is still guarded: if any pair of
      MSs lies in clearly non-overlapping bands (gap > band_guard_factor *
      individual_bw) the run aborts with a helpful error message.

  cross-instrument  [--mode cross]
      All non-reference MSs are frequency-matched to the first (or
      --ref-vis) MS before concatenation, using the same channel-by-channel
      overlap algorithm as match_freq_and_split.py.

Stages (each independently switchable)
---------------------------------------
  0. Inspect     Log SPW tables and UV stats for every input MS.  [--do-inspect]
  1. Split-1     Fix SPW IDs; time-average only.                  [--do-split]
  2. Split-2     Channel-average; create WEIGHT_SPECTRUM.          [--do-chanavg]
  3. Phaseshift  Shift all MSs to a common phase centre.           [--do-phaseshift]
  4. Freq-match  Match freq coverage to the reference MS.          [--do-freq-match]
                 (Required for --mode cross.)
  5. Statwt      Compute statistical weights; derive per-MS
                 scaling factors so all MSs contribute equally.    [--do-statwt]
  6. Concat      Concatenate into a single output MS.              [--do-concat]
  7. Wtspectrum  Embed WEIGHT_SPECTRUM in the concatenated MS.     [--do-wtspectrum]

Directory layout
----------------
  --temp-dir   Intermediate MSs (split, chanavg, pshift, freqmatch).
               Default: parent directory of the first input MS.
  --output-dir Final products (concat MS, wtspectrum MS, UV plot, listobs).
               Default: parent directory of the first input MS.

Usage examples
--------------
  # Same-instrument multi-epoch, full pipeline
  python concat_vis.py \\
      --vis epoch1.ms epoch2.ms epoch3.ms \\
      --source-name M82 --mode same \\
      --do-inspect --do-split --do-chanavg \\
      --do-phaseshift --do-statwt --do-concat \\
      --temp-dir /fast_ssd/tmp --output-dir ./concat_out

  # Cross-instrument (e-MERLIN + VLA)
  python concat_vis.py \\
      --vis emerlin.ms vla.ms \\
      --source-name NGC1234 --mode cross \\
      --do-split --do-chanavg --do-phaseshift \\
      --do-freq-match --do-statwt --do-concat \\
      --temp-dir /fast_ssd/tmp --output-dir ./concat_cross

  # Via config file
  python concat_vis.py --config concat_vis_cfg.yaml

Author: concat_vis / ph4ser workflow
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import json
import logging
import os
import shutil
import sys

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):   # graceful no-op fallback
        return iterable

# ---------------------------------------------------------------------------
# CASA (modular installation assumed)
# ---------------------------------------------------------------------------
try:
    import casatools
    import casatasks
    from casatasks import (
        listobs, mstransform, initweights,
        phaseshift, statwt, concat,
    )
    msmd = casatools.msmetadata()
    ms_tool = casatools.ms()        # renamed to avoid shadowing the 'ms' arg name
    tb   = casatools.table()
except ImportError as _err:
    print(
        "[ERROR] Could not import casatools / casatasks.\n"
        "        Make sure you are running inside a modular CASA environment.\n"
        "        Details: {}".format(_err)
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Astropy (used for phase-centre formatting)
# ---------------------------------------------------------------------------
try:
    from astropy.coordinates import SkyCoord
    import astropy.units as _u
    _ASTROPY_AVAILABLE = True
except ImportError:
    _ASTROPY_AVAILABLE = False


# ===========================================================================
# Logging
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("concat_vis")


def separator(char="=", width=72):
    log.info(char * width)


def section(title, char="-", width=72):
    separator(char, width)
    log.info("  %s", title)
    separator(char, width)


# ===========================================================================
# Config
# ===========================================================================

DEFAULT_CONFIG_NAME = "concat_vis_cfg.yaml"


# ===========================================================================
# Path helpers
# ===========================================================================

def _parent_of_first_vis(vis_list):
    """Return the parent directory of the first MS in *vis_list*."""
    first = vis_list[0].rstrip("/")
    parent = os.path.dirname(os.path.abspath(first))
    return parent


def _make_output_path(vis, suffix, dest_dir):
    """
    Build an output path for a processed MS derived from *vis*.

    The base name of *vis* is used; the trailing '.ms' is replaced by
    *suffix*.  The file is placed in *dest_dir*.

    Example
    -------
    >>> _make_output_path('/data/foo.ms', '_split.ms', '/tmp')
    '/tmp/foo_split.ms'
    """
    vis_clean = vis.rstrip("/")
    base      = os.path.basename(vis_clean)
    # Strip all trailing ".ms" variants (handles foo.ms, foo_split.ms, etc.)
    stem      = base[:-3] if base.lower().endswith(".ms") else base
    return os.path.join(dest_dir, stem + suffix)


def _skip_or_overwrite(path, force):
    """
    Return True  -> caller should skip (exists and force=False).
    Return False -> caller should proceed (doesn't exist or force=True).
    """
    if os.path.exists(path):
        if force:
            log.info("  force-overwrite: removing %s", path)
            shutil.rmtree(path)
            return False
        log.info("  Already exists, skipping: %s", path)
        return True
    return False


# ===========================================================================
# SPW / frequency helpers
# ===========================================================================

def get_spwids(vis):
    """
    Return a comma-separated string of the SPW IDs that actually appear
    in scan metadata (filters out SPWs in the SPECTRAL_WINDOW sub-table
    that were never observed - common in some VLA datasets).
    """
    lobs = listobs(vis=vis)
    if lobs is None:
        raise RuntimeError(
            "listobs returned None for {}.  "
            "Cannot determine observed SPW IDs.".format(vis)
        )

    scans = {k: lobs[k] for k in lobs if "scan_" in str(k)}
    if not scans:
        raise RuntimeError(
            "No scan entries found in listobs output for {}.".format(vis)
        )

    unique_ids = set()
    for scan_dict in scans.values():
        for field_info in scan_dict.values():
            spwids = field_info.get("SpwIds", [])
            for sid in spwids:
                unique_ids.add(int(sid))

    if not unique_ids:
        raise RuntimeError(
            "Could not extract any SPW IDs from listobs for {}.".format(vis)
        )

    sorted_ids = sorted(unique_ids)
    spw_str    = ",".join(str(s) for s in sorted_ids)
    log.info("    Observed SPW IDs: %s", spw_str)
    return spw_str


def get_spw_freq_map(vis):
    """
    Return a dict  {spw_id: {freqs, fmin, fmax, fmean, nchan, bw}}
    for every SPW in *vis*.
    """
    msmd.open(vis)
    try:
        nspw   = msmd.nspw()
        result = {}
        for sid in range(nspw):
            freqs = msmd.chanfreqs(sid)
            bws   = msmd.chanwidths(sid)
            result[sid] = {
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
    """Log a per-SPW frequency table for *vis*."""
    spw_map = get_spw_freq_map(vis)
    log.info("  SPW info for: %s", os.path.basename(vis))
    hdr = "    {:<5}  {:<7}  {:<14}  {:<14}  {:<14}  {:<10}  {:<10}".format(
        "SPW", "Nchan", "Fmin [GHz]", "Fmax [GHz]", "Fmean [GHz]",
        "BW [MHz]", "ChanW [kHz]",
    )
    log.info(hdr)
    log.info("    " + "-" * 68)
    for sid, info in sorted(spw_map.items()):
        cw = info["bw"] / max(info["nchan"], 1) / 1e3
        log.info(
            "    {:<5d}  {:<7d}  {:<14.6f}  {:<14.6f}  {:<14.6f}"
            "  {:<10.3f}  {:<10.3f}".format(
                sid, info["nchan"],
                info["fmin"] / 1e9, info["fmax"] / 1e9, info["fmean"] / 1e9,
                info["bw"] / 1e6, cw,
            )
        )
    total_bw = sum(v["bw"]   for v in spw_map.values())
    fmin_all = min(v["fmin"] for v in spw_map.values())
    fmax_all = max(v["fmax"] for v in spw_map.values())
    log.info(
        "    Total SPWs: %d  |  Total BW: %.3f MHz  |  "
        "Full range: %.6f -- %.6f GHz",
        len(spw_map), total_bw / 1e6, fmin_all / 1e9, fmax_all / 1e9,
    )
    return spw_map


def get_chan_avg_map(vis, chan_out_avg=64):
    """
    Compute per-SPW channel bin-widths to average down to at most
    *chan_out_avg* output channels.

    Returns a list of int bin-widths, one per SPW, suitable for the
    mstransform ``chanbin`` parameter.
    """
    msmd.open(vis)
    try:
        nspw         = msmd.nspw()
        chan_per_spw = np.array([len(msmd.chanfreqs(sid)) for sid in range(nspw)])
    finally:
        msmd.done()

    chan_width_avg = [max(1, int(n / chan_out_avg)) for n in chan_per_spw]
    return chan_width_avg


def check_band_overlap(vis_list, guard_factor=0.5):
    """
    Same-instrument mode guard: raise RuntimeError if any MS has less than
    *guard_factor* fractional bandwidth overlap with the reference (first) MS.
    """
    log.info("  Band compatibility check (same-instrument mode) ...")
    maps = {v: get_spw_freq_map(v) for v in vis_list}

    ranges = {}
    for v, m in maps.items():
        ranges[v] = (
            min(info["fmin"] for info in m.values()),
            max(info["fmax"] for info in m.values()),
        )

    ref_lo, ref_hi = ranges[vis_list[0]]
    ref_bw = ref_hi - ref_lo

    problems = []
    for v in vis_list[1:]:
        lo, hi = ranges[v]
        bw     = hi - lo
        ovlp   = max(0.0, min(ref_hi, hi) - max(ref_lo, lo))
        min_bw = min(ref_bw, bw)
        if min_bw > 0 and (ovlp / min_bw) < guard_factor:
            problems.append(
                "  {} (freq {:.4f}-{:.4f} GHz) vs reference {} "
                "({:.4f}-{:.4f} GHz): overlap {:.1f}% < required {:.0f}%".format(
                    os.path.basename(v),   lo / 1e9, hi / 1e9,
                    os.path.basename(vis_list[0]), ref_lo / 1e9, ref_hi / 1e9,
                    100.0 * ovlp / min_bw, 100.0 * guard_factor,
                )
            )
    if problems:
        raise RuntimeError(
            "Band-compatibility check FAILED for same-instrument mode.\n"
            "The following MSs appear to be in a different observing band:\n"
            + "\n".join(problems)
            + "\n\nIf you want to concatenate different bands use "
            "--mode cross with --do-freq-match, or remove the offending MSs."
        )

    log.info("    All MSs appear to be in the same observing band.  OK.")


# ===========================================================================
# UV range helper
# ===========================================================================

def log_uvrange_stats(vis):
    """
    Log basic UV range statistics (klambda) for *vis*.

    Uses the mean frequency across all SPWs to convert baseline lengths
    to wavelength units.  Autocorrelations and fully flagged rows are
    excluded.
    """
    LIGHT_SPEED = 299792458.0   # m/s

    tb.open(vis)
    uvw      = tb.getcol("UVW")      # shape (3, nrows)
    ant1     = tb.getcol("ANTENNA1")
    ant2     = tb.getcol("ANTENNA2")
    flag_row = tb.getcol("FLAG_ROW")
    tb.close()

    good = (ant1 != ant2) & (~flag_row.astype(bool))
    if not np.any(good):
        log.warning("  No usable (cross-correlation, unflagged) rows in %s",
                    os.path.basename(vis))
        return None

    spw_map   = get_spw_freq_map(vis)
    freq_mean = float(np.mean([info["fmean"] for info in spw_map.values()]))
    # UV distance in klambda: |uv| [m] * freq [Hz] / c [m/s] / 1000
    wavelen_m = LIGHT_SPEED / freq_mean     # metres per wavelength

    u  = uvw[0, good]
    v  = uvw[1, good]
    uv_klambda = np.sqrt(u**2 + v**2) / wavelen_m / 1e3

    log.info(
        "  UV stats  %-45s  min=%.3f  max=%.3f  mean=%.3f  [klambda]",
        os.path.basename(vis),
        float(uv_klambda.min()),
        float(uv_klambda.max()),
        float(uv_klambda.mean()),
    )
    return {
        "uvmin"  : float(uv_klambda.min()),
        "uvmax"  : float(uv_klambda.max()),
        "uvmean" : float(uv_klambda.mean()),
    }


# ===========================================================================
# Phase-centre helpers
# ===========================================================================

def get_phase_centre(vis, return_degrees=False):
    """
    Return the phase centre of *vis* as a J2000 CASA-style string,
    e.g. 'J2000 11:28:31.32 +58.33.41.70'.

    If *return_degrees* is True also return (ra_deg, dec_deg) as floats.

    Notes
    -----
    Uses msmd.phasecenter(fieldid=0) which returns the centre for the
    first field.  For single-field observations this is always correct.
    For multi-field MS, pass a different fieldid if needed.
    """
    if not _ASTROPY_AVAILABLE:
        raise RuntimeError(
            "astropy is required for phase-centre handling.  "
            "Install it with:  pip install astropy"
        )

    msmd.open(vis)
    try:
        pc      = msmd.phasecenter(fieldid=0)
        ra_rad  = pc["m0"]["value"]
        dec_rad = pc["m1"]["value"]
    finally:
        msmd.done()

    coord   = SkyCoord(ra=ra_rad * _u.radian, dec=dec_rad * _u.radian,
                       frame="icrs")
    fmt_str = coord.to_string("hmsdms")
    fmt_ra, fmt_dec = fmt_str.split()

    # Convert astropy separators to CASA J2000 format
    fmt_ra  = fmt_ra.replace("h", ":").replace("m", ":").replace("s", "")
    fmt_dec = fmt_dec.replace("d", ".").replace("m", ".").replace("s", "")
    j2000   = "J2000 {} {}".format(fmt_ra, fmt_dec)

    if return_degrees:
        return j2000, float(coord.ra.deg), float(coord.dec.deg)
    return j2000


def angular_separation_arcsec(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """Angular separation between two ICRS positions in arcseconds."""
    d2r = np.pi / 180.0
    cos_a = (
        np.sin(dec1_deg * d2r) * np.sin(dec2_deg * d2r)
        + np.cos(dec1_deg * d2r) * np.cos(dec2_deg * d2r)
          * np.cos((ra1_deg - ra2_deg) * d2r)
    )
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))) * 3600.0


# ===========================================================================
# Frequency-matching helpers  (from match_freq_and_split.py)
# ===========================================================================

def _contiguous_runs(bool_mask):
    """(first, last) index pairs for every contiguous True block in *bool_mask*."""
    runs, inside, start = [], False, 0
    for i, val in enumerate(bool_mask):
        if val and not inside:
            start, inside = i, True
        elif not val and inside:
            runs.append((start, i - 1))
            inside = False
    if inside:
        runs.append((start, len(bool_mask) - 1))
    return runs


def _build_spw_selection(target_spw_map, ref_intervals):
    """
    Map target SPW channels onto reference frequency intervals.

    Returns (spw_str, summary_list) where *spw_str* is a CASA-style
    SPW/channel selection string ready for mstransform.
    """
    parts, summary = [], []
    for sid, info in sorted(target_spw_map.items()):
        freqs = info["freqs"]
        mask  = np.array(
            [any(lo <= f <= hi for lo, hi in ref_intervals) for f in freqs]
        )
        if not np.any(mask):
            log.info("    SPW %2d  [%.6f-%.6f GHz]  no overlap, skipped",
                     sid, info["fmin"] / 1e9, info["fmax"] / 1e9)
            continue
        for c_first, c_last in _contiguous_runs(mask):
            token = "{}:{:d}~{:d}".format(sid, c_first, c_last)
            parts.append(token)
            n_sel = c_last - c_first + 1
            summary.append({
                "spw"     : sid,
                "c_first" : c_first,
                "c_last"  : c_last,
                "fmin"    : float(freqs[c_first]),
                "fmax"    : float(freqs[c_last]),
                "n_sel"   : n_sel,
                "n_tot"   : info["nchan"],
            })
            log.info("    SPW %2d  chans %d~%d  [%.6f-%.6f GHz]  "
                     "(%d/%d, %.1f%%)",
                     sid, c_first, c_last,
                     freqs[c_first] / 1e9, freqs[c_last] / 1e9,
                     n_sel, info["nchan"],
                     100.0 * n_sel / info["nchan"])
    return ",".join(parts), summary



# ===========================================================================
# Stage 0: Inspect
# ===========================================================================

def run_inspect(vis_list):
    """Log SPW tables and UV stats for every MS in *vis_list*."""
    section("Stage 0 - Inspect input visibilities")
    for vis in vis_list:
        separator("-")
        log.info("  %s", vis)
        try:
            log_spw_info(vis)
            log_uvrange_stats(vis)
        except Exception as exc:
            log.warning("  Could not fully inspect %s: %s", vis, exc)
    separator()


# ===========================================================================
# Stage 1: mstransform (fix SPW IDs, time average)
# ===========================================================================

def run_mstransform(vis_list, timebin="6s", correlation="RR,LL",
                    datacolumn="data", force_overwrite=False, temp_dir=None):
    """
    Stage 1: for each MS determine the truly observed SPW IDs via listobs,
    then mstransform to those SPWs only with optional time-averaging.

    Outputs go to *temp_dir*.
    Returns the new list of paths.
    """
    section("Stage 1 - mstransform  (fix SPW IDs, time average)")
    os.makedirs(temp_dir, exist_ok=True)
    out_list = []

    for i, vis in enumerate(tqdm(vis_list, desc="mstransform")):
        vis_temp_dir = os.path.join(temp_dir, "vis_{:d}".format(i))
        os.makedirs(vis_temp_dir, exist_ok=True)
        out = _make_output_path(vis, "_split.ms", vis_temp_dir)
        out_list.append(out)
        if _skip_or_overwrite(out, force_overwrite):
            continue
        log.info("  Processing: %s", os.path.basename(vis))
        try:
            spw_ids = get_spwids(vis)
            # spw_ids           = '0,1,2,3,4,5,6'
            mstransform(
                vis           = vis,
                outputvis     = out,
                spw           = spw_ids,
                timeaverage   = False,
                timebin       = timebin,
                datacolumn    = datacolumn,
                correlation   = correlation,
                keepflags     = True,
                usewtspectrum = True,
            )
            listobs(vis=out, listfile=out + ".listobs", overwrite=True)
            # spw_ids = get_spwids(out)
            log.info("  -> %s", out)
        except Exception as exc:
            log.error("  mstransform FAILED for %s: %s", vis, exc)
            raise

    return out_list


# ===========================================================================
# Stage 2: Channel average + WEIGHT_SPECTRUM
# ===========================================================================

def run_chanavg(vis_list, chan_out=64, force_overwrite=False, temp_dir=None):
    """
    Stage 2: channel-average each MS to ~*chan_out* channels per SPW and
    create a WEIGHT_SPECTRUM column via mstransform + initweights.

    Outputs go to *temp_dir*.
    Returns the new list of paths.
    """
    section("Stage 2 - Channel average + WEIGHT_SPECTRUM")
    os.makedirs(temp_dir, exist_ok=True)
    out_list = []

    for i, vis in enumerate(tqdm(vis_list, desc="chanavg")):
        vis_temp_dir = os.path.join(temp_dir, "vis_{:d}".format(i))
        os.makedirs(vis_temp_dir, exist_ok=True)
        out = _make_output_path(vis, "_chanavg.ms", vis_temp_dir)
        out_list.append(out)
        if _skip_or_overwrite(out, force_overwrite):
            continue
        log.info("  Processing: %s", os.path.basename(vis))
        try:
            chan_bins = get_chan_avg_map(vis, chan_out_avg=chan_out)
            log.info("  Channel bin-widths (per SPW): %s", chan_bins)

            # Initialise WEIGHT_SPECTRUM on the *input* before transforming
            initweights(vis=vis, wtmode="weight", dowtsp=True)

            mstransform(
                vis           = vis,
                outputvis     = out,
                datacolumn    = "data",
                keepflags     = True,
                usewtspectrum = True,
                regridms      = False,
                chanaverage   = True,
                chanbin       = chan_bins,
                timeaverage   = False,
            )

            # Remove the channelised weight column from the *input* (clean-up)
            initweights(vis=vis, wtmode="delwtsp")

            listobs(vis=out, listfile=out + ".listobs", overwrite=True)
            log.info("  -> %s", out)
        except Exception as exc:
            log.error("  chanavg FAILED for %s: %s", vis, exc)
            raise

    return out_list


# ===========================================================================
# Stage 3: Phaseshift
# ===========================================================================

def run_phaseshift(vis_list, ref_idx=0, ref_phasecentre=None,
                   force_overwrite=False, temp_dir=None):
    """
    Stage 3: shift all MSs (except the reference) to the reference phase
    centre.

    Parameters
    ----------
    vis_list       : list of paths (current active list)
    ref_idx        : index of the reference MS inside *vis_list*
    ref_phasecentre: optional override J2000 string; if None the phase
                     centre is read from vis_list[ref_idx]
    temp_dir       : directory for shifted MSs

    The reference MS itself is left unchanged and copied to the output list
    at the same position.

    Returns the new list of paths.
    """
    section("Stage 3 - Phase shift to common phase centre")
    os.makedirs(temp_dir, exist_ok=True)

    ref_vis = vis_list[ref_idx]

    # Determine reference phase centre
    if ref_phasecentre:
        pcentre  = ref_phasecentre
        ra_ref   = None
        dec_ref  = None
        log.info("  Using user-supplied phase centre: %s", pcentre)
    else:
        pcentre, ra_ref, dec_ref = get_phase_centre(ref_vis, return_degrees=True)
        log.info("  Reference phase centre from %s", os.path.basename(ref_vis))
        log.info("    %s", pcentre)

    out_list = []
    for i, vis in enumerate(tqdm(vis_list, desc="phaseshift")):
        if i == ref_idx:
            # Reference MS is kept as-is
            out_list.append(vis)
            log.info("  [ref] %s  (unchanged)", os.path.basename(vis))
            continue

        vis_temp_dir = os.path.join(temp_dir, "vis_{:d}".format(i))
        os.makedirs(vis_temp_dir, exist_ok=True)
        out = _make_output_path(vis, "_pshift.ms", vis_temp_dir)
        out_list.append(out)
        if _skip_or_overwrite(out, force_overwrite):
            continue

        try:
            old_pc, ra_k, dec_k = get_phase_centre(vis, return_degrees=True)
            sep = (
                angular_separation_arcsec(ra_ref, dec_ref, ra_k, dec_k)
                if ra_ref is not None else float("nan")
            )
            log.info("  %s", os.path.basename(vis))
            log.info("    old centre : %s", old_pc)
            log.info("    shift      : %.4f arcsec  ->  %s", sep, pcentre)
            phaseshift(vis=vis, phasecenter=pcentre, outputvis=out)
            log.info("  -> %s", out)
        except Exception as exc:
            log.error("  phaseshift FAILED for %s: %s", vis, exc)
            raise

    return out_list


# ===========================================================================
# Stage 4: Frequency matching (cross-instrument)
# ===========================================================================

def run_freq_match(vis_list, ref_idx=0, freq_padding_mhz=500.0,
                   datacolumn="data", keepflags=True,
                   force_overwrite=False, temp_dir=None):
    """
    Stage 4: match the frequency coverage of every non-reference MS to the
    reference MS using per-channel overlap (gap-aware).

    Parameters
    ----------
    vis_list        : list of paths (current active list)
    ref_idx         : index of the reference MS inside *vis_list*
    freq_padding_mhz: padding on each side of each reference SPW interval
    temp_dir        : directory for frequency-matched MSs

    Returns the new list of paths.
    """
    section("Stage 4 - Frequency matching to reference MS (cross-instrument)")
    os.makedirs(temp_dir, exist_ok=True)

    ref_vis      = vis_list[ref_idx]
    padding_hz   = freq_padding_mhz * 1e6

    log.info("  Reference MS : %s", ref_vis)
    ref_map      = get_spw_freq_map(ref_vis)

    # Build reference intervals, one per SPW (preserves gaps between SPWs)
    ref_intervals = [
        (info["fmin"] - padding_hz, info["fmax"] + padding_hz)
        for info in sorted(ref_map.values(), key=lambda x: x["fmin"])
    ]
    log.info("  Reference SPW intervals (+%.1f MHz padding):", freq_padding_mhz)
    for lo, hi in ref_intervals:
        log.info("    %.6f - %.6f GHz", lo / 1e9, hi / 1e9)

    out_list = []
    for i, vis in enumerate(tqdm(vis_list, desc="freq-match")):
        if i == ref_idx:
            out_list.append(vis)
            log.info("  [ref] %s  (unchanged)", os.path.basename(vis))
            continue

        vis_temp_dir = os.path.join(temp_dir, "vis_{:d}".format(i))
        os.makedirs(vis_temp_dir, exist_ok=True)
        out = _make_output_path(vis, "_freqmatch.ms", vis_temp_dir)
        out_list.append(out)
        if _skip_or_overwrite(out, force_overwrite):
            continue

        log.info("  Matching: %s", os.path.basename(vis))
        target_map = get_spw_freq_map(vis)
        spw_str, summary = _build_spw_selection(target_map, ref_intervals)

        if not spw_str:
            raise RuntimeError(
                "No spectral overlap between reference MS ({}) "
                "and target ({}).  Check that both are in the same "
                "observing band.".format(
                    os.path.basename(ref_vis), os.path.basename(vis))
            )

        n_sel = sum(s["n_sel"] for s in summary)
        n_tot = sum(s["n_tot"] for s in summary)
        log.info("    Selected %d / %d channels  SPW string: %s",
                 n_sel, n_tot, spw_str)

        try:
            mstransform(
                vis           = vis,
                outputvis     = out,
                spw           = spw_str,
                datacolumn    = datacolumn,
                keepflags     = keepflags,
                usewtspectrum = True,   # preserve WEIGHT_SPECTRUM if present
            )
            listobs(vis=out, listfile=out + ".listobs", overwrite=True)
            log.info("  -> %s", out)
        except Exception as exc:
            log.error("  freq-match FAILED for %s: %s", vis, exc)
            raise

    return out_list


# ===========================================================================
# Stage 5: Statistical weights + scaling factors
# ===========================================================================

def run_statwt(vis_list, timebin="12s", statalg="chauvenet", preview=True,
               force_overwrite=False):
    """
    Stage 5: run CASA ``statwt`` on each MS and compute per-MS weight
    scaling factors so that all MSs contribute roughly equally to the
    concatenated product.

    Scaling strategy
    ----------------
    For each MS, ``statwt`` returns the mean weight.  The scale factor is:

        wt_factor_i = mean_overall / mean_i

    so that high-weight MSs are attenuated and low-weight MSs are boosted
    until all have the same effective mean weight.

    Parameters
    ----------
    preview : if True, weights are computed but NOT written back to the MS
              (useful for a dry-run inspection before committing).
    force_overwrite : if False and a per-MS sentinel cache file exists from a
              previous run with the same preview mode, statwt is skipped and
              the cached mean weight is reused.

    Returns
    -------
    scale_factors : list[float]
    statwt_results : list[dict]
    """
    section("Stage 5 - Statistical weights and scaling factors")

    results = []
    for vis in tqdm(vis_list, desc="statwt"):
        # Per-MS sentinel: <vis>.statwt_cache  (JSON: {"mean": float, "preview": bool})
        sentinel = vis + ".statwt_cache"
        if not force_overwrite and os.path.exists(sentinel):
            try:
                with open(sentinel) as _f:
                    cached = json.load(_f)
                if cached.get("preview") == preview:
                    log.info("  Cached statwt result: %s  (mean=%.6g)",
                             os.path.basename(vis), cached["mean"])
                    results.append({"mean": cached["mean"]})
                    continue
            except Exception as exc:
                log.warning("  Could not read statwt cache (%s); re-running.", exc)

        log.info("  Running statwt on: %s", os.path.basename(vis))
        # Build kwargs carefully - chanbin is not supported in all CASA builds
        kwargs = dict(
            vis        = vis,
            preview    = preview,
            datacolumn = "data",
            timebin    = timebin,
            statalg    = statalg,
        )
        try:
            res = statwt(**kwargs)
        except TypeError:
            # Fallback: drop any unsupported keyword that CASA rejected
            log.warning("  statwt with full kwargs failed; retrying with "
                        "minimal kwargs.")
            res = statwt(vis=vis, preview=preview, datacolumn="data")

        results.append(res if isinstance(res, dict) else {})
        mean_wt = results[-1].get("mean", float("nan"))
        log.info("    mean weight: %.6g", mean_wt)

        try:
            with open(sentinel, "w") as _f:
                json.dump({"mean": mean_wt, "preview": preview}, _f)
        except Exception as exc:
            log.warning("  Could not write statwt cache: %s", exc)

    means = np.array([r.get("mean", np.nan) for r in results])
    if np.all(np.isnan(means)):
        log.warning("  statwt returned no mean weights; scale factors set to 1.")
        scale_factors = [1.0] * len(vis_list)
    else:
        overall_mean  = float(np.nanmean(means))
        scale_factors = [
            float(overall_mean / m) if not np.isnan(m) else 1.0
            for m in means
        ]

    log.info("  Per-MS weight summary:")
    log.info("  %s", "-" * 60)
    for vis, m, f in zip(vis_list, means, scale_factors):
        log.info("  %-50s  mean=%.5g  factor=%.4f",
                 os.path.basename(vis), m, f)

    return scale_factors, results


# ===========================================================================
# Source-name and band-label helpers
# ===========================================================================

# Band boundaries in Hz  (ITU / standard radio-astronomy convention)
_BAND_EDGES = [
    ("L",  1.0e9,  2.0e9),
    ("S",  2.0e9,  4.0e9),
    ("C",  4.0e9,  8.0e9),
    ("X",  8.0e9, 12.0e9),
    ("Ku",12.0e9, 18.0e9),
    ("K", 18.0e9, 26.5e9),
    ("Ka",26.5e9, 40.0e9),
    ("Q", 40.0e9, 50.0e9),
]


def get_field_name(vis):
    """
    Return the name of the first field in *vis* by reading the FIELD
    sub-table directly with casatools.table.

    Returns an empty string on any failure so callers can always fall back
    gracefully.
    """
    try:
        tb.open(os.path.join(vis.rstrip("/"), "FIELD"))
        names = tb.getcol("NAME")
        tb.close()
        if len(names):
            # Strip whitespace and replace spaces/slashes with underscores
            name = str(names[0]).strip().replace(" ", "_").replace("/", "_")
            return name
    except Exception as exc:
        log.warning("  Could not read FIELD table from %s: %s", vis, exc)
        try:
            tb.close()
        except Exception:
            pass
    return ""


def get_band_label(vis):
    """
    Return a single band-label string (e.g. 'C', 'L', 'Ka') for *vis*
    based on its mean observing frequency.

    Tries the standard L/S/C/X/Ku/K/Ka/Q boundaries.  Returns an empty
    string if the frequency falls outside all defined ranges or if reading
    the MS fails, so callers can always fall back gracefully.
    """
    try:
        spw_map   = get_spw_freq_map(vis)
        freq_mean = float(np.mean([info["fmean"] for info in spw_map.values()]))
        for label, lo, hi in _BAND_EDGES:
            if lo <= freq_mean < hi:
                log.info("  Band label: %s  (mean freq %.4f GHz)",
                         label, freq_mean / 1e9)
                return label
        log.warning("  Mean freq %.4f GHz is outside all defined bands; "
                    "band label omitted.", freq_mean / 1e9)
    except Exception as exc:
        log.warning("  Could not determine band label from %s: %s", vis, exc)
    return ""


def build_concat_name(source_name, band_label, n_vis):
    """
    Construct the base name (no path, no .ms) for the concatenated MS.

    Pattern:  <source>_<band>_<N>x   e.g. M82_C_3x
    If *band_label* is empty:          <source>_<N>x   e.g. M82_3x
    """
    parts = [source_name]
    if band_label:
        parts.append(band_label)
    parts.append("{}x".format(n_vis))
    return "_".join(parts)


# ===========================================================================
# POINTING subtable sanity check / repair
# ===========================================================================
#
# Background
# ----------
# ``concat`` can leave a POINTING subtable in an inconsistent state: when the
# MS being appended has no POINTING rows but the target does, CASA's
# ``MSConcat::copyPointing`` takes its "Result won't have one" branch and
# discards the row data *without* resetting the row count stored in
# ``table.dat``.  The result is a table whose header claims N rows while its
# StandardStMan index holds none.
#
# Such a table reads back as::
#
#     RuntimeError: SSMIndex::getIndex - access to non-existing row 0
#                   in column TIME of table <ms>/POINTING
#
# The damage is silent and sticky: split / mstransform / phaseshift all copy
# the broken subtable through faithfully, so it propagates down an entire
# processing chain and only surfaces at the *next* concat, which dies with
# ``access to non-existing row <N-1> in column DIRECTION``.
#
# Because the row data is already gone by the time we see it, the only
# meaningful repair is to make the table honestly empty.  That is exactly what
# ``concat(copypointing=False)`` would have produced anyway, and it costs
# nothing scientifically -- POINTING is not used by imaging for these data.

_POINTING_PROBE_COLUMNS = ("TIME", "DIRECTION", "ANTENNA_ID")


def check_pointing_table(vis):
    """
    Inspect the POINTING subtable of *vis*.

    Returns ``(status, nrows)`` where status is one of:

    ``"absent"``
        No POINTING subtable at all.
    ``"empty"``
        ``nrows == 0``.  Harmless -- concat copes with this.
    ``"ok"``
        Row data is readable.
    ``"corrupt"``
        Header claims ``nrows > 0`` but the row data is unreadable.  This MS
        will break ``concat``.
    """
    pt = os.path.join(vis, "POINTING")
    if not os.path.isdir(pt):
        return "absent", 0

    t = casatools.table()
    try:
        t.open(pt, nomodify=True)
    except Exception:
        return "corrupt", -1
    try:
        nrows = t.nrows()
        if nrows == 0:
            return "empty", 0
        cols = t.colnames()
        for col in _POINTING_PROBE_COLUMNS:
            if col not in cols:
                continue
            try:
                t.getcell(col, 0)
            except Exception:
                return "corrupt", nrows
        return "ok", nrows
    finally:
        t.close()


def repair_pointing_table(vis, backup_dir=None):
    """
    Replace a corrupt POINTING subtable of *vis* with an empty one that has an
    identical layout (columns, column descriptors, MEASINFO / QuantumUnits
    keywords, data managers and table keywords are all preserved).

    If *backup_dir* is given, the damaged table is moved there instead of being
    deleted, so nothing is destroyed irreversibly.

    Returns True if a repair was performed.
    """
    pt  = os.path.join(vis, "POINTING")
    tmp = os.path.join(vis, "_POINTING_damaged_{}".format(os.getpid()))

    shutil.move(pt, tmp)
    try:
        t = casatools.table()
        t.open(tmp, nomodify=True)
        try:
            # norows=True never touches the row data, so this succeeds even
            # though the rows themselves are unreadable.
            clone = t.copy(newtablename=pt, deep=False,
                           valuecopy=True, norows=True)
            clone.close()
        finally:
            t.close()
    except Exception:
        # Put the original back so we never leave the MS worse than we found it
        if not os.path.isdir(pt):
            shutil.move(tmp, pt)
        raise

    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        dest = os.path.join(
            backup_dir, "{}_POINTING_damaged".format(os.path.basename(vis)))
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(tmp, dest)
        log.info("    damaged table kept at %s", dest)
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return True


def sanitise_pointing_tables(vis_list, repair=True, backup_dir=None):
    """
    Check every MS in *vis_list* and, when *repair* is True, fix the ones whose
    POINTING subtable would abort ``concat``.

    Returns the list of MSs that were found to be corrupt.
    """
    section("Stage 6a - POINTING subtable check")
    bad = []
    for v in vis_list:
        status, nrows = check_pointing_table(v)
        log.info("  %-8s nrows=%-8s %s", status, nrows, os.path.basename(v))
        if status == "corrupt":
            bad.append(v)

    if not bad:
        log.info("  All POINTING subtables are consistent.")
        return bad

    log.warning("  %d MS(s) carry a damaged POINTING subtable "
                "(header row count without row data).", len(bad))
    log.warning("  This is the scar of an earlier concat and *will* abort this "
                "one with 'SSMIndex::getIndex - access to non-existing row'.")

    if not repair:
        log.warning("  Repair disabled (--no-repair-pointing); concat is "
                    "expected to fail.")
        return bad

    for v in bad:
        log.info("  Repairing POINTING of %s ...", os.path.basename(v))
        repair_pointing_table(v, backup_dir=backup_dir)
        status, nrows = check_pointing_table(v)
        if status not in ("empty", "ok"):
            raise RuntimeError(
                "POINTING repair of {} did not succeed (status={})".format(
                    v, status))
        log.info("    -> now '%s' (nrows=%d)", status, nrows)
    return bad


# ===========================================================================
# Stage 6: Concatenate
# ===========================================================================

def run_concat(vis_list, source_name, output_dir, ref_vis=None,
               scale_factors=None, dirtol="1000.0arcsec", freqtol="1MHz",
               force_overwrite=False, repair_pointing=True):
    """
    Stage 6: concatenate all MSs in *vis_list*.

    Output name pattern:  ``<source>_<band>_<N>x_concat.ms``
    where N = len(vis_list) and band is derived from the mean frequency
    of *ref_vis* (or vis_list[0]).  If band detection fails the band token
    is omitted: ``<source>_<N>x_concat.ms``.

    *scale_factors* is an optional list of per-MS weight scaling values;
    ``None`` means no scaling.

    Returns the path to the concatenated MS.
    """
    section("Stage 6 - Concatenate")
    os.makedirs(output_dir, exist_ok=True)

    # --- Determine band label (best-effort) --------------------------------
    probe_vis = ref_vis if ref_vis else vis_list[0]
    try:
        band_label = get_band_label(probe_vis)
    except Exception:
        band_label = ""

    # --- Build output name --------------------------------------------------
    base_name = build_concat_name(source_name, band_label, len(vis_list))
    out_ms    = os.path.join(output_dir, "{}_concat.ms".format(base_name))
    log.info("  Output name stem: %s  (band=%s, N=%d)",
             base_name, band_label or "<none>", len(vis_list))

    if _skip_or_overwrite(out_ms, force_overwrite):
        return out_ms

    log.info("  Output : %s", out_ms)
    log.info("  Inputs (%d):", len(vis_list))
    for v in vis_list:
        log.info("    %s", v)

    # A damaged POINTING subtable in *any* input aborts concat part-way through,
    # after it has already copied the chronologically-first MS to the output.
    sanitise_pointing_tables(vis_list, repair=repair_pointing,
                             backup_dir=os.path.join(output_dir,
                                                     "damaged_subtables"))

    kwargs = dict(
        vis          = list(vis_list),
        concatvis    = out_ms,
        dirtol       = dirtol,
        copypointing = True,
        timesort     = True,
        freqtol      = freqtol,
    )
    if scale_factors is not None:
        kwargs["visweightscale"] = [float(f) for f in scale_factors]
        log.info("  Weight scale factors: %s",
                 ["{:.4f}".format(f) for f in scale_factors])

    try:
        concat(**kwargs)
    except Exception as exc:
        log.error("  concat FAILED: %s", exc)
        raise

    listobs(vis=out_ms, listfile=out_ms + ".listobs", overwrite=True)
    log.info("  -> %s", out_ms)
    return out_ms


# ===========================================================================
# Stage 7: Embed WEIGHT_SPECTRUM in concat MS
# ===========================================================================

def run_wtspectrum(concat_ms, output_dir, force_overwrite=False):
    """
    Stage 7: re-run mstransform to ensure the concatenated MS has a
    WEIGHT_SPECTRUM column.

    Output is ``<output_dir>/<source_stem>_concat_wts.ms``.
    Returns the path to the new MS.
    """
    section("Stage 7 - Embed WEIGHT_SPECTRUM in concatenated MS")
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.basename(concat_ms.rstrip("/"))
    stem = base[:-3] if base.lower().endswith(".ms") else base
    out  = os.path.join(output_dir, stem + "_wts.ms")

    if _skip_or_overwrite(out, force_overwrite):
        return out

    try:
        mstransform(
            vis           = concat_ms,
            outputvis     = out,
            datacolumn    = "data",
            timeaverage   = False,
            usewtspectrum = True,
            keepflags     = True,
        )
        listobs(vis=out, listfile=out + ".listobs", overwrite=True)
        log.info("  -> %s", out)
    except Exception as exc:
        log.error("  wtspectrum FAILED: %s", exc)
        raise

    return out


# ===========================================================================
# UV coverage comparison plot
# ===========================================================================


def _get_available_memory_gb():
    """
    Return currently available system RAM in GB.
    Tries psutil first, then /proc/meminfo, returns None on failure.
    """
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


def _auto_plot_params(vis, mem_target_gb, fallback_chunk=4, fallback_ds=150):
    """
    Estimate safe ``chunk_size`` and ``downsample_factor`` for
    plot_uv_comparison so that plotting *vis* stays within *mem_target_gb*.

    Memory model
    ------------
    The dominant allocations inside _plot_one are:

      UVW read   :  n_rows x 3 x 8 bytes  (float64 columns)
      ant columns:  n_rows x 2 x 4 bytes
      freq tile  :  n_chunks x (2 x n_rows) x 2 x 8 bytes
                    (u_kl + v_kl, conjugate doubles the baseline axis)

    where  n_chunks = n_freq / chunk_size.

    We solve for chunk_size such that the tile fits in half the budget
    (the other half covers the column reads), then set downsample_factor
    so that the total number of matplotlib data-points is ≤ 2 million
    (empirically a comfortable ceiling before the figure itself gets heavy).

    Returns (chunk_size: int, downsample_factor: int).
    Falls back to (*fallback_chunk*, *fallback_ds*) on any error.
    """
    try:
        tb.open(vis)
        n_rows = tb.nrows()
        tb.close()

        msmd.open(vis)
        nspw   = msmd.nspw()
        n_freq = sum(len(msmd.chanfreqs(s)) for s in range(nspw))
        msmd.done()
    except Exception as exc:
        log.warning("  _auto_plot_params: could not read metadata from %s: %s",
                    os.path.basename(vis), exc)
        try:
            tb.close()
        except Exception:
            pass
        try:
            msmd.done()
        except Exception:
            pass
        return fallback_chunk, fallback_ds

    target_bytes = mem_target_gb * 1e9

    # ── chunk_size: tile memory = n_chunks x 2*n_rows x 2 x 8
    #    Budget for tile = target / 2  (leave room for column reads)
    #    n_chunks ≤ target / (2 * 2 * n_rows * 2 * 8)
    #    chunk_size ≥ n_freq / n_chunks
    tile_budget = target_bytes / 2.0
    bytes_per_chunk_row = 2 * 2 * 8   # u_kl + v_kl, conjugate axis
    max_chunks = max(1, int(tile_budget / (n_rows * bytes_per_chunk_row)))
    chunk_size = max(1, int(np.ceil(n_freq / max_chunks)))

    # Recompute actual n_chunks after snapping chunk_size
    n_chunks = max(1, n_freq // chunk_size)

    # ── downsample_factor: keep total matplotlib points ≤ 2 million
    #    Points per plot call ~ n_chunks x 2 x n_rows  (conjugate doubles rows)
    target_points  = 2_000_000
    raw_points     = n_chunks * 2 * n_rows
    downsample_factor = max(1, int(np.ceil(raw_points / target_points)))

    log.info(
        "  Plot params for %-40s  n_rows=%d  n_freq=%d  "
        "chunk_size=%d  downsample=%d  "
        "(tile~%.2f GB, pts~%.1fM)",
        os.path.basename(vis), n_rows, n_freq,
        chunk_size, downsample_factor,
        (n_chunks * 2 * n_rows * bytes_per_chunk_row) / 1e9,
        raw_points / downsample_factor / 1e6,
    )
    return chunk_size, downsample_factor


def plot_uv_comparison(vis_list, concat_ms=None, output_dir=".",
                       chunk_size=None, downsample_factor=None,
                       mem_fraction=0.3):
    """
    Plot UV coverage for all input MSs in different colours; optionally
    overlay the concatenated MS in semi-transparent silver (background).

    chunk_size and downsample_factor control frequency resolution and
    point density.  When left as None (default) both are computed
    automatically so that plotting each MS stays within
    ``mem_fraction`` x available RAM (default 50%%).

    Saves ``<output_dir>/uv_coverage_comparison.png``.
    """
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend - safe on HPC nodes
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    section("UV coverage comparison plot")
    os.makedirs(output_dir, exist_ok=True)

    LIGHT_SPEED = 299792458.0

    # Import the exact same helpers that plot_uvwave uses
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from plot_vis_python import _spw_meta, _iter_spw_chunks, _bl_sort_index
    except Exception as _pvp_err:
        log.error("  Cannot import helpers from plot_vis_python.py: %s", _pvp_err)
        log.error("  Skipping UV coverage plot.")
        return None

    avail_gb = _get_available_memory_gb() or 2.0
    log.info("  Available RAM: %.1f GB", avail_gb)
    # Chunk rows exactly as in plot_uvwave (UVW only: 3 float64 per row)
    chunk_rows = min(500_000, max(10_000,
                                  int(avail_gb * mem_fraction * 1e9 / (3 * 8))))

    # Colour map - twilight_shifted, matching plot_uvwave
    try:
        # cmap = matplotlib.colormaps["twilight_shifted"]
        cmap = matplotlib.colormaps["tab20"]
    except AttributeError:
        # cmap = matplotlib.cm.get_cmap("twilight_shifted")
        cmap = matplotlib.cm.get_cmap("tab20")

    n_ms = max(len(vis_list), 1)
    colors = [cmap(i / n_ms) for i in range(len(vis_list))]

    fig, ax = plt.subplots(figsize=(8, 8))

    def _plot_one(vis, color, alpha=0.4):
        """UV coverage, ported exactly from plot_uvwave in plot_vis_python.py.

        SPW-by-SPW iteration, per-baseline accumulation, cs=16 frequency
        chunks, markersize=0.05 - identical to plot_uvwave.  Only the colour
        is different: per-MS instead of per-baseline so input MSs remain
        visually distinguishable in the overlay.
        """
        try:
            dd_info = _spw_meta(vis, tb, msmd)
            dd_ids  = sorted(dd_info.keys())
        except Exception as exc:
            log.warning("  Could not read SPW info from %s: %s",
                        os.path.basename(vis), exc)
            try:
                msmd.done()
            except Exception:
                pass
            return

        try:
            tb.open(vis + '/ANTENNA')
            n_ant = int(tb.nrows())
            tb.close()
        except Exception as exc:
            log.warning("  Could not read ANTENNA table from %s: %s",
                        os.path.basename(vis), exc)
            return

        bl_ukl = {}
        bl_vkl = {}

        for chunk in _iter_spw_chunks(vis, ms_tool, dd_ids, chunk_rows,
                                       ["uvw", "antenna1", "antenna2", "flag_row"]):
            dd_id    = chunk["dd_id"]
            uvw      = chunk["uvw"]
            ant1     = chunk["antenna1"].astype(np.int32)
            ant2     = chunk["antenna2"].astype(np.int32)
            flag_row = chunk["flag_row"]

            good = (ant1 != ant2) & (~flag_row)
            if not good.any():
                continue

            g_u  = uvw[0, good]
            g_v  = uvw[1, good]
            g_a1 = ant1[good]
            g_a2 = ant2[good]

            # 16-channel frequency chunks - identical to plot_uvwave
            freqs  = dd_info[dd_id]["chan_freqs"]
            n_freq = len(freqs)
            if n_freq == 0:
                continue
            # Clamp chunk size when SPW has fewer than 16 channels (e.g. after freq-match trimming)
            cs  = min(16, n_freq)
            n_c = max(1, n_freq // cs)
            cf  = freqs[:n_c * cs].reshape(n_c, cs).mean(axis=1)
            wl     = (LIGHT_SPEED / cf).reshape(-1, 1)

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

        for key in bl_ukl:
            if not bl_ukl[key]:
                continue
            uw = np.concatenate(bl_ukl[key])
            vw = np.concatenate(bl_vkl[key])
            ax.plot(uw, vw, ".", markersize=0.05,
                    color=color, alpha=alpha,
                    rasterized=True, linewidth=0)

    # # Plot concat FIRST (background) so individual MSs render on top
    # if concat_ms and os.path.exists(concat_ms):
    #     log.info("  Plotting concat MS (background): %s",
    #              os.path.basename(concat_ms))
    #     _plot_one(concat_ms, color="silver", alpha=0.99)

    # Individual MSs plotted on top in distinct colours
    for vis, col in zip(vis_list, colors):
        log.info("  Plotting: %s", os.path.basename(vis))
        _plot_one(vis, col)

    # Legend  (concat entry first so it reads as background reference)
    handles = []
    # if concat_ms and os.path.exists(concat_ms):
    #     handles.append(
    #         mlines.Line2D([], [], color="silver", marker=".", linestyle="None",
    #                       markersize=6,
    #                       label="concat: " + os.path.basename(concat_ms))
    #     )
    handles += [
        mlines.Line2D([], [], color=colors[i], marker=".", linestyle="None",
                      markersize=6, label=os.path.basename(vis_list[i]))
        for i in range(len(vis_list))
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7)
    ax.set_xlabel(r"$u\;[\mathrm{k}\lambda]$")
    ax.set_ylabel(r"$v\;[\mathrm{k}\lambda]$")
    # Equal data-unit scaling without forcing the axes box to be square;
    # bbox_inches="tight" at save time will make the figure compact.
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_title("Total $uv$ coverage")

    fig_path = os.path.join(output_dir, f"{os.path.basename(concat_ms).replace('.ms', '')}_uv_coverage_total.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", fig_path)
    return fig_path


# ===========================================================================
# Clean-up
# ===========================================================================

def cleanup_intermediates(paths):
    """Remove all MSs listed in *paths* that actually exist on disk."""
    section("Clean-up intermediate files")
    for path in paths:
        if path and os.path.exists(path):
            log.info("  Removing: %s", path)
            shutil.rmtree(path, ignore_errors=True)
        # Also remove any .listobs companion files
        lb = path + ".listobs" if path else None
        if lb and os.path.exists(lb):
            os.remove(lb)


# ===========================================================================
# Argument parsing + YAML config
# ===========================================================================

def _load_yaml_config(path):
    if not _YAML_AVAILABLE:
        log.warning("PyYAML is not installed; cannot load config file.")
        return {}
    try:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("Could not parse config file %s: %s", path, exc)
        return {}


def parse_args(argv=None):
    # ── Pre-parse: discover --config flag ──────────────────────────────────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(argv)

    cfg_path = pre_args.config or DEFAULT_CONFIG_NAME
    cfg      = _load_yaml_config(cfg_path)
    if cfg:
        log.info("Loaded config: %s", cfg_path)

    # ── Main parser ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        prog            = "concat_vis.py",
        description     = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None,
                        help="Path to YAML config file.")

    # Input / output
    io = parser.add_argument_group("Input / output")
    io.add_argument("--vis", nargs="+", metavar="MS",
                    help="Input measurement sets (≥2, space-separated).")
    io.add_argument("--ref-vis", default=None, metavar="MS",
                    help=(
                        "Reference MS for phase shifting and frequency matching.  "
                        "Must be one of the --vis inputs.  "
                        "Default: first MS in --vis."
                    ))
    io.add_argument("--source-name", default=None, metavar="NAME",
                    help=(
                        "Source name used in the output MS filename.  "
                        "Default: read the FIELD name from the reference MS "
                        "(first MS if --ref-vis is not set)."
                    ))
    io.add_argument("--temp-dir", default=None, metavar="DIR",
                    help=(
                        "Directory for intermediate MSs (split, chanavg, pshift, "
                        "freqmatch).  Default: parent directory of the first MS."
                    ))
    io.add_argument("--output-dir", default=None, metavar="DIR",
                    help=(
                        "Directory for final products (concat MS, wtspectrum MS, "
                        "UV plot, listobs files).  Default: parent directory of "
                        "the first MS."
                    ))
    io.add_argument("--force-overwrite", action="store_true", default=False,
                    help="Remove and reprocess existing output MSs.")
    io.add_argument("--force-concat", action="store_true", default=False,
                    help=(
                        "Skip the band-overlap frequency check and force "
                        "concatenation of all input MSs.  Use this when MSs "
                        "cover different sub-bands of the same receiver band "
                        "(e.g. C-low 4-6 GHz and C-high 6-8 GHz)."
                    ))
    io.add_argument("--keep-intermediates", action="store_true", default=False,
                    help="Do NOT delete intermediate MSs after the run.")
    io.add_argument("--no-repair-pointing", dest="repair_pointing",
                    action="store_false", default=True,
                    help=(
                        "Do not repair damaged POINTING subtables before "
                        "concat.  By default an MS whose POINTING header "
                        "claims rows it cannot read (the scar of an earlier "
                        "concat) has that subtable replaced by an empty one "
                        "of identical layout, which is required for concat to "
                        "succeed.  The damaged table is kept under "
                        "<output-dir>/damaged_subtables/."
                    ))

    # Mode
    mode_grp = parser.add_argument_group("Pipeline mode")
    mode_grp.add_argument(
        "--mode", choices=["same", "cross"], default="same",
        help=(
            "same  : same instrument, increase bandwidth; band check enforced.\n"
            "cross : different instruments; requires --do-freq-match.\n"
            "Default: same."
        ),
    )

    # Stage switches
    stages = parser.add_argument_group("Stage switches")
    stages.add_argument("--do-inspect",    action="store_true", default=False,
                        help="Stage 0: SPW tables + UV stats.")
    stages.add_argument("--do-split",      action="store_true", default=False,
                        help="Stage 1: fix SPW IDs, time-average.")
    stages.add_argument("--do-chanavg",    action="store_true", default=False,
                        help="Stage 2: channel-average, create WEIGHT_SPECTRUM.")
    stages.add_argument("--do-phaseshift", action="store_true", default=False,
                        help="Stage 3: shift to common phase centre.")
    stages.add_argument("--do-freq-match", action="store_true", default=False,
                        help="Stage 4: match frequency coverage to reference MS.")
    stages.add_argument("--do-statwt",     action="store_true", default=False,
                        help="Stage 5: compute statwt scale factors.")
    stages.add_argument("--do-concat",     action="store_true", default=False,
                        help="Stage 6: concatenate all prepared MSs.")
    stages.add_argument("--do-wtspectrum", action="store_true", default=False,
                        help="Stage 7: embed WEIGHT_SPECTRUM in concat MS.")
    stages.add_argument("--plot-uv",       action="store_true", default=False,
                        help="Save UV coverage comparison PNG.")

    # Transform parameters
    xform = parser.add_argument_group("Transform parameters")
    xform.add_argument("--timebin",     default="6s",
                       help="Time averaging bin for Stage 1 (default: 6s).")
    xform.add_argument("--correlation", default="RR,LL",
                       help="Correlation selection for Stage 1 (default: RR,LL).")
    xform.add_argument("--datacolumn",  default="data",
                       choices=["data", "corrected", "all"],
                       help="Data column (default: data).")
    xform.add_argument("--chan-out", type=int, default=64, metavar="N",
                       help="Target channels per SPW for Stage 2 (default: 64).")
    xform.add_argument("--ref-phasecentre", default=None, metavar="J2000",
                       help=(
                           "Override reference phase centre for Stage 3.  "
                           "Format: 'J2000 HH:MM:SS.s +DD.MM.SS.s'.  "
                           "Default: read from --ref-vis."
                       ))
    xform.add_argument("--freq-padding-mhz", type=float, default=500.0,
                       metavar="FLOAT",
                       help="Freq padding (MHz) per SPW interval for Stage 4 "
                            "(default: 500.0).")
    xform.add_argument("--statwt-preview", dest="statwt_preview",
                       action="store_true", default=True,
                       help=(
                           "Run statwt in preview mode: compute weights but "
                           "do NOT write them back to the MS (default: True). "
                           "Scale factors are still computed and applied by "
                           "concat via visweightscale.  Use "
                           "--no-statwt-preview only if you explicitly want "
                           "to rewrite in-place weights."
                       ))
    xform.add_argument("--no-statwt-preview", dest="statwt_preview",
                       action="store_false",
                       help="Write statwt weights back in-place.")
    xform.add_argument("--dirtol",  default="1000.0arcsec",
                       help="CASA concat direction tolerance (default: 1000.0arcsec).")
    xform.add_argument("--freqtol", default="1MHz",
                       help="CASA concat frequency tolerance (default: 1MHz).")
    xform.add_argument("--band-guard-factor", type=float, default=0.5,
                       metavar="FLOAT",
                       help="Min fractional overlap for same-instrument band check "
                            "(default: 0.5).")

    # ── Inject YAML values as defaults ─────────────────────────────────────
    if cfg:
        bool_flags = {
            "do_inspect", "do_split", "do_chanavg", "do_phaseshift",
            "do_freq_match", "do_statwt", "do_concat", "do_wtspectrum",
            "plot_uv", "force_overwrite", "force_concat", "keep_intermediates",
            "statwt_preview",
        }
        clean_cfg = {
            k: (True if k in bool_flags and v is True else v)
            for k, v in cfg.items()
            if not (k in bool_flags and not v)
        }
        parser.set_defaults(**clean_cfg)

    args = parser.parse_args(argv)

    # ── Validation ──────────────────────────────────────────────────────────
    if args.vis is None:
        parser.error("--vis is required.")
    if len(args.vis) < 2:
        parser.error("--vis requires at least 2 measurement sets.")

    if not any([
        args.do_inspect, args.do_split, args.do_chanavg, args.do_phaseshift,
        args.do_freq_match, args.do_statwt, args.do_concat, args.do_wtspectrum,
    ]):
        parser.error(
            "No processing stage selected.  Pass at least one of: "
            "--do-inspect --do-split --do-chanavg --do-phaseshift "
            "--do-freq-match --do-statwt --do-concat --do-wtspectrum"
        )

    if args.ref_vis and args.ref_vis not in args.vis:
        parser.error(
            "--ref-vis '{}' is not in the --vis list.  "
            "The reference must be one of the input MSs.".format(args.ref_vis)
        )

    if args.mode == "cross" and args.do_concat and not args.do_freq_match:
        log.warning(
            "--mode cross selected but --do-freq-match is NOT set.  "
            "Frequency coverage will not be matched.  "
            "Add --do-freq-match if that is intended."
        )

    return args


# ===========================================================================
# Main
# ===========================================================================

def main(argv=None):
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # Resolve directories
    # ------------------------------------------------------------------
    first_ms_parent = _parent_of_first_vis(args.vis)
    temp_dir   = args.temp_dir   or first_ms_parent
    output_dir = args.output_dir or first_ms_parent
    os.makedirs(temp_dir,   exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    separator()
    log.info("  concat_vis.py  -  MS concatenation pipeline")
    separator()
    log.info("Mode               : %s", args.mode)
    log.info("Source name (arg)  : %s", args.source_name or "(auto from FIELD)")
    log.info("Temp directory     : %s", temp_dir)
    log.info("Output directory   : %s", output_dir)
    log.info("Input MSs (%d):", len(args.vis))
    for v in args.vis:
        log.info("  %s", v)
    log.info("Stage switches:")
    log.info("  0 inspect    : %s", args.do_inspect)
    log.info("  1 split      : %s", args.do_split)
    log.info("  2 chanavg    : %s", args.do_chanavg)
    log.info("  3 phaseshift : %s", args.do_phaseshift)
    log.info("  4 freq-match : %s", args.do_freq_match)
    log.info("  5 statwt     : %s", args.do_statwt)
    log.info("  6 concat     : %s", args.do_concat)
    log.info("  7 wtspectrum : %s", args.do_wtspectrum)
    log.info("  plot-uv      : %s", args.plot_uv)
    separator()

    # ------------------------------------------------------------------
    # Validate input files
    # ------------------------------------------------------------------
    for v in args.vis:
        if not os.path.exists(v):
            log.error("Input MS not found: %s", v)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Determine reference index (used to track the ref through stages)
    # ------------------------------------------------------------------
    if args.ref_vis:
        ref_idx = args.vis.index(args.ref_vis)
    else:
        ref_idx = 0
    log.info("Reference MS index : %d  (%s)",
             ref_idx, os.path.basename(args.vis[ref_idx]))
    separator()

    # ------------------------------------------------------------------
    # Resolve source name (read FIELD table if not supplied)
    # ------------------------------------------------------------------
    source_name = args.source_name
    if not source_name:
        ref_ms_for_name = args.vis[ref_idx]
        source_name = get_field_name(ref_ms_for_name)
        if source_name:
            log.info("Source name        : %s  (from FIELD table of %s)",
                     source_name, os.path.basename(ref_ms_for_name))
        else:
            source_name = "SOURCE"
            log.warning("Could not read FIELD name from %s; "
                        "using fallback name '%s'.",
                        os.path.basename(ref_ms_for_name), source_name)
    else:
        log.info("Source name        : %s  (user-supplied)", source_name)

    # ------------------------------------------------------------------
    # Band-compatibility guard (same-instrument mode)
    # ------------------------------------------------------------------
    if args.mode == "same":
        if args.force_concat:
            log.warning(
                "--force-concat: skipping band-overlap frequency check.  "
                "MSs from different sub-bands (e.g. C-low / C-high) will be "
                "concatenated without a frequency compatibility guard."
            )
        else:
            check_band_overlap(args.vis, guard_factor=args.band_guard_factor)

    # ------------------------------------------------------------------
    # active_vis: updated after every stage that runs
    # ------------------------------------------------------------------
    active_vis   = list(args.vis)
    intermediates = []   # paths to remove at the end

    # ------------------------------------------------------------------
    # Stage 0: Inspect
    # ------------------------------------------------------------------
    if args.do_inspect:
        run_inspect(active_vis)
    else:
        log.info("Skipping Stage 0 (inspect).")

    # ------------------------------------------------------------------
    # Stage 1: mstransform
    # ------------------------------------------------------------------
    if args.do_split:
        active_vis = run_mstransform(
            active_vis,
            timebin        = args.timebin,
            correlation    = args.correlation,
            datacolumn     = args.datacolumn,
            force_overwrite= args.force_overwrite,
            temp_dir       = temp_dir,
        )
        intermediates.extend(active_vis)
    else:
        log.info("Skipping Stage 1 (mstransform).")

    # ------------------------------------------------------------------
    # Stage 2: Channel average
    # ------------------------------------------------------------------
    if args.do_chanavg:
        active_vis = run_chanavg(
            active_vis,
            chan_out        = args.chan_out,
            force_overwrite = args.force_overwrite,
            temp_dir        = temp_dir,
        )
        intermediates.extend(active_vis)
    else:
        log.info("Skipping Stage 2 (chanavg).")

    # ------------------------------------------------------------------
    # Stage 3: Phase shift
    # ------------------------------------------------------------------
    if args.do_phaseshift:
        prev       = list(active_vis)
        active_vis = run_phaseshift(
            active_vis,
            ref_idx         = ref_idx,
            ref_phasecentre = args.ref_phasecentre,
            force_overwrite = args.force_overwrite,
            temp_dir        = temp_dir,
        )
        for v_old, v_new in zip(prev, active_vis):
            if v_new != v_old:
                intermediates.append(v_new)
    else:
        log.info("Skipping Stage 3 (phaseshift).")

    # ------------------------------------------------------------------
    # Stage 4: Frequency matching
    # ------------------------------------------------------------------
    if args.do_freq_match:
        if args.mode == "same":
            log.warning(
                "--do-freq-match with --mode same is unusual "
                "(same-instrument mode normally does not need freq matching)."
            )
        prev       = list(active_vis)
        active_vis = run_freq_match(
            active_vis,
            ref_idx          = ref_idx,
            freq_padding_mhz = args.freq_padding_mhz,
            datacolumn       = args.datacolumn,
            force_overwrite  = args.force_overwrite,
            temp_dir         = temp_dir,
        )
        for v_old, v_new in zip(prev, active_vis):
            if v_new != v_old:
                intermediates.append(v_new)
    else:
        log.info("Skipping Stage 4 (freq-match).")

    # ------------------------------------------------------------------
    # Stage 5: Statistical weights
    # ------------------------------------------------------------------
    scale_factors = None
    if args.do_statwt:
        scale_factors, _ = run_statwt(
            active_vis,
            preview         = args.statwt_preview,
            force_overwrite = args.force_overwrite,
        )
    else:
        log.info("Skipping Stage 5 (statwt).")

    # ------------------------------------------------------------------
    # Stage 6: Concatenate
    # ------------------------------------------------------------------
    concat_ms = None
    if args.do_concat:
        concat_ms = run_concat(
            active_vis,
            source_name     = source_name,
            output_dir      = output_dir,
            ref_vis         = args.vis[ref_idx],
            scale_factors   = scale_factors,
            dirtol          = args.dirtol,
            freqtol         = args.freqtol,
            force_overwrite = args.force_overwrite,
            repair_pointing = args.repair_pointing,
        )
    else:
        log.info("Skipping Stage 6 (concat).")

    # ------------------------------------------------------------------
    # Stage 7: Embed WEIGHT_SPECTRUM in concat MS
    # ------------------------------------------------------------------
    if args.do_wtspectrum:
        if concat_ms is None:
            log.warning(
                "--do-wtspectrum requested but --do-concat was not run; skipping."
            )
        else:
            run_wtspectrum(concat_ms,
                           output_dir      = output_dir,
                           force_overwrite = args.force_overwrite)
    else:
        log.info("Skipping Stage 7 (wtspectrum).")

    # ------------------------------------------------------------------
    # UV coverage comparison plot
    # ------------------------------------------------------------------
    if args.plot_uv:
        plot_uv_comparison(
            active_vis,         # final processed MSs (fixed metadata, correct SPWs)
            concat_ms  = concat_ms,
            output_dir = output_dir,
        )

    # ------------------------------------------------------------------
    # Clean up intermediates
    # ------------------------------------------------------------------
    if not args.keep_intermediates:
        # Never remove: original inputs, the final concat MS,
        # the ref MS if it passed through unchanged
        protected = set(os.path.abspath(v) for v in args.vis)
        if concat_ms:
            protected.add(os.path.abspath(concat_ms))

        seen       = set()
        to_remove  = []
        for p in intermediates:
            ap = os.path.abspath(p)
            if ap not in protected and ap not in seen:
                seen.add(ap)
                to_remove.append(p)

        cleanup_intermediates(to_remove)
    else:
        log.info("--keep-intermediates: all intermediate MSs retained.")

    separator()
    log.info("  concat_vis.py  -  complete.")
    if concat_ms:
        log.info("  Final concat MS : %s", concat_ms)
    separator()


if __name__ == "__main__":
    main()
