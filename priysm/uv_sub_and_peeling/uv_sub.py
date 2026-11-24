#!/usr/bin/env python3
"""
uv_sub.py
=========
Automated subtraction of outlier sources from interferometric visibilities.

The field is imaged at low sampling, outlier sources are detected with
photutils, grouped into small sub-images, and each group is imaged with
WSClean (through morphen's ``mlibs.run_wsclean`` with ``savemodel=True``) and
subtracted from the visibilities with CASA ``uvsub``.  The science target and
any other position the user wants to keep are protected by an exclusion
radius.

Stages (each one optional)
--------------------------
  1. --do-detect       Wide-field detection image + source detection.
  2. (always)          Exclusion of protected positions and grouping.
  3. --do-subtract     Per-group imaging and uv-subtraction.
  4. --do-final-image  Wide-field image of the result and before/after QA.
     --do-target-image Small image of the target before and after, to see
                       how much the outliers affect a small-field image.

Subtraction methods (--method)
------------------------------
  phaseshift             phaseshift to each group, image it at the phase
                         centre, uvsub, and carry the corrected column to the
                         next group.  A last phaseshift restores the original
                         phase centre.
  shift                  One working copy of the MS; each group is imaged with
                         WSClean ``-shift`` and uvsub accumulates the
                         subtraction in CORRECTED_DATA.
  model       (default)  One deep wide-field clean of the whole field, target
                         included, restricted to a mask of the detected
                         sources.  The model inside the protected regions is
                         set to zero, the rest is predicted to MODEL_DATA
                         (wsclean -predict) and removed with a single uvsub.
                         Use it when the target is bright: it is cleaned
                         together with the outliers, so its sidelobes do not
                         contaminate their models.

With phaseshift and shift the model of every sub-image is checked: a warning
is logged when it holds much more flux than the catalogued sources of the
group, or a large negative flux (see --sub-model-flux-warn and
--sub-model-neg-warn).  Both are signs that the sidelobes of a bright source
outside the sub-image were cleaned.

Usage
-----
    python uv_sub.py --config uv_sub_cfg.yaml
    python uv_sub.py --vis /data/target.ms --mlibs-path /path/to/morphen/morphen \
        --do-detect --dry-run
    python uv_sub.py --vis /data/target.ms --mlibs-path /path/to/morphen/morphen \
        --do-detect --do-subtract --do-final-image

The input MS is never modified.  The result is ``<output_dir>/<stem>_uvsub.ms``.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import csv
import datetime
import glob
import logging
import math
import os
import re
import shutil
import sys
import time

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

# ---------------------------------------------------------------------------
# CASA (modular installation assumed)
# ---------------------------------------------------------------------------
try:
    # Load Python's own SQLite before CASA: casatools ships an older
    # libsqlite3 and, once it is loaded, 'import mlibs' fails with
    # "undefined symbol: sqlite3_deserialize".
    import sqlite3  # noqa: F401
except ImportError:
    pass

try:
    import casatools
    import casatasks
    msmd = casatools.msmetadata()
    tb   = casatools.table()
except ImportError as _casa_err:
    print(
        "[ERROR] Could not import casatools / casatasks.\n"
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
log = logging.getLogger("uv_sub")

LIGHTSPEED = 299792458.0  # m/s
RAD2ASEC = 180.0 / np.pi * 3600.0


def separator(char="=", width=72):
    log.info(char * width)


def section(title, char="-", width=72):
    separator(char, width)
    log.info("  %s", title)
    separator(char, width)


def _attach_log_file(output_dir):
    """Add a FileHandler to the logger, writing into *output_dir*."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, "uv_sub_{}.log".format(ts))
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(fh)
    log.info("  Log file: %s", log_path)
    return log_path


# ===========================================================================
# Helpers: coordinates
# ===========================================================================

def parse_coord(text):
    """
    Parse a sky position into an astropy SkyCoord.

    Accepted forms (an optional leading frame word such as J2000 is ignored):
        'J2000 09:14:57.207 +44.33.09.462'   CASA style (dots in declination)
        '09:14:57.207 +44:33:09.462'
        '09h14m57.207s +44d33m09.462s'
        '138.7384 44.5526'                   decimal degrees
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    tokens = str(text).replace(",", " ").split()
    if tokens and re.match(r"^[A-Za-z]", tokens[0]) and len(tokens) == 3:
        tokens = tokens[1:]
    if len(tokens) != 2:
        raise ValueError("Cannot parse coordinate: {!r}".format(text))
    ra_s, dec_s = tokens

    # CASA writes declination as dd.mm.ss.sss
    if re.match(r"^[+-]?\d+\.\d+\.\d+(\.\d+)?$", dec_s):
        dec_s = dec_s.replace(".", ":", 2)

    try:
        ra_deg, dec_deg = float(ra_s), float(dec_s)
        return SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="fk5")
    except ValueError:
        return SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg), frame="fk5")


def coord_to_casa(coord, frame="J2000"):
    """SkyCoord -> 'J2000 hh:mm:ss.ssss +dd.mm.ss.sss' (CASA phasecenter)."""
    ra, dec = coord.to_string("hmsdms", precision=4).split()
    ra = ra.replace("h", ":").replace("m", ":").replace("s", "")
    dec = dec.replace("d", ".").replace("m", ".").replace("s", "")
    return "{} {} {}".format(frame, ra, dec)


def coord_to_wsclean(coord):
    """SkyCoord -> '09h14m57.2070s +44d33m09.4620s' (WSClean -shift)."""
    return coord.to_string("hmsdms", precision=4)


def get_phase_centre(vis):
    """Return (SkyCoord, frame_name) of the phase centre of field 0."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    msmd.open(vis)
    try:
        pc = msmd.phasecenter()
    finally:
        msmd.close()
    coord = SkyCoord(pc["m0"]["value"] * u.rad, pc["m1"]["value"] * u.rad,
                     frame="fk5")
    return coord, pc.get("refer", "J2000")


def offsets_arcsec(centre, coord):
    """Tangent-plane offsets (dlon, dlat) of *coord* from *centre*, arcsec."""
    dlon, dlat = centre.spherical_offsets_to(coord)
    return float(dlon.arcsec), float(dlat.arcsec)


def offsets_to_coord(centre, dx, dy):
    """Inverse of offsets_arcsec()."""
    import astropy.units as u
    return centre.spherical_offsets_by(dx * u.arcsec, dy * u.arcsec)


# ===========================================================================
# Helpers: MS metadata and image geometry
# ===========================================================================

def get_mean_frequency(vis):
    """Mean channel frequency over all SPWs [Hz]."""
    msmd.open(vis)
    try:
        means = [float(np.mean(msmd.chanfreqs(s))) for s in range(msmd.nspw())]
    finally:
        msmd.close()
    return float(np.mean(means))


def get_max_baseline(vis):
    """Longest unflagged cross-correlation baseline [m]."""
    tb.open(vis)
    try:
        uvw      = tb.getcol("UVW")
        ant1     = tb.getcol("ANTENNA1")
        ant2     = tb.getcol("ANTENNA2")
        flag_row = tb.getcol("FLAG_ROW")
    finally:
        tb.close()
    good = (ant1 != ant2) & (~flag_row.astype(bool))
    return float(np.max(np.hypot(uvw[0, good], uvw[1, good])))


def get_dish_diameter(vis, default=25.0):
    """Median antenna diameter from the ANTENNA table [m]."""
    try:
        tb.open(os.path.join(vis, "ANTENNA"))
        try:
            diam = np.asarray(tb.getcol("DISH_DIAMETER"), dtype=float)
        finally:
            tb.close()
        diam = diam[diam > 0]
        if diam.size:
            return float(np.median(diam))
    except Exception as exc:
        log.warning("  Could not read DISH_DIAMETER (%s)", exc)
    log.warning("  Using default dish diameter of %.1f m", default)
    return default


def has_column(vis, column):
    tb.open(vis)
    try:
        return column in tb.colnames()
    finally:
        tb.close()


def get_ms_geometry(vis):
    """
    Read the quantities that set the image geometry of *vis*.

    Returns a dict with: freq [Hz], b_max [m], dish [m],
    beam [arcsec] (lambda / B_max) and pb_fwhm [arcsec] (1.02 lambda / D).
    """
    freq  = get_mean_frequency(vis)
    b_max = get_max_baseline(vis)
    dish  = get_dish_diameter(vis)
    lam   = LIGHTSPEED / freq
    geom = {
        "freq": freq, "b_max": b_max, "dish": dish,
        "beam": lam / b_max * RAD2ASEC,
        "pb_fwhm": 1.02 * lam / dish * RAD2ASEC,
    }
    log.info("  nu_mean=%.3f GHz  Bmax=%.1f m  dish=%.1f m  "
             "beam~%.3f arcsec  PB FWHM~%.2f arcmin",
             freq / 1e9, b_max, dish, geom["beam"], geom["pb_fwhm"] / 60.0)
    return geom


def _even(n):
    n = int(math.ceil(n))
    return n + (n % 2)


def plan_detection_image(geom, pix_per_beam=3.0, max_imsize=10240,
                         fov_pb_factor=2.0, fov_arcmin=None):
    """
    Cell size, image size and uv-taper of the wide-field detection image.

    The image covers ``fov_pb_factor`` x PB FWHM (or ``fov_arcmin``) with
    ``pix_per_beam`` pixels per beam.  When that needs more than
    ``max_imsize`` pixels, the cell is coarsened to cover the same field and a
    Gaussian uv-taper of ``pix_per_beam`` cells keeps the beam sampled.
    """
    fov = fov_arcmin * 60.0 if fov_arcmin else fov_pb_factor * geom["pb_fwhm"]
    cell = geom["beam"] / pix_per_beam
    imsize = _even(fov / cell)
    taper = ""
    if imsize > max_imsize:
        imsize = int(max_imsize)
        cell = fov / imsize
        taper = "{:.4f}asec".format(pix_per_beam * cell)
        log.info("  Field needs more than %d px at beam/%.0f: coarsening the "
                 "cell and tapering to %s", max_imsize, pix_per_beam, taper)
    plan = {"cell": cell, "imsize": imsize, "taper": taper, "fov": fov}
    log.info("  Detection image: FOV=%.2f arcmin  cell=%.4f arcsec  "
             "imsize=%d  taper=%s", fov / 60.0, cell, imsize, taper or "none")
    return plan


# ===========================================================================
# Helpers: mlibs / morphen import and WSClean wrapper
# ===========================================================================

def import_mlibs(mlibs_path=None):
    """Import mlibs from the morphen package (see --mlibs-path)."""
    if mlibs_path and mlibs_path not in sys.path:
        sys.path.insert(0, mlibs_path)
    try:
        import mlibs as _mlibs
    except ImportError as exc:
        log.error("  Could not import mlibs: %s\n"
                  "  Specify the path to morphen/morphen/ with --mlibs-path.",
                  exc)
        return None
    log.info("  mlibs imported from: %s",
             getattr(_mlibs, "__file__", "unknown location"))
    return _mlibs


def _image_products(image):
    return {
        "image": image,
        "residual": image.replace("MFS-image.fits", "MFS-residual.fits"),
        "model": image.replace("MFS-image.fits", "MFS-model.fits"),
    }


def run_imaging(mlibs, vis, outdir, base_name, imsize, cell, robust,
                nsigma_automask, nsigma_autothreshold, niter, nc,
                with_multiscale, savemodel=False, shift=None,
                datacolumn="DATA", taper="", opt_args="''",
                reuse=False, mask=None):
    """
    Image *vis* with WSClean through ``mlibs.run_wsclean``.

    Images are written to *outdir* and their names start with *base_name*,
    which must be unique per call: products of an earlier call with the same
    name are removed first, because morphen skips WSClean when the image
    already exists and MODEL_DATA would then be stale.

    *mask* is a FITS clean mask (WSClean -fits-mask) or None.

    With ``reuse=True`` an existing image with this name is returned instead
    (never use it together with ``savemodel``).

    Returns a dict with the paths of the MFS 'image', 'residual' and 'model'.
    Raises RuntimeError when WSClean produced no image.
    """
    os.makedirs(outdir, exist_ok=True)
    if reuse:
        found = glob.glob(os.path.join(outdir, base_name + "_*MFS-image.fits"))
        if found:
            image = max(found, key=os.path.getmtime)
            log.info("  Reusing existing image: %s  (use --force-overwrite "
                     "to remake it)", os.path.basename(image))
            return _image_products(image)
    for old in glob.glob(os.path.join(outdir, base_name + "_*")):
        if os.path.isdir(old):
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.remove(old)

    t0 = time.time()
    try:
        mlibs.run_wsclean(
            vis,
            robust=robust,
            imsize=int(imsize), imsizey=int(imsize),
            cell="{:.4f}asec".format(cell),
            opt_args=opt_args,
            savemodel=savemodel,
            shift="'{}'".format(shift) if shift else None,
            base_name=base_name,
            nsigma_automask=str(nsigma_automask),
            nsigma_autothreshold=str(nsigma_autothreshold),
            quiet=True,
            nc=int(nc),
            with_multiscale=bool(with_multiscale),
            scales="None",
            datacolumn=datacolumn,
            uvtaper=[taper],
            niter=int(niter),
            calculate_subband_fluxes=False,
            image_list={}, image_statistics={},
            outdir=outdir,
            mask=mask,
        )
    except Exception as exc:
        # run_wsclean also plots and measures the image after WSClean ends;
        # a failure there does not invalidate the image or the model.
        log.warning("  mlibs.run_wsclean post-processing failed: %s", exc)

    found = [f for f in glob.glob(
                 os.path.join(outdir, base_name + "_*MFS-image.fits"))
             if os.path.getmtime(f) >= t0 - 1.0]
    if not found:
        raise RuntimeError(
            "WSClean produced no image for '{}' (see output above)".format(
                base_name))
    image = max(found, key=os.path.getmtime)
    log.info("  Image: %s  (%.1f min)", os.path.basename(image),
             (time.time() - t0) / 60.0)
    return _image_products(image)


# ===========================================================================
# Stage 2: source detection
# ===========================================================================

def _load_image(fits_file):
    """Return (2D data, celestial WCS, header) of a WSClean image."""
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(fits_file) as hdul:
        data = np.squeeze(hdul[0].data).astype(float)
        header = hdul[0].header
    return data, WCS(header).celestial, header


def detect_sources(images, nsigma=20.0, extent_nsigma=5.0):
    """
    Detect sources in a WSClean image with photutils.

    Islands are segmented at ``extent_nsigma`` x rms so that their bounding
    box covers the whole source, and only islands whose peak is above
    ``nsigma`` x rms are kept.  The rms is the MAD standard deviation of the
    residual image.

    ``info`` holds the rms, the cell size, the beam and 'islands', a boolean
    map of the kept islands.

    Returns (sources, info); each source is a dict with the peak position
    (SkyCoord 'coord'), 'peak' [Jy/beam], 'snr', the bounding-box centre
    ('box_coord') and half sizes 'hx', 'hy' [arcsec], and 'flux' [Jy], the
    integrated flux of the island.
    """
    from astropy.stats import mad_std
    from astropy.wcs.utils import proj_plane_pixel_scales
    from photutils.segmentation import detect_sources as _phot_detect
    from scipy import ndimage

    data, wcs, header = _load_image(images["image"])
    residual = _load_image(images["residual"])[0]
    rms = float(mad_std(residual, ignore_nan=True))
    cell = float(np.mean(proj_plane_pixel_scales(wcs))) * 3600.0
    beam_pix = (np.pi * header.get("BMAJ", 0.0) * header.get("BMIN", 0.0)
                * 3600.0 ** 2 / (4.0 * np.log(2.0)) / cell ** 2)
    npixels = max(4, int(round(0.5 * beam_pix)))
    info = {"rms": rms, "cell": cell, "shape": data.shape,
            "bmaj": header.get("BMAJ", 0.0) * 3600.0}
    log.info("  rms(MAD, residual)=%.3e Jy/beam  beam=%.2f arcsec  "
             "threshold=%.1f sigma  island level=%.1f sigma  npixels=%d",
             rms, info["bmaj"], nsigma, extent_nsigma, npixels)

    clean = np.nan_to_num(data, nan=0.0)
    seg = _phot_detect(clean, min(extent_nsigma, nsigma) * rms,
                       npixels=npixels)
    info["islands"] = np.zeros(data.shape, dtype=bool)
    if seg is None:
        return [], info

    labels = np.arange(1, seg.nlabels + 1)
    peaks = ndimage.maximum(clean, seg.data, labels)
    # Integrated flux of each island [Jy]
    fluxes = (ndimage.sum(clean, seg.data, labels) / beam_pix
              if beam_pix > 0 else np.full(len(labels), np.nan))
    positions = ndimage.maximum_position(clean, seg.data, labels)
    slices = ndimage.find_objects(seg.data)

    sources = []
    for peak, flux, (py, px), slc in zip(peaks, fluxes, positions, slices):
        if peak < nsigma * rms:
            continue
        ys, xs = slc
        box_x = 0.5 * (xs.start + xs.stop - 1)
        box_y = 0.5 * (ys.start + ys.stop - 1)
        sources.append({
            "coord": wcs.pixel_to_world(px, py).fk5,
            "box_coord": wcs.pixel_to_world(box_x, box_y).fk5,
            "hx": 0.5 * (xs.stop - xs.start) * cell,
            "hy": 0.5 * (ys.stop - ys.start) * cell,
            "peak": float(peak),
            "flux": float(flux),
            "snr": float(peak / rms),
            "origin": "detected",
        })
    sources.sort(key=lambda s: -s["peak"])
    # Map of the kept islands (protected ones included), used as clean mask
    # by the 'model' method.
    info["islands"] = np.isin(seg.data, labels[np.asarray(peaks)
                                               >= nsigma * rms])
    log.info("  %d island(s) above %.1f sigma", len(sources), nsigma)
    return sources, info


def manual_sources(coord_strings):
    """Turn user-given positions into point-like source entries."""
    sources = []
    for text in coord_strings or []:
        coord = parse_coord(text)
        sources.append({"coord": coord, "box_coord": coord, "hx": 0.0,
                        "hy": 0.0, "peak": float("nan"), "snr": float("nan"),
                        "flux": float("nan"),
                        "origin": "manual"})
    return sources


# ===========================================================================
# Stage 2: exclusion and grouping
# ===========================================================================

def apply_exclusion(sources, protected, radius):
    """
    Drop the sources that lie within *radius* arcsec of a protected position.
    Sets source['status'] and returns the list of sources to subtract.
    """
    kept = []
    for src in sources:
        seps = [src["coord"].separation(p).arcsec for p in protected]
        if seps and min(seps) <= radius:
            src["status"] = "protected"
            log.info("  Protected (%.1f arcsec from a kept position): %s",
                     min(seps), coord_to_casa(src["coord"]))
        else:
            src["status"] = "subtract"
            kept.append(src)
    return kept


def group_sources(sources, centre, protected, radius, cell, max_subimage=1024,
                  min_subimage=256, pad=10.0):
    """
    Group nearby sources into square sub-images.

    Works in the tangent plane around *centre* (arcsec).  The brightest
    unassigned source seeds a group, which absorbs unassigned neighbours as
    long as the padded bounding box fits in ``max_subimage`` pixels of size
    *cell* and stays clear of every protected circle.  *pad* is the margin
    added on each side [arcsec].

    Returns a list of groups (dicts with 'coord', 'imsize', 'half' [arcsec]
    and 'members'), brightest first.  Sources that cannot be imaged without
    touching a protected circle get status 'skipped'.
    """
    for src in sources:
        src["x"], src["y"] = offsets_arcsec(centre, src["box_coord"])
    prot_xy = [offsets_arcsec(centre, p) for p in protected]
    max_half = 0.5 * max_subimage * cell
    min_half = 0.5 * min_subimage * cell
    min_pad = min(pad, 3.0 * cell)

    def box_of(members):
        x0 = min(m["x"] - m["hx"] for m in members)
        x1 = max(m["x"] + m["hx"] for m in members)
        y0 = min(m["y"] - m["hy"] for m in members)
        y1 = max(m["y"] + m["hy"] for m in members)
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * max(x1 - x0, y1 - y0)

    def fit(members):
        """Return (cx, cy, half) of a valid sub-image, or None."""
        cx, cy, extent = box_of(members)
        # A square of half size h centred on (cx, cy) is clear of a circle
        # of radius r at (px, py) when max(|dx|, |dy|) - h > r.
        allowed = max_half
        for px, py in prot_xy:
            allowed = min(allowed,
                          max(abs(px - cx), abs(py - cy)) - radius)
        if extent + min_pad <= allowed:
            return cx, cy, min(max(extent + pad, min_half), allowed)
        if len(members) == 1 and allowed >= max_half:
            # A single source larger than the maximum sub-image.
            return cx, cy, max_half
        return None

    order = sorted(sources, key=lambda s: (-s["peak"]
                                           if np.isfinite(s["peak"]) else 0.0))
    unassigned = list(order)
    groups = []
    while unassigned:
        seed = unassigned.pop(0)
        if fit([seed]) is None:
            seed["status"] = "skipped"
            log.warning("  Skipped (sub-image would overlap a protected "
                        "position): %s", coord_to_casa(seed["coord"]))
            continue
        members = [seed]
        for cand in sorted(unassigned, key=lambda s: math.hypot(
                s["x"] - seed["x"], s["y"] - seed["y"])):
            if math.hypot(cand["x"] - seed["x"],
                          cand["y"] - seed["y"]) > 3.0 * max_half:
                break
            if fit(members + [cand]) is not None:
                members.append(cand)
        for m in members[1:]:
            unassigned.remove(m)
        cx, cy, half = fit(members)
        if box_of(members)[2] > half:
            log.warning("  Source larger than the maximum sub-image, it will "
                        "be only partly modelled: %s",
                        coord_to_casa(seed["coord"]))
        imsize = min(_even(2.0 * half / cell), int(max_subimage))
        groups.append({"coord": offsets_to_coord(centre, cx, cy),
                       "x": cx, "y": cy, "half": half, "imsize": imsize,
                       "members": members})

    for k, grp in enumerate(groups, start=1):
        grp["id"] = k
        for m in grp["members"]:
            m["group"] = k
        log.info("  Group %02d: %s  imsize=%d  (%d source%s, brightest "
                 "%.2f mJy/beam)", k, coord_to_casa(grp["coord"]),
                 grp["imsize"], len(grp["members"]),
                 "" if len(grp["members"]) == 1 else "s",
                 grp["members"][0]["peak"] * 1e3)
    return groups


# ===========================================================================
# Stage 2: products (catalogue, regions, overlay)
# ===========================================================================

def write_catalogue(path, sources):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "ra_deg", "dec_deg", "position",
                         "peak_jy_beam", "snr", "half_x_arcsec",
                         "half_y_arcsec", "origin", "status", "group",
                         "flux_jy"])
        for i, s in enumerate(sources, start=1):
            writer.writerow([
                i, "{:.7f}".format(s["coord"].ra.deg),
                "{:.7f}".format(s["coord"].dec.deg), coord_to_casa(s["coord"]),
                "{:.6e}".format(s["peak"]), "{:.1f}".format(s["snr"]),
                "{:.2f}".format(s["hx"]), "{:.2f}".format(s["hy"]),
                s["origin"], s.get("status", ""), s.get("group", ""),
                "{:.6e}".format(s.get("flux", float("nan")))])
    log.info("  Catalogue: %s", path)


def write_regions(path, sources, groups, protected, radius):
    """DS9 region file: sources, sub-image boxes and protected circles."""
    lines = ["# Region file format: DS9 version 4.1", "fk5"]
    colours = {"subtract": "red", "protected": "green", "skipped": "yellow"}
    for i, s in enumerate(sources, start=1):
        size = max(s["hx"], s["hy"], 2.0)
        lines.append('circle({:.7f},{:.7f},{:.2f}") # color={} text={{{}}}'
                     .format(s["coord"].ra.deg, s["coord"].dec.deg, size,
                             colours.get(s.get("status"), "red"), i))
    for g in groups:
        lines.append('box({:.7f},{:.7f},{:.2f}",{:.2f}",0) # color=cyan '
                     'text={{G{}}}'.format(g["coord"].ra.deg,
                                           g["coord"].dec.deg,
                                           2 * g["half"], 2 * g["half"],
                                           g["id"]))
    for p in protected:
        lines.append('circle({:.7f},{:.7f},{:.2f}") # color=green dash=1'
                     .format(p.ra.deg, p.dec.deg, radius))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("  Regions:   %s", path)


def _downsampled(data, max_size=2000):
    """Block-maximum downsampling, so point sources stay visible."""
    f = int(math.ceil(max(data.shape) / float(max_size)))
    if f <= 1:
        return data
    ny, nx = (data.shape[0] // f) * f, (data.shape[1] // f) * f
    with np.errstate(invalid="ignore"):
        return np.nanmax(
            data[:ny, :nx].reshape(ny // f, f, nx // f, f), axis=(1, 3))


def _show_field(ax, fits_file, rms, title):
    """Show a wide-field image in arcmin offsets from its centre."""
    from astropy.wcs.utils import proj_plane_pixel_scales
    data, wcs, _ = _load_image(fits_file)
    cell = float(np.mean(proj_plane_pixel_scales(wcs))) * 60.0  # arcmin
    half_x = 0.5 * data.shape[1] * cell
    half_y = 0.5 * data.shape[0] * cell
    ax.imshow(_downsampled(np.nan_to_num(data)), origin="lower",
              cmap="gray_r", vmin=-3 * rms, vmax=30 * rms,
              extent=[half_x, -half_x, -half_y, half_y],
              interpolation="nearest")
    ax.set_xlabel("RA offset [arcmin]")
    ax.set_ylabel("Dec offset [arcmin]")
    ax.set_title(title, fontsize=10)

    def to_offsets(coord):
        px, py = wcs.world_to_pixel(coord)
        return (-(float(px) - 0.5 * (data.shape[1] - 1)) * cell,
                (float(py) - 0.5 * (data.shape[0] - 1)) * cell)
    return to_offsets


def _draw_overlays(ax, to_offsets, sources, groups, protected, radius):
    from matplotlib.patches import Circle, Rectangle
    colours = {"subtract": "tab:red", "protected": "tab:green",
               "skipped": "gold"}
    for i, s in enumerate(sources, start=1):
        x, y = to_offsets(s["coord"])
        ax.plot(x, y, "o", mfc="none", ms=9, mew=1.0,
                color=colours.get(s.get("status"), "tab:red"))
        ax.annotate(str(i), (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color="tab:red")
    for g in groups:
        x, y = to_offsets(g["coord"])
        h = g["half"] / 60.0
        ax.add_patch(Rectangle((x - h, y - h), 2 * h, 2 * h, fill=False,
                               ec="tab:cyan", lw=0.8))
    for p in protected:
        x, y = to_offsets(p)
        ax.add_patch(Circle((x, y), max(radius / 60.0, 0.15), fill=False,
                            ec="tab:green", lw=1.2, ls="--"))


def plot_detections(path, images, rms, sources, groups, protected, radius):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    to_offsets = _show_field(ax, images["image"], rms,
                             "Detections (red), sub-images (cyan), "
                             "protected (green)")
    _draw_overlays(ax, to_offsets, sources, groups, protected, radius)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  Overlay:   %s", path)


# ===========================================================================
# Stage 3: subtraction
# ===========================================================================

def subtract_model(vis, mode="uvsub"):
    """
    Remove the model in MODEL_DATA of *vis* from its visibilities.  The
    result is left in CORRECTED_DATA.

    This is the single place where the subtraction method is chosen:

    uvsub  Plain subtraction of the model (casatasks.uvsub).
    peel   Not implemented.  A peeling step would, with the group at the
           phase centre and its model in MODEL_DATA, (1) solve gains towards
           it with gaincal, (2) subtract the model corrupted by those gains,
           and (3) leave the result in CORRECTED_DATA, exactly like uvsub
           does.  Nothing else in this module needs to change.
    """
    if mode == "uvsub":
        casatasks.uvsub(vis=vis)
    elif mode == "peel":
        raise NotImplementedError(
            "subtract_mode 'peel' is not implemented yet; use 'uvsub'.")
    else:
        raise ValueError("Unknown subtract_mode: {}".format(mode))


def _rm_ms(path):
    if path and os.path.exists(path):
        shutil.rmtree(path)
    if path and os.path.exists(path + ".flagversions"):
        shutil.rmtree(path + ".flagversions")


def check_group_model(products, group, args):
    """
    Compare the clean model of a sub-image with the catalogued sources of
    its group and warn when it looks contaminated.

    Two signs of a model that fitted the sidelobes of a bright source lying
    outside the sub-image (usually the target):
      - the model flux is more than ``sub_model_flux_warn`` times the
        integrated flux of the group's sources in the detection image;
      - the negative model flux is more than ``sub_model_neg_warn`` times
        the positive one.
    Sets group['model_flux'], group['expected_flux'] and group['suspect'];
    returns True when a warning was raised.  The group is still subtracted.
    """
    group["suspect"] = False
    model_file = products.get("model") if products else None
    if not model_file or not os.path.exists(model_file):
        return False
    try:
        model = _load_image(model_file)[0]
    except Exception as exc:
        log.warning("  Could not read %s (%s)", os.path.basename(model_file),
                    exc)
        return False
    pos = float(np.nansum(model[model > 0]))
    neg = float(-np.nansum(model[model < 0]))
    total = pos - neg
    expected = float(np.nansum([m.get("flux", np.nan)
                                for m in group["members"]]))
    known = any(np.isfinite(m.get("flux", np.nan)) for m in group["members"])
    group["model_flux"] = total
    group["expected_flux"] = expected if known else float("nan")
    log.info("  Model flux: %.3f mJy (positive %.3f, negative %.3f)%s",
             total * 1e3, pos * 1e3, -neg * 1e3,
             "  catalogue: {:.3f} mJy".format(expected * 1e3) if known
             else "")

    reasons = []
    if (known and expected > 0 and args.sub_model_flux_warn > 0
            and total > args.sub_model_flux_warn * expected):
        reasons.append("model flux is {:.1f}x the catalogued flux of its "
                       "source(s) ({:.3f} vs {:.3f} mJy)".format(
                           total / expected, total * 1e3, expected * 1e3))
    if (args.sub_model_neg_warn > 0 and neg > 0
            and neg > args.sub_model_neg_warn * max(pos, 1e-30)):
        if pos > 1e-3 * neg:
            reasons.append("negative model flux is {:.0f}% of the positive "
                           "one ({:.3f} vs {:.3f} mJy)".format(
                               100.0 * neg / pos, -neg * 1e3, pos * 1e3))
        else:
            reasons.append("the model is entirely negative "
                           "({:.3f} mJy)".format(-neg * 1e3))
    for reason in reasons:
        log.warning("  Group %02d: %s.", group["id"], reason)
    if reasons:
        log.warning("  Group %02d: the model probably contains sidelobes of "
                    "a bright source outside the sub-image; subtracting it "
                    "can remove flux from the target.", group["id"])
        group["suspect"] = True
    return group["suspect"]


def report_suspect_groups(groups):
    """Summary warning after a phaseshift / shift subtraction."""
    bad = [g for g in groups if g.get("suspect")]
    if not bad:
        return 0
    separator("!")
    log.warning("  WARNING: %d of %d sub-image model(s) look contaminated: "
                "group(s) %s", len(bad), len(groups),
                ", ".join("{:02d}".format(g["id"]) for g in bad))
    log.warning("  They hold %.1f mJy in total; the catalogued sources of "
                "these groups add up to %.1f mJy.",
                sum(g.get("model_flux", 0.0) for g in bad) * 1e3,
                float(np.nansum([g.get("expected_flux", np.nan)
                                 for g in bad])) * 1e3)
    log.warning("  This happens when the target (or another bright source) "
                "is not cleaned in the sub-images.  Check the target "
                "before/after image and flux, and prefer --method model.")
    separator("!")
    return len(bad)


def _image_group(mlibs, vis, group, tag, args, cell, shift=None,
                 datacolumn="DATA"):
    """Image one group with savemodel=True; return True if a model exists."""
    try:
        products = run_imaging(
            mlibs, vis, args.image_dir, "{}_g{:03d}".format(tag, group["id"]),
            imsize=group["imsize"], cell=cell, robust=args.sub_robust,
            nsigma_automask=args.sub_nsigma_automask,
            nsigma_autothreshold=args.sub_nsigma_autothreshold,
            niter=args.sub_niter, nc=args.n_subimages,
            with_multiscale=args.sub_multiscale, savemodel=True,
            shift=shift, datacolumn=datacolumn, opt_args=args.opt_args)
    except RuntimeError as exc:
        log.error("  %s", exc)
        return False
    if not has_column(vis, "MODEL_DATA"):
        log.error("  WSClean did not write MODEL_DATA to %s",
                  os.path.basename(vis))
        return False
    check_group_model(products, group, args)
    return True


def subtract_phaseshift(mlibs, vis, out_vis, groups, tag, args, cell,
                        centre, frame):
    """
    Notebook recipe: phaseshift to each group, image, uvsub, and carry the
    corrected column to the next group.  A last phaseshift to *centre*
    writes *out_vis*.  Returns the number of groups subtracted.
    """
    stem = os.path.splitext(os.path.basename(vis))[0]
    src, src_col = vis, args.datacolumn
    done = 0
    for grp in groups:
        section("Group {:02d}/{:02d}  {}".format(
            grp["id"], len(groups), coord_to_casa(grp["coord"], frame)),
            char=".")
        tmp = os.path.join(args.work_dir,
                           "{}_{}_g{:03d}.ms".format(stem, tag, grp["id"]))
        _rm_ms(tmp)
        casatasks.phaseshift(vis=src, outputvis=tmp, datacolumn=src_col,
                             phasecenter=coord_to_casa(grp["coord"], frame))
        if src != vis:
            _rm_ms(src)
        src, src_col = tmp, "data"
        if _image_group(mlibs, tmp, grp, tag, args, cell):
            subtract_model(tmp, args.subtract_mode)
            src_col = "corrected"
            done += 1
        else:
            log.warning("  Group %02d was NOT subtracted.", grp["id"])

    section("Restoring the original phase centre", char=".")
    _rm_ms(out_vis)
    casatasks.phaseshift(vis=src, outputvis=out_vis, datacolumn=src_col,
                         phasecenter=coord_to_casa(centre, frame))
    if src != vis and not args.keep_tmp:
        _rm_ms(src)
    return done


def subtract_shift(mlibs, vis, out_vis, groups, tag, args, cell):
    """
    One working copy of the MS; each group is imaged with WSClean -shift and
    uvsub accumulates the subtraction in CORRECTED_DATA.  Returns the number
    of groups subtracted.
    """
    stem = os.path.splitext(os.path.basename(vis))[0]
    work = os.path.join(args.work_dir, "{}_{}_work.ms".format(stem, tag))
    _rm_ms(work)
    log.info("  Copying %s -> %s", os.path.basename(vis), work)
    shutil.copytree(vis, work)
    column = "CORRECTED_DATA" if args.datacolumn == "corrected" else "DATA"
    done = 0
    for grp in groups:
        section("Group {:02d}/{:02d}  {}".format(
            grp["id"], len(groups), coord_to_casa(grp["coord"])), char=".")
        if _image_group(mlibs, work, grp, tag, args, cell,
                        shift=coord_to_wsclean(grp["coord"]),
                        datacolumn=column):
            subtract_model(work, args.subtract_mode)
            column = "CORRECTED_DATA"
            done += 1
        else:
            log.warning("  Group %02d was NOT subtracted.", grp["id"])

    _rm_ms(out_vis)
    casatasks.split(vis=work, outputvis=out_vis,
                    datacolumn="corrected" if column == "CORRECTED_DATA"
                    else "data")
    if not args.keep_tmp:
        _rm_ms(work)
    return done


# ---------------------------------------------------------------------------
# Method 'model': wide-field model, edited and predicted
# ---------------------------------------------------------------------------

def _fill_disc(array, x, y, radius, value):
    """Set *array* to *value* within *radius* pixels of (x, y), in place."""
    ny, nx = array.shape
    x0, x1 = int(math.floor(x - radius)), int(math.ceil(x + radius)) + 1
    y0, y1 = int(math.floor(y - radius)), int(math.ceil(y + radius)) + 1
    x0, x1, y0, y1 = max(x0, 0), min(x1, nx), max(y0, 0), min(y1, ny)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    array[y0:y1, x0:x1][(xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2] = value


def build_clean_mask(path, image, info, protected, radius, manual,
                     dilate_beams=2.0):
    """
    Write the FITS clean mask of the 'model' method on the grid of *image*.

    The mask allows cleaning on the detected islands (the target included),
    grown by ``dilate_beams`` beams, on the protected circles of *radius*
    arcsec, and on a small circle around each *manual* position.
    """
    from astropy.io import fits
    from scipy import ndimage

    with fits.open(image) as hdul:
        header = hdul[0].header.copy()
        shape = hdul[0].data.shape
    wcs = _load_image(image)[1]
    cell = info["cell"]
    beam_pix = (info["bmaj"] / cell) if info["bmaj"] > 0 else 3.0

    mask = info["islands"].copy()
    grow = int(round(dilate_beams * beam_pix))
    if grow > 0 and mask.any():
        mask = ndimage.binary_dilation(mask, iterations=grow)
    for coord in protected:
        x, y = wcs.world_to_pixel(coord)
        _fill_disc(mask, float(x), float(y), radius / cell, True)
    for src in manual:
        x, y = wcs.world_to_pixel(src["coord"])
        _fill_disc(mask, float(x), float(y), 3.0 * beam_pix, True)

    fits.writeto(path, mask.astype(np.float32).reshape(shape), header,
                 overwrite=True)
    log.info("  Clean mask: %s  (%.3f%% of the image)", path,
             100.0 * mask.mean())
    return path


def edit_model_images(prefix, new_prefix, protected, radius):
    """
    Copy the WSClean model images ``<prefix>-NNNN-model.fits`` to
    ``<new_prefix>-NNNN-model.fits`` with every pixel within *radius* arcsec
    of a protected position set to zero.  The originals are kept.

    Returns (n_images, flux_kept, flux_subtracted): the model flux [Jy,
    mean over the sub-bands] left in the data and the flux to be subtracted.
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales

    for old in glob.glob(new_prefix + "-*"):
        os.remove(old)
    models = sorted(glob.glob(prefix + "-[0-9][0-9][0-9][0-9]-model.fits"))
    if not models:
        raise RuntimeError(
            "No model images found for '{}'".format(os.path.basename(prefix)))
    kept, removed = [], []
    for model in models:
        with fits.open(model) as hdul:
            header = hdul[0].header
            data = hdul[0].data.copy()
            plane = np.squeeze(data)          # view on the same pixels
            wcs = WCS(header).celestial
            cell = float(np.mean(proj_plane_pixel_scales(wcs))) * 3600.0
            total = float(np.nansum(plane))
            for coord in protected:
                x, y = wcs.world_to_pixel(coord)
                _fill_disc(plane, float(x), float(y), radius / cell, 0.0)
            outliers = float(np.nansum(plane))
            kept.append(total - outliers)
            removed.append(outliers)
            fits.writeto(new_prefix + model[len(prefix):], data, header,
                         overwrite=True)
    return len(models), float(np.mean(kept)), float(np.mean(removed))


def subtract_model_image(mlibs, vis, out_vis, to_subtract, protected, tag,
                         args, plan, before, info):
    """
    Method 'model': deep wide-field clean of the whole field (target
    included) inside a mask, model of the protected regions set to zero,
    wsclean -predict of the rest and a single uvsub.

    Raises RuntimeError when a step fails.  Returns the number of sources
    subtracted.
    """
    stem = os.path.splitext(os.path.basename(vis))[0]
    work = os.path.join(args.work_dir, "{}_{}_work.ms".format(stem, tag))
    work_stem = os.path.splitext(os.path.basename(work))[0]
    _rm_ms(work)
    log.info("  Copying %s -> %s", os.path.basename(vis), work)
    shutil.copytree(vis, work)

    section("Clean mask and deep wide-field clean", char=".")
    mask = build_clean_mask(
        os.path.join(args.image_dir, "mask_{}_{}.fits".format(tag, stem)),
        before["image"], info, protected, args.exclusion_radius,
        [s for s in to_subtract if s["origin"] == "manual"],
        args.model_mask_dilate_beams)
    deep = run_imaging(
        mlibs, work, args.image_dir, "model_{}".format(tag),
        imsize=plan["imsize"], cell=plan["cell"], robust=args.model_robust,
        nsigma_automask=args.model_nsigma_automask,
        nsigma_autothreshold=args.model_nsigma_autothreshold,
        niter=args.model_niter, nc=args.n_subimages,
        with_multiscale=args.model_multiscale, taper=plan["taper"],
        datacolumn="CORRECTED_DATA" if args.datacolumn == "corrected"
        else "DATA",
        opt_args=args.opt_args, mask=mask)

    section("Model without the protected regions", char=".")
    prefix = deep["image"][:-len("-MFS-image.fits")]
    new_prefix = os.path.join(args.image_dir,
                              "outliers_{}_{}".format(tag, work_stem))
    n_img, kept, removed = edit_model_images(prefix, new_prefix, protected,
                                             args.exclusion_radius)
    log.info("  %d model image(s): %s-NNNN-model.fits", n_img,
             os.path.basename(new_prefix))
    log.info("  Model flux kept in the data (protected): %10.3f mJy",
             kept * 1e3)
    log.info("  Model flux to subtract (outliers):       %10.3f mJy",
             removed * 1e3)

    section("Predict and subtract", char=".")
    if not hasattr(mlibs, "run_wsclean_predict"):
        raise RuntimeError("This morphen version has no run_wsclean_predict; "
                           "update morphen to use --method model.")
    status = mlibs.run_wsclean_predict(
        work, os.path.basename(new_prefix), nc=int(args.n_subimages),
        outdir=args.image_dir, quiet=True)
    if status != 0 or not has_column(work, "MODEL_DATA"):
        raise RuntimeError("wsclean -predict failed (status {})".format(
            status))
    subtract_model(work, args.subtract_mode)

    _rm_ms(out_vis)
    casatasks.split(vis=work, outputvis=out_vis, datacolumn="corrected")
    if not args.keep_tmp:
        _rm_ms(work)
    return len(to_subtract)


# ===========================================================================
# Stage 4: QA
# ===========================================================================

def _peak_near(data, wcs, coord, half_pix):
    px, py = wcs.world_to_pixel(coord)
    px, py = int(round(float(px))), int(round(float(py)))
    h = max(int(math.ceil(half_pix)), 2)
    cut = data[max(py - h, 0):py + h + 1, max(px - h, 0):px + h + 1]
    return float(np.nanmax(cut)) if cut.size else float("nan")


def run_qa(prefix, before, after, info, sources, groups, protected, radius):
    """Before/after figure and table of peak fluxes at each position."""
    import matplotlib.pyplot as plt

    rms = info["rms"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    to_offsets = _show_field(axes[0], before["image"], rms, "Before")
    _draw_overlays(axes[0], to_offsets, sources, groups, protected, radius)
    to_offsets = _show_field(axes[1], after["image"], rms, "After uv-sub")
    _draw_overlays(axes[1], to_offsets, [], groups, protected, radius)
    fig.savefig(prefix + "_before_after.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  Figure: %s", prefix + "_before_after.png")

    d0, w0, _ = _load_image(before["image"])
    d1, w1, _ = _load_image(after["image"])
    cell = info["cell"]
    rows = []
    for i, p in enumerate(protected, start=1):
        rows.append(("keep{}".format(i), "protected", p, radius / cell))
    for i, s in enumerate(sources, start=1):
        rows.append((str(i), s.get("status", ""), s["coord"],
                     max(s["hx"], s["hy"]) / cell))
    log.info("  %-6s %-10s %-32s %12s %12s %8s", "id", "status", "position",
             "before[mJy]", "after[mJy]", "ratio")
    with open(prefix + "_fluxes.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "status", "position", "peak_before_jy_beam",
                         "peak_after_jy_beam", "ratio"])
        for name, status, coord, half in rows:
            b = _peak_near(d0, w0, coord, half)
            a = _peak_near(d1, w1, coord, half)
            ratio = a / b if b else float("nan")
            log.info("  %-6s %-10s %-32s %12.3f %12.3f %8.3f", name, status,
                     coord_to_casa(coord), b * 1e3, a * 1e3, ratio)
            writer.writerow([name, status, coord_to_casa(coord),
                             "{:.6e}".format(b), "{:.6e}".format(a),
                             "{:.4f}".format(ratio)])
    log.info("  Table:  %s", prefix + "_fluxes.csv")


def run_target_qa(prefix, before, after):
    """
    Before/after comparison of the small image centred on the target.

    Writes a three-panel figure (before, after, after - before; the first two
    share the colour scale) and a table with the peak, the rms (MAD of the
    residual image) and the dynamic range of each image.
    """
    import matplotlib.pyplot as plt
    from astropy.stats import mad_std
    from astropy.wcs.utils import proj_plane_pixel_scales

    stats, maps = [], []
    for label, images in (("before", before), ("after", after)):
        data, wcs, _ = _load_image(images["image"])
        rms = float(mad_std(_load_image(images["residual"])[0],
                            ignore_nan=True))
        peak = float(np.nanmax(data))
        stats.append((label, peak, rms, peak / rms))
        maps.append(np.nan_to_num(data))
    cell = float(np.mean(proj_plane_pixel_scales(wcs))) * 3600.0
    half_x = 0.5 * maps[0].shape[1] * cell
    half_y = 0.5 * maps[0].shape[0] * cell
    extent = [half_x, -half_x, -half_y, half_y]
    rms0 = stats[0][2]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, data, (label, peak, rms, dr) in zip(axes, maps, stats):
        im = ax.imshow(data, origin="lower", cmap="magma", extent=extent,
                       vmin=-3 * rms0, vmax=20 * rms0,
                       interpolation="nearest")
        ax.set_title("{}: peak={:.3f} mJy/beam  rms={:.1f} uJy/beam  "
                     "DR={:.0f}".format(label.capitalize(), peak * 1e3,
                                        rms * 1e6, dr), fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                     label="Jy/beam")
    if maps[0].shape == maps[1].shape:
        im = axes[2].imshow(maps[1] - maps[0], origin="lower", cmap="RdBu_r",
                            extent=extent, vmin=-5 * rms0, vmax=5 * rms0,
                            interpolation="nearest")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02,
                     label="Jy/beam")
    axes[2].set_title("After - before", fontsize=10)
    for ax in axes:
        ax.set_xlabel("RA offset [arcsec]")
        ax.set_ylabel("Dec offset [arcsec]")
    fig.savefig(prefix + "_target_before_after.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    log.info("  Figure: %s", prefix + "_target_before_after.png")

    log.info("  %-8s %14s %14s %10s", "target", "peak[mJy/bm]",
             "rms[uJy/bm]", "DR")
    with open(prefix + "_target_stats.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "peak_jy_beam", "rms_jy_beam",
                         "dynamic_range", "fits"])
        for (label, peak, rms, dr), images in zip(stats, (before, after)):
            log.info("  %-8s %14.3f %14.1f %10.0f", label, peak * 1e3,
                     rms * 1e6, dr)
            writer.writerow([label, "{:.6e}".format(peak),
                             "{:.6e}".format(rms), "{:.1f}".format(dr),
                             os.path.basename(images["image"])])
    log.info("  rms after / before = %.3f", stats[1][2] / stats[0][2])
    log.info("  Table:  %s", prefix + "_target_stats.csv")


# ===========================================================================
# Config file and CLI
# ===========================================================================

DEFAULT_CONFIG_NAME = "uv_sub_cfg.yaml"


def load_config(config_path=None):
    """
    Load a YAML configuration file as a dict of argparse defaults.

    Uses *config_path* when given, otherwise DEFAULT_CONFIG_NAME in the
    current working directory.  Null values are dropped.
    """
    if not _YAML_AVAILABLE:
        log.warning("PyYAML is not installed -- config file support disabled.")
        return {}
    candidate = config_path or os.path.join(os.getcwd(), DEFAULT_CONFIG_NAME)
    if not os.path.exists(candidate):
        if config_path:
            log.error("Config file not found: %s", config_path)
            sys.exit(1)
        return {}
    log.info("Loading config file: %s", candidate)
    with open(candidate) as fh:
        raw = yaml.safe_load(fh) or {}
    return {k: v for k, v in raw.items() if v is not None}


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    cfg = load_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description=(
            "Automated uv-subtraction of outlier sources.\n"
            "Stages: detection image -> source detection and grouping -> "
            "per-group imaging and uvsub -> QA image.\n\n"
            "A YAML config file can supply any of the arguments below.\n"
            "Auto-loaded from '{}' in the current directory, or given with "
            "--config.".format(DEFAULT_CONFIG_NAME)),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    flag = argparse.BooleanOptionalAction

    parser.add_argument("--config", default=None, help="YAML config file.")

    g = parser.add_argument_group("Input and output")
    g.add_argument("--vis", default=None, help="Input Measurement Set.")
    g.add_argument("--output-dir", default=None,
                   help="Output directory (default: <vis stem>_uvsub_out next "
                        "to the MS).")
    g.add_argument("--mlibs-path", default=None,
                   help="Path to morphen/morphen/ (directory with mlibs.py).")
    g.add_argument("--datacolumn", default="data",
                   choices=["data", "corrected"],
                   help="Column of the input MS to work on (default: data).")
    g.add_argument("--force-overwrite", action=flag, default=False,
                   help="Overwrite an existing output MS.")
    g.add_argument("--keep-tmp", action=flag, default=False,
                   help="Keep the intermediate Measurement Sets.")

    g = parser.add_argument_group("Stage switches")
    g.add_argument("--do-detect", action=flag, default=False,
                   help="Make the detection image and detect sources.")
    g.add_argument("--do-subtract", action=flag, default=False,
                   help="Image and subtract each group.")
    g.add_argument("--do-final-image", action=flag, default=False,
                   help="Image the result and write the before/after QA.")
    g.add_argument("--do-target-image", action=flag, default=False,
                   help="Small image centred on the target before and after "
                        "the subtraction, with a comparison figure and "
                        "table.")
    g.add_argument("--dry-run", action=flag, default=False,
                   help="Stop after detection and grouping.")
    g.add_argument("--n-cycles", type=int, default=1,
                   help="Number of detect/subtract cycles (default: 1).")

    g = parser.add_argument_group("Sources to subtract and to keep")
    g.add_argument("--target", default=None,
                   help="Position of the science target, e.g. "
                        "'J2000 09:15:55.5 +44.19.58.0' (default: phase "
                        "centre of the MS).")
    g.add_argument("--keep", nargs="+", default=[],
                   help="Other positions that must not be subtracted.")
    g.add_argument("--exclusion-radius", type=float, default=10.0,
                   help="Radius around the target and --keep positions "
                        "[arcsec] (default: 10).")
    g.add_argument("--sources", nargs="+", default=[],
                   help="Positions to subtract, added to the detected ones "
                        "(first cycle only).")

    g = parser.add_argument_group("Detection image")
    g.add_argument("--detect-pix-per-beam", type=float, default=3.0)
    g.add_argument("--detect-max-imsize", type=int, default=10240)
    g.add_argument("--fov-pb-factor", type=float, default=2.0,
                   help="Field of view in units of the primary beam FWHM.")
    g.add_argument("--fov-arcmin", type=float, default=None,
                   help="Field of view [arcmin]; overrides --fov-pb-factor.")
    g.add_argument("--detect-nsigma", type=float, nargs="+", default=[20.0],
                   help="Detection threshold in MAD sigma; one value per "
                        "cycle (the last one is reused).")
    g.add_argument("--detect-extent-nsigma", type=float, default=5.0,
                   help="Level at which the extent of a source is measured.")
    g.add_argument("--detect-robust", type=float, default=0.5)
    g.add_argument("--detect-niter", type=int, default=50000)
    g.add_argument("--detect-nsigma-automask", default="5.0")
    g.add_argument("--detect-nsigma-autothreshold", default="3.0")
    g.add_argument("--detect-multiscale", action=flag, default=False)

    g = parser.add_argument_group("Target image (--do-target-image)")
    g.add_argument("--target-imsize", type=int, default=1024,
                   help="Size of the target image [pixels] (default: 1024).")
    g.add_argument("--target-pix-per-beam", type=float, default=5.0)
    g.add_argument("--target-robust", type=float, default=0.5)
    g.add_argument("--target-niter", type=int, default=100000)
    g.add_argument("--target-nsigma-automask", default="4.0")
    g.add_argument("--target-nsigma-autothreshold", default="2.0")
    g.add_argument("--target-multiscale", action=flag, default=True)

    g = parser.add_argument_group("Wide-field model (--method model)")
    g.add_argument("--model-robust", type=float, default=0.5)
    g.add_argument("--model-niter", type=int, default=200000)
    g.add_argument("--model-nsigma-automask", default="4.0")
    g.add_argument("--model-nsigma-autothreshold", default="1.5")
    g.add_argument("--model-multiscale", action=flag, default=True)
    g.add_argument("--model-mask-dilate-beams", type=float, default=2.0,
                   help="Growth of the detected islands in the clean mask "
                        "[beams] (default: 2).")

    g = parser.add_argument_group("Sub-images and subtraction")
    g.add_argument("--method", default="model",
                   choices=["phaseshift", "shift", "model"],
                   help="Subtraction method (default: model).")
    g.add_argument("--sub-model-flux-warn", type=float, default=2.0,
                   help="phaseshift / shift: warn when the model of a "
                        "sub-image holds more than this factor times the "
                        "catalogued flux of its sources; 0 disables "
                        "(default: 2).")
    g.add_argument("--sub-model-neg-warn", type=float, default=0.2,
                   help="phaseshift / shift: warn when the negative model "
                        "flux of a sub-image exceeds this fraction of the "
                        "positive one; 0 disables (default: 0.2).")
    g.add_argument("--subtract-mode", default="uvsub",
                   choices=["uvsub", "peel"])
    g.add_argument("--sub-pix-per-beam", type=float, default=5.0)
    g.add_argument("--max-subimage", type=int, default=1024)
    g.add_argument("--min-subimage", type=int, default=256)
    g.add_argument("--subimage-pad-beams", type=float, default=10.0,
                   help="Margin around the sources of a group [beams].")
    g.add_argument("--sub-robust", type=float, default=0.0)
    g.add_argument("--sub-niter", type=int, default=100000)
    g.add_argument("--sub-nsigma-automask", default="7.0")
    g.add_argument("--sub-nsigma-autothreshold", default="1.5")
    g.add_argument("--sub-multiscale", action=flag, default=True)
    g.add_argument("--n-subimages", type=int, default=4,
                   help="WSClean -channels-out (default: 4).")
    g.add_argument("--opt-args", default="''",
                   help="Extra WSClean arguments, shell quoted, e.g. "
                        "\"' -circular-beam '\".")

    if cfg:
        known = {a.dest for a in parser._actions}
        unknown = sorted(set(cfg) - known)
        if unknown:
            parser.error("Unknown config key(s): {}".format(unknown))
        parser.set_defaults(**cfg)
    args = parser.parse_args(argv)

    if args.vis is None:
        parser.error("--vis is required (CLI or config file).")
    if not (args.do_detect or args.sources):
        parser.error("Nothing to subtract: use --do-detect and/or --sources.")
    if args.method == "model" and not args.do_detect:
        parser.error("--method model (the default) needs --do-detect; "
                     "with --sources only, use --method phaseshift.")
    if isinstance(args.detect_nsigma, (int, float)):
        args.detect_nsigma = [float(args.detect_nsigma)]
    for name in ("keep", "sources"):
        if isinstance(getattr(args, name), str):
            setattr(args, name, [getattr(args, name)])
    return args


# ===========================================================================
# Main entry point
# ===========================================================================

def main(argv=None):
    args = parse_args(argv)

    vis = os.path.abspath(args.vis.rstrip("/"))
    if not os.path.isdir(vis):
        log.error("Measurement Set not found: %s", vis)
        sys.exit(1)
    stem = os.path.splitext(os.path.basename(vis))[0]
    out_dir = os.path.abspath(
        args.output_dir or os.path.join(os.path.dirname(vis),
                                        stem + "_uvsub_out"))
    args.image_dir = os.path.join(out_dir, "images")
    args.work_dir = os.path.join(out_dir, "work")
    for d in (out_dir, args.image_dir, args.work_dir):
        os.makedirs(d, exist_ok=True)
    _attach_log_file(out_dir)
    casatasks.casalog.setlogfile(os.path.join(out_dir, "casa_uv_sub.log"))

    final_vis = os.path.join(out_dir, stem + "_uvsub.ms")
    subtracting = args.do_subtract and not args.dry_run
    if subtracting and os.path.exists(final_vis) and not args.force_overwrite:
        log.error("Output already exists: %s  (use --force-overwrite)",
                  final_vis)
        sys.exit(1)

    section("uv_sub: {}".format(vis), char="=")
    for key in sorted(vars(args)):
        log.info("  %-28s %s", key, getattr(args, key))

    mlibs = import_mlibs(args.mlibs_path)
    if mlibs is None:
        sys.exit(1)

    centre, frame = get_phase_centre(vis)
    target = parse_coord(args.target) if args.target else centre
    protected = [target] + [parse_coord(k) for k in args.keep]
    log.info("  Phase centre: %s", coord_to_casa(centre, frame))
    for p in protected:
        log.info("  Protected:    %s  (r = %.1f arcsec)",
                 coord_to_casa(p, frame), args.exclusion_radius)

    geom = get_ms_geometry(vis)
    sub_cell = geom["beam"] / args.sub_pix_per_beam
    plan = plan_detection_image(
        geom, args.detect_pix_per_beam, args.detect_max_imsize,
        args.fov_pb_factor, args.fov_arcmin) if args.do_detect else None

    def field_image(ms_path, base_name, reuse=False):
        return run_imaging(
            mlibs, ms_path, args.image_dir, base_name,
            imsize=plan["imsize"], cell=plan["cell"],
            robust=args.detect_robust,
            nsigma_automask=args.detect_nsigma_automask,
            nsigma_autothreshold=args.detect_nsigma_autothreshold,
            niter=args.detect_niter, nc=args.n_subimages,
            with_multiscale=args.detect_multiscale, taper=plan["taper"],
            datacolumn="CORRECTED_DATA" if (ms_path == vis and
                                            args.datacolumn == "corrected")
            else "DATA",
            opt_args=args.opt_args, reuse=reuse)

    def target_image(ms_path, base_name):
        # -shift is only needed when the target is not the phase centre.
        off_centre = target.separation(centre).arcsec > 0.01
        return run_imaging(
            mlibs, ms_path, args.image_dir, base_name,
            imsize=args.target_imsize,
            cell=geom["beam"] / args.target_pix_per_beam,
            robust=args.target_robust,
            nsigma_automask=args.target_nsigma_automask,
            nsigma_autothreshold=args.target_nsigma_autothreshold,
            niter=args.target_niter, nc=args.n_subimages,
            with_multiscale=args.target_multiscale,
            shift=coord_to_wsclean(target) if off_centre else None,
            datacolumn="CORRECTED_DATA" if (ms_path == vis and
                                            args.datacolumn == "corrected")
            else "DATA",
            opt_args=args.opt_args)

    current = vis
    after = None
    target_after = None
    for cycle in range(1, args.n_cycles + 1):
        tag = "c{}".format(cycle)
        prefix = os.path.join(out_dir, "{}_{}".format(stem, tag))
        section("Cycle {}/{}".format(cycle, args.n_cycles), char="=")

        # ---- Stage 1/2: detection --------------------------------------
        sources, before, info = [], None, None
        if args.do_detect:
            section("Stage 1: detection image")
            before = after or field_image(
                current, "field_{}_before".format(tag),
                reuse=(cycle == 1 and not args.force_overwrite))
            nsigma = args.detect_nsigma[min(cycle, len(args.detect_nsigma)) - 1]
            sources, info = detect_sources(before, nsigma,
                                           args.detect_extent_nsigma)
        if cycle == 1:
            sources += manual_sources(args.sources)

        section("Stage 2: exclusion and grouping")
        to_subtract = apply_exclusion(sources, protected,
                                      args.exclusion_radius)
        if args.method == "model":
            # No sub-images: every source outside the protected circles is
            # in the wide-field model.
            groups = []
        else:
            groups = group_sources(
                to_subtract, centre, protected, args.exclusion_radius,
                sub_cell, max_subimage=args.max_subimage,
                min_subimage=args.min_subimage,
                pad=args.subimage_pad_beams * geom["beam"])
        write_catalogue(prefix + "_sources.csv", sources)
        write_regions(prefix + "_sources.reg", sources, groups, protected,
                      args.exclusion_radius)
        if before is not None:
            plot_detections(prefix + "_detections.png", before, info["rms"],
                            sources, groups, protected, args.exclusion_radius)
        if args.method == "model":
            log.info("  %d source(s) to subtract with the wide-field model",
                     len(to_subtract))
        else:
            log.info("  %d source(s) to subtract in %d group(s)",
                     sum(len(g["members"]) for g in groups), len(groups))

        if not (to_subtract if args.method == "model" else groups):
            log.info("  Nothing to subtract in this cycle.")
            break
        if not subtracting:
            log.info("  Subtraction not requested (--do-subtract off or "
                     "--dry-run): stopping here.")
            break

        # The target is imaged before the subtraction with the same settings
        # used after it, so that the two images can be compared.
        target_before = None
        if args.do_target_image:
            section("Target image before the subtraction")
            try:
                target_before = target_after or target_image(
                    current, "target_{}_before".format(tag))
            except RuntimeError as exc:
                log.error("  %s", exc)

        # ---- Stage 3: subtraction --------------------------------------
        section("Stage 3: subtraction ({}, {})".format(args.method,
                                                       args.subtract_mode))
        cycle_vis = os.path.join(args.work_dir,
                                 "{}_uvsub_{}.ms".format(stem, tag))
        if args.method == "phaseshift":
            done = subtract_phaseshift(mlibs, current, cycle_vis, groups, tag,
                                       args, sub_cell, centre, frame)
            log.info("  %d of %d group(s) subtracted", done, len(groups))
            report_suspect_groups(groups)
        elif args.method == "shift":
            done = subtract_shift(mlibs, current, cycle_vis, groups, tag,
                                  args, sub_cell)
            log.info("  %d of %d group(s) subtracted", done, len(groups))
            report_suspect_groups(groups)
        else:
            try:
                done = subtract_model_image(
                    mlibs, current, cycle_vis, to_subtract, protected, tag,
                    args, plan, before, info)
            except RuntimeError as exc:
                log.error("  %s", exc)
                log.error("  Subtraction failed; no output was written for "
                          "this cycle.")
                sys.exit(1)
            log.info("  %d source(s) subtracted", done)
        if current != vis:
            _rm_ms(current)
        current = cycle_vis
        # Later cycles read the DATA column of the cycle output.
        args.datacolumn = "data"

        # ---- Stage 4: QA -----------------------------------------------
        after = None
        if args.do_final_image and plan is not None:
            section("Stage 4: QA image")
            after = field_image(current, "field_{}_after".format(tag))
            run_qa(prefix, before, after, info, sources, groups, protected,
                   args.exclusion_radius)
        target_after = None
        if target_before is not None:
            section("Target image after the subtraction")
            try:
                target_after = target_image(current,
                                            "target_{}_after".format(tag))
                run_target_qa(prefix, target_before, target_after)
            except RuntimeError as exc:
                log.error("  %s", exc)

    if current != vis:
        _rm_ms(final_vis)
        shutil.move(current, final_vis)
        section("Done: {}".format(final_vis), char="=")
    else:
        section("Done (no visibilities were written)", char="=")
    if not args.keep_tmp and not os.listdir(args.work_dir):
        os.rmdir(args.work_dir)


if __name__ == "__main__":
    main()