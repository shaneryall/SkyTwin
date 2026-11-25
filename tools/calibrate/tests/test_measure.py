"""Per-frame measures on frames clipped at zero and with corrupted rows."""

import numpy as np

from skytwin_calibrate import measure, profile


def _frame(mu, sigma=30.0, n=512, seed=1):
    """An n × n frame of rounded Gaussian noise about `mu`, unclipped."""
    rng = np.random.default_rng(seed)
    return np.round(rng.normal(mu, sigma, (n, n)))


def _hits(d, k, seed=2):
    """`d` with `k` single-pixel hits of 2000 ADU, in place."""
    rng = np.random.default_rng(seed)
    ys, xs = rng.integers(5, d.shape[0] - 5, k), rng.integers(5, d.shape[1] - 5, k)
    d[ys, xs] += 2000.0
    return d


def test_noise_and_hits_survive_clipping():
    """With the bias at or below zero (63% of pixels clipped at -10, where the MAD reads 0), noise and hits
    are still measured."""
    for mu in (40.0, 0.0, -10.0):
        d = np.clip(_hits(_frame(mu), 100), 0, None)
        b = measure.background(d, 0)
        assert abs(b["noise"] - 30.0) < 1.5, (mu, b["noise"])
        assert 95 <= measure.sharp_events(d, b["median"], b["noise"]) <= 100


def test_bad_rows_found_and_kept_out_of_hits():
    """Salt-and-pepper rows 100-101, dropped row 300 and a level jump at 400 are found; a clipped frame
    with none has no bad rows."""
    rng = np.random.default_rng(3)
    d = _hits(_frame(40.0), 100)
    d[100] = rng.choice([0.0, 60000.0], d.shape[1])
    d[101] = rng.choice([0.0, 60000.0], d.shape[1])
    d[300] = 0.0
    d[400] += 25.0
    d = np.clip(d, 0, None)
    b = measure.background(d, 0)
    bad = measure.bad_rows(d, b["noise"])
    assert set(np.flatnonzero(bad)) == {100, 101, 300, 400}
    n = measure.sharp_events(d, b["median"], b["noise"], bad)
    assert 90 <= n <= 100
    c = np.clip(_frame(-10.0), 0, None)
    assert not measure.bad_rows(c, measure.background(c, 0)["noise"]).any()


def test_comb_gradient_and_storm_are_not_bad_rows():
    """An odd/even offset of 40 ADU is measured, not flagged; nor is glare steep along both axes and
    curving at the edge, nor a radiation storm of 20,000 hits filling every row."""
    n = 512
    yy, xx = np.mgrid[0:n, 0:n]
    d = _frame(60.0) + np.where(yy % 2 == 0, 20.0, -20.0)
    d = np.clip(d, 0, None)
    assert abs(measure.row_alternation(d) - 40.0) < 3
    assert not measure.bad_rows(d, measure.background(d, 0)["noise"]).any()
    g = _frame(0.0) + 3000.0 * np.exp(-np.hypot(xx, yy) / 120.0) + 200.0
    assert not measure.bad_rows(g, measure.background(g, 0)["noise"]).any()
    s = np.clip(_hits(_frame(40.0), 20000), 0, None)
    b = measure.background(s, 0)
    bad = measure.bad_rows(s, b["noise"])
    assert bad.sum() <= 2
    assert measure.sharp_events(s, b["median"], b["noise"], bad) > 15000


def test_rate_spread_removes_poisson_noise():
    """A unit lognormal spread is recovered (p95 at e^1.645 = 5.2); pure Poisson scatter with no real
    spread is shrunk to almost nothing."""
    rng = np.random.default_rng(4)
    true = np.exp(rng.normal(0, 1.0, 2000))
    counts = rng.poisson(100 * true)
    ratios = np.maximum(counts, 1) / 100.0
    sc, _ = profile.rate_spread(ratios, counts)
    v = np.log(sc["empirical"])
    assert abs(np.median(v)) < 0.05
    assert abs(v[95] - 1.645) < 0.15
    counts = rng.poisson(100, 2000)
    sc, k = profile.rate_spread(np.maximum(counts, 1) / 100.0, counts)
    assert k < 0.3


def test_readout_faults_learned_from_bands():
    """10 of 40 frames carry a 3-row noise band and a dropped row; one of them is also cut short."""
    from skytwin_calibrate import analyse

    rng = np.random.default_rng(6)
    rows = []
    for i in range(40):
        d = _frame(40.0, n=256, seed=10 + i)
        if i % 4 == 0:
            d[50:53] = rng.choice([0.0, 60000.0], (3, 256))
            d[120] = 0.0
            if i == 0:
                d[200:] = 0.0
        d = np.clip(d, 0, None)
        b = measure.background(d, 0)
        bad = measure.bad_rows(d, b["noise"])
        rows.append(dict(shape=list(d.shape), bands=measure.band_list(d, bad, 65535.0)))
    block, _ = analyse.readout_faults(rows)
    assert block["probability"] == 0.25
    assert block["truncate_probability"] == 0.1
    assert block["kind_weights"][0] == 10 and block["kind_weights"][1] == 10
    assert block["bands"]["empirical"][50] == 2
    clean = [dict(shape=[256, 256], bands=[]) for _ in range(30)]
    assert analyse.readout_faults(clean)[0] is None


def test_radiation_readout_from_the_split():
    """Exposure t = 3 s and read-out R = 6 s put hits in the halves as 3 + 4.5 = 7.5 against 3 + 1.5 = 4.5."""
    from skytwin_calibrate import analyse

    rows = [dict(sharp=1200, sharp_upper=750, meta={"exposure_s": 3.0}) for _ in range(20)]
    ro, first, _ = analyse.radiation_readout(rows)
    assert first == "bottom" and abs(ro - 6.0) < 0.01
    flip = [dict(sharp=1200, sharp_upper=450, meta={"exposure_s": 3.0}) for _ in range(20)]
    assert analyse.radiation_readout(flip)[1] == "top"
    even = [dict(sharp=1200, sharp_upper=600, meta={"exposure_s": 3.0}) for _ in range(20)]
    assert analyse.radiation_readout(even)[0] is None


def _snr(m):
    """S/N of a star of magnitude `m`: 400 at magnitude 2."""
    return 400 * 10 ** (-0.4 * (m - 2))


def test_forced_zeropoint_unbiased_and_flags_nonlinearity():
    """A flat zeropoint is recovered without warning; one drifting 0.4 mag per mag is flagged."""
    from skytwin_calibrate import astrometry

    rng = np.random.default_rng(1)
    lin = [(m, 15 + 0.3 * rng.normal(), _snr(m)) for m in rng.uniform(2, 8, 3000)]
    zp, notes = astrometry.zeropoint_from_forced(lin)
    assert abs(zp - 15.0) < 0.05 and not any("WARNING" in n for n in notes)
    drift = [(m, 15 + 0.4 * (m - 4) + 0.1 * rng.normal(), _snr(m)) for m in rng.uniform(2, 8, 3000)]
    assert any("WARNING" in n for n in astrometry.zeropoint_from_forced(drift)[1])


def test_forced_zeropoint_fits_a_colour_term():
    """A zeropoint of 15 + 0.35 × colour gives zeropoint 15 and colour term -0.35."""
    from skytwin_calibrate import astrometry

    rng = np.random.default_rng(2)
    m, col = rng.uniform(2, 8, 4000), rng.uniform(0, 2.5, 4000)
    rows = [(a, 15 + 0.35 * c + 0.15 * rng.normal(), _snr(a), c) for a, c in zip(m, col)]
    zp, notes, ct = astrometry.zeropoint_from_forced(rows, with_colour=True)
    assert abs(zp - 15.0) < 0.03 and abs(ct["coefficient"] + 0.35) < 0.02, (zp, ct)
    assert not any("WARNING" in n for n in notes)


def test_profile_contract_records_and_checks_the_catalogue(tmp_path):
    """A CSV catalogue is recorded with its columns, rows and hash; a different extract fails the check."""
    import json

    from skytwin_calibrate import contract

    cat = tmp_path / "stars.csv"
    cat.write_text(
        "source_id,ra,dec,phot_g_mean_mag,pmra,pmdec,parallax,bp_rp\n1,10,20,5.0,,,,1.2\n2,11,21,6.0,,,,0.4\n"
    )
    req = contract.catalog_block(str(cat), 9.87, name="test extract", with_colour=True)
    assert req["kind"] == "csv" and req["rows"] == 2 and req["mag_limit"] == 9.9 and req["columns"]["color"] == "bp_rp"
    prof = contract.stamp({"camera": {"width": 8, "height": 8}}, [{"INSTRUME": "CAM-1"}], ["data/"], 2, catalog=req)
    assert prof["schema_version"] == 1 and prof["sensor"]["name"] == "CAM-1" and prof["learned"]["frames"] == 2
    p = tmp_path / "profile.json"
    p.write_text(json.dumps(prof))
    assert contract.check(str(p), str(cat)) == []
    other = tmp_path / "other.csv"
    other.write_text(cat.read_text() + "3,12,22,7.0,,,,0.9\n")
    probs = contract.check(str(p), str(other))
    assert any("rows" in x for x in probs) and any("SHA-256" in x for x in probs)
    assert "scene.catalog" not in contract.describe(str(p)) and '"path"' in contract.describe(str(p))


def test_background_cap_keeps_raw_frames_with_a_high_bias():
    """Raw frames whose bias alone is about 1000 ADU are not mistaken for glare; a frame far above the
    archive's usual level still is, and archives with low levels keep the 1000 ADU floor."""
    raw = [{"median": m} for m in (969.0, 1014.0, 1030.0, 1051.0, 1400.0, 6000.0)]
    assert profile.background_cap(raw) == 2 * 1040.5
    assert [r["median"] for r in profile.usable(raw)] == [969.0, 1014.0, 1030.0, 1051.0, 1400.0]
    low = [{"median": m} for m in (20.0, 45.0, 60.0, 1500.0)]
    assert profile.background_cap(low) == profile.GLARE_FLOOR_ADU
    assert [r["median"] for r in profile.usable(low)] == [20.0, 45.0, 60.0]
    assert profile.usable([{"median": 5.0, "error": "x"}]) == []


def test_catalogue_limit_deepens_for_small_fields(monkeypatch):
    """A sub-degree field holds a handful of stars at the default limit; the limit goes fainter until
    there are enough, and stops at the cap."""
    from skytwin_calibrate import astrometry

    calls = []

    def fake(skytwin, work, w_, h_, wcs, cat, lim, tag):
        calls.append(lim)
        return [(0.0, 0.0, 0.0, i) for i in range(int(10 ** (0.5 * (lim - 8))))], 1.0, "positive"

    monkeypatch.setattr(astrometry, "_catalogue", fake)
    stars, _, _, lim = astrometry.catalogue_to_count(None, "", 100, 100, None, "c.csv", 8.0, 0)
    assert calls == [8.0, 10.0, 12.0] and lim == 12.0 and len(stars) >= 40
    calls.clear()
    _, _, _, lim = astrometry.catalogue_to_count(None, "", 100, 100, None, "c.csv", 8.0, 0, min_stars=10**9)
    assert lim == 18.0 and calls[-1] == 18.0


def test_hot_pixel_density_is_per_chip_pixel_when_binned():
    """Skytwin counts hot pixels per million chip pixels; a 2 × 2 binned image measures four times
    as many per million of its own pixels."""
    from skytwin_calibrate import learn

    assert learn.chip_density(785.83, 2, 2) == 196.46
    assert learn.chip_density(84.28, 1, 1) == 84.28
