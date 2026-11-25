//! `skytwin`: render a scene file to a FITS image and a truth JSON.
//!
//! Besides the images it writes `<out>.provenance.json`, rewritten as the
//! run goes, recording the command, inputs and outputs with their SHA-256
//! hashes.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Mutex, OnceLock};

use skytwin_core::fits::write_image;
use skytwin_core::param::DrawCtx;
use skytwin_core::rng::Rng;
use skytwin_core::{
    add_catalog_stars, add_tle_file_targets, load_config, render_profile, target_visible, Config,
    Scene, TimeSpec, Utc,
};

/// Command-line usage, printed for `--help` and on bad arguments.
const USAGE: &str = "usage: skytwin <scene.json> <out.fits> [--seed N] [--write-flat] [--frames N] [--jobs N] [--compact-truth]\n       skytwin --version\nWrites <out.fits> and <out>.truth.json.\nThe scene file may name an instrument profile: \"instrument\": \"profiles/x.json\".\n--frames N renders N independent frames (a \"batch\"), on --jobs threads (default: all cores).\n--compact-truth writes the truth JSON without indentation (about 30% smaller, faster).";

/// Output options.
#[derive(Clone, Copy, Default)]
struct Opts {
    /// Worker threads for a batch (`--jobs`); default: all cores.
    jobs: Option<usize>,
    /// Frames of a batch (`--frames`); overrides the batch block.
    frames: Option<usize>,
    /// Write the truth JSON without indentation (`--compact-truth`).
    compact: bool,
    /// Do not print a line per frame written.
    quiet: bool,
}

/// `path: error`, the form of every file error the tool reports.
fn path_err(path: &Path) -> impl Fn(std::io::Error) -> String + '_ {
    move |e| format!("{}: {e}", path.display())
}

/// Creates the folder `out_path` goes in, if it has one and it is missing.
fn make_output_folder(out_path: &Path) -> Result<(), String> {
    match out_path.parent().filter(|d| !d.as_os_str().is_empty()) {
        Some(dir) => std::fs::create_dir_all(dir).map_err(path_err(dir)),
        None => Ok(()),
    }
}

/// Parses the arguments, renders, and keeps the provenance record up to
/// date (written once before loading the scene and once at the end).
fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut pos = Vec::new();
    let mut seed_override = None;
    let mut write_flat = false;
    let mut opts = Opts::default();
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "-V" | "--version" => {
                println!("skytwin {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--write-flat" => write_flat = true,
            "--compact-truth" => opts.compact = true,
            "--jobs" | "--frames" => {
                let v = args.next().ok_or(format!("{a} needs a value"))?;
                let n = v.parse::<usize>().map_err(|e| format!("{a}: {e}"))?;
                if n == 0 {
                    return Err(format!("{a}: must be at least 1"));
                }
                if a == "--jobs" {
                    opts.jobs = Some(n);
                } else {
                    opts.frames = Some(n);
                }
            }
            "--seed" => {
                let v = args.next().ok_or("--seed needs a value")?;
                seed_override = Some(v.parse::<u64>().map_err(|e| format!("--seed: {e}"))?);
            }
            _ => pos.push(PathBuf::from(a)),
        }
    }
    let [cfg_path, out_path] = <[PathBuf; 2]>::try_from(pos).map_err(|_| USAGE.to_string())?;
    make_output_folder(&out_path)?;

    let mut prov = Provenance::start(&cfg_path, &out_path, seed_override);
    prov.write()?;
    let result = load_config(&cfg_path).and_then(|cfg| {
        prov.note_inputs(&cfg_path, &cfg);
        let mut files = generate(&cfg, seed_override.unwrap_or(cfg.seed), &out_path, opts)?;
        if write_flat {
            files.push(flat_file(&cfg, &out_path)?);
        }
        Ok(files)
    });
    prov.finish(&result);
    let written = prov.write();
    result?;
    written.map_err(|e| format!("the frames were written, but not their provenance record: {e}"))
}

/// The index file of a sequence or batch written to `out_path`:
/// `visit.g.fits` gives `visit.g.batch.json`, so batches whose names differ
/// only before the last extension keep separate indexes.
fn index_path(out_path: &Path, kind: &str) -> PathBuf {
    PathBuf::from(format!(
        "{}.{kind}.json",
        out_path.with_extension("").display()
    ))
}

/// Renders the frame, sequence or batch; returns the files written.
///
/// In a sequence each frame starts `cadence_s` after the last and frame `k`
/// uses seed + `k`; the detector maps stay fixed. Slow pointing changes are
/// a random walk of the commanded pointing plus a settling offset of the
/// true boresight from it, decaying with time and in a direction fixed for
/// the sequence; both are drawn from their own RNG stream, so they do not
/// disturb the frames' draws. The header reports the commanded pointing, so
/// the header bias moves by minus the settling offset.
fn generate(cfg: &Config, seed: u64, out_path: &Path, opts: Opts) -> Result<Vec<PathBuf>, String> {
    if cfg.batch.is_some() || opts.frames.is_some() {
        if cfg.sequence.is_some() {
            return Err("--frames: the scene has a sequence; give one or the other".into());
        }
        return generate_batch(cfg, seed, out_path, opts);
    }
    let Some(q) = cfg.sequence.clone() else {
        return render_one(cfg, cfg.scene.clone(), seed, out_path, opts);
    };
    let n = q.frames;
    let t0 = cfg.scene.time.as_ref().expect("checked").start_utc.clone();
    let start = Utc::parse(&t0).ok_or("scene.time.start_utc: bad time")?;
    let stem = out_path.with_extension("");
    let mut index = Vec::new();
    let mut written = Vec::new();
    let max_steps = q.max_steps.unwrap_or(100 * n);
    let (mut step, mut kept, mut skipped) = (0usize, 0usize, 0usize);
    let mut prng = Rng::new(seed ^ 0x5E9_0E7C_E000_0008);
    let ctx = &mut DrawCtx::default();
    let settle = q.settling.as_ref().map(|s| {
        let a = s.offset_arcsec.draw(&mut prng, ctx).max(0.0);
        let tau = s.time_s.draw(&mut prng, ctx).max(1e-6);
        let az = prng.range(0.0, 2.0 * std::f64::consts::PI);
        (a * libm::sin(az), a * libm::cos(az), tau)
    });
    let base = cfg.scene.pointing.clone();
    let (mut walk_e, mut walk_n, mut walk_roll) = (0.0, 0.0, 0.0);
    while kept < n {
        if step >= max_steps {
            return Err(format!(
                "sequence: found {kept} of {n} frames with the target visible in {max_steps} steps"
            ));
        }
        let mut scene = cfg.scene.clone();
        let t_rel = step as f64 * q.cadence_s;
        let tk = start.add_seconds(t_rel).to_string();
        if let Some(w) = &q.pointing_walk {
            if step > 0 {
                walk_e += w.step_arcsec * prng.normal();
                walk_n += w.step_arcsec * prng.normal();
                walk_roll += w.roll_step_deg * prng.normal();
            }
        }
        let mut offsets = None;
        if let (Some(p0), true) = (&base, q.pointing_walk.is_some() || settle.is_some()) {
            let (se, sn) = settle.map_or((0.0, 0.0), |(e, n, tau)| {
                let f = libm::exp(-t_rel / tau);
                (e * f, n * f)
            });
            let (ra0, dec0) = (p0.ra_deg.expect("checked"), p0.dec_deg.expect("checked"));
            let shift = |e: f64, n: f64| {
                let dec = (dec0 + n / 3600.0).clamp(-90.0, 90.0);
                (
                    (ra0 + e / 3600.0 / libm::cos(dec0.to_radians())).rem_euclid(360.0),
                    dec,
                )
            };
            let (ra, dec) = shift(walk_e + se, walk_n + sn);
            let mut p = p0.clone();
            p.ra_deg = Some(ra);
            p.dec_deg = Some(dec);
            p.roll_deg = Some(p0.roll_deg.unwrap_or(0.0) + walk_roll);
            scene.pointing = Some(p);
            if settle.is_some() {
                let [be, bn] = scene.header_bias_arcsec.unwrap_or([0.0, 0.0]);
                scene.header_bias_arcsec = Some([be - se, bn - sn]);
            }
            offsets = Some([walk_e, walk_n, walk_roll, se, sn]);
        }
        scene.time = Some(TimeSpec {
            start_utc: tk.clone(),
        });
        step += 1;
        if let Some(name) = &q.only_visible {
            if !target_visible(&scene, &cfg.profile.camera, name).map_err(|e| e.to_string())? {
                skipped += 1;
                continue;
            }
        }
        let k = kept;
        let frame_seed = seed.wrapping_add(k as u64);
        let path = PathBuf::from(format!("{}_{k:03}.fits", stem.display()));
        written.extend(render_one(cfg, scene, frame_seed, &path, opts)?);
        index.push(serde_json::json!({
            "frame": k,
            "file": path.file_name().map(|f| f.to_string_lossy().into_owned()),
            "start_utc": tk,
            "seed": frame_seed,
            "skipped_before": skipped,
            "walk_arcsec": offsets.map(|o| [o[0], o[1]]),
            "walk_roll_deg": offsets.map(|o| o[2]),
            "settling_arcsec": offsets.map(|o| [o[3], o[4]]),
        }));
        kept += 1;
    }
    let idx = index_path(out_path, "sequence");
    write_index(&idx, &index, true)?;
    println!("wrote {n} frames and {}", idx.display());
    written.push(idx);
    Ok(written)
}

/// Independent frames on all cores: frame k uses seed + k (and its own
/// random pointing, if asked). The profile is loaded once, and with a fixed
/// pointing the catalogue is read once. Output does not depend on the
/// number of threads.
///
/// Frame numbers in file names have at least 5 digits, so names do not
/// depend on the number of frames. The first error stops all workers.
fn generate_batch(
    cfg: &Config,
    seed: u64,
    out_path: &Path,
    opts: Opts,
) -> Result<Vec<PathBuf>, String> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let gen = skytwin_core::Generator::new(cfg.clone(), Some(seed))?;
    let n = opts.frames.unwrap_or(gen.frames());
    let jobs = opts
        .jobs
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
        .min(n);
    let digits = ((n.max(2) - 1).ilog10() as usize + 1).max(5);
    let stem = out_path.with_extension("");
    let name = |k: usize| PathBuf::from(format!("{}_{k:0digits$}.fits", stem.display()));
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let first_err: Mutex<Option<String>> = Mutex::new(None);
    let t0 = std::time::Instant::now();
    let quiet = Opts {
        quiet: true,
        ..opts
    };
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= n || failed.load(Ordering::Relaxed) {
                    break;
                }
                let res = gen
                    .render(k as u64)
                    .and_then(|f| write_frame(cfg, &f, &name(k), quiet));
                if let Err(e) = res {
                    failed.store(true, Ordering::Relaxed);
                    first_err
                        .lock()
                        .expect("lock")
                        .get_or_insert(format!("frame {k}: {e}"));
                    break;
                }
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                if !opts.quiet && (d % 100 == 0 || d == n) {
                    let el = t0.elapsed().as_secs_f64();
                    eprintln!("{d}/{n} frames, {:.1} frames/s", d as f64 / el);
                }
            });
        }
    });
    if let Some(e) = first_err.into_inner().expect("lock") {
        return Err(e);
    }
    let index: Vec<_> = (0..n)
        .map(|k| {
            let mut e = serde_json::json!({
                "frame": k,
                "file": name(k).file_name().map(|f| f.to_string_lossy().into_owned()),
                "seed": gen.seed_of(k as u64),
            });
            if let Some((ra, dec, roll)) = gen.pointing(k as u64) {
                e["pointing"] =
                    serde_json::json!({ "ra_deg": ra, "dec_deg": dec, "roll_deg": roll });
            }
            e
        })
        .collect();
    let idx = index_path(out_path, "batch");
    write_index(&idx, &index, false)?;
    println!(
        "wrote {n} frames in {:.1} s on {jobs} threads, and {}",
        t0.elapsed().as_secs_f64(),
        idx.display()
    );
    let mut written: Vec<PathBuf> = (0..n)
        .flat_map(|k| [name(k), name(k).with_extension("truth.json")])
        .collect();
    written.push(idx);
    Ok(written)
}

/// Writes a sequence or batch index, pretty-printed or compact.
fn write_index(path: &Path, index: &[serde_json::Value], pretty: bool) -> Result<(), String> {
    let text = if pretty {
        serde_json::to_string_pretty(index)
    } else {
        serde_json::to_string(index)
    }
    .map_err(|e| e.to_string())?;
    std::fs::write(path, text + "\n").map_err(path_err(path))
}

/// Renders one frame of `scene`, with its catalogue stars and TLE targets
/// loaded, and writes its FITS file and truth.
fn render_one(
    cfg: &Config,
    mut scene: Scene,
    seed: u64,
    out_path: &Path,
    opts: Opts,
) -> Result<Vec<PathBuf>, String> {
    let cam = &cfg.profile.camera;
    add_catalog_stars(&mut scene, cam).map_err(|e| e.to_string())?;
    add_tle_file_targets(&mut scene, cam).map_err(|e| e.to_string())?;
    let frame = render_profile(&scene, &cfg.profile, seed).map_err(|e| e.to_string())?;
    write_frame(cfg, &frame, out_path, opts)
}

/// Writes a rendered frame's FITS file and truth; returns their paths.
fn write_frame(
    cfg: &Config,
    frame: &skytwin_core::Frame,
    out_path: &Path,
    opts: Opts,
) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::with_capacity(frame.adu.len() * 2 + 32 * 2880);
    write_image(
        &mut out,
        frame.width,
        frame.height,
        &frame.adu,
        frame.output,
        &frame
            .truth
            .styled_cards(cfg.profile.header.as_ref(), cfg.profile.layout().binning),
    )
    .map_err(path_err(out_path))?;
    write_hashed(out_path, &out)?;
    let truth_path = out_path.with_extension("truth.json");
    let truth = if opts.compact {
        serde_json::to_string(&frame.truth)
    } else {
        serde_json::to_string_pretty(&frame.truth)
    }
    .map_err(|e| e.to_string())?;
    write_hashed(&truth_path, (truth + "\n").as_bytes())?;
    if !opts.quiet {
        println!(
            "wrote {} ({}x{}, {} stars, {} streaks, {} targets, {} on the image) and {}",
            out_path.display(),
            frame.width,
            frame.height,
            frame.truth.stars.len(),
            frame.truth.streaks.len(),
            frame.truth.targets.len(),
            frame.truth.targets.iter().filter(|t| t.on_image).count(),
            truth_path.display()
        );
    }
    Ok(vec![out_path.to_path_buf(), truth_path])
}

/// Writes the noiseless flat field as 32-bit floats beside the output, as
/// `<out>.flat.fits`.
fn flat_file(cfg: &Config, out_path: &Path) -> Result<PathBuf, String> {
    let cam = &cfg.profile.camera;
    let (w, h) = (cam.width, cam.height);
    let axis = cam
        .optical_axis_px
        .map_or(((w as f64 - 1.0) / 2.0, (h as f64 - 1.0) / 2.0), |a| {
            (a[0], a[1])
        });
    let flat = cfg.profile.flat.clone().unwrap_or_default();
    let (map, _) = skytwin_core::flat::flat_map(
        &flat,
        &cfg.profile.layout(),
        w,
        h,
        axis,
        cfg.profile.detector_seed,
    );
    let path = out_path.with_extension("flat.fits");
    let mut f = BufWriter::new(File::create(&path).map_err(path_err(&path))?);
    write_image::<_, &str, &str>(&mut f, w, h, &map, skytwin_core::fits::OutputType::F32, &[])
        .map_err(path_err(&path))?;
    println!("wrote {}", path.display());
    Ok(path)
}

/// SHA-256 of `bytes`, lowercase hex.
fn hex_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// Hashes of files this run wrote, taken from the bytes as they were
/// written (in the worker threads), so provenance need not read them back.
fn written_hashes() -> &'static Mutex<HashMap<PathBuf, String>> {
    static H: OnceLock<Mutex<HashMap<PathBuf, String>>> = OnceLock::new();
    H.get_or_init(Default::default)
}

/// Writes `bytes` to `path` and remembers their hash.
fn write_hashed(path: &Path, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(path, bytes).map_err(path_err(path))?;
    let h = hex_sha256(bytes);
    written_hashes()
        .lock()
        .expect("lock")
        .insert(path.to_path_buf(), h);
    Ok(())
}

/// SHA-256 of the file at `path`, hex: remembered if this run wrote it,
/// else read from disk; `None` if it cannot be read.
fn sha256(path: &Path) -> Option<String> {
    if let Some(h) = written_hashes().lock().expect("lock").get(path) {
        return Some(h.clone());
    }
    let bytes = std::fs::read(path).ok()?;
    Some(hex_sha256(&bytes))
}

/// `{ "path", "sha256" }` record of an input or output file.
fn file_record(path: &Path) -> serde_json::Value {
    serde_json::json!({ "path": path.display().to_string(), "sha256": sha256(path) })
}

/// Now as UTC, ISO 8601, from the system clock (Unix time, which skips leap
/// seconds, so a day is always 86 400 s; MJD 40 587 is 1970-01-01).
fn now_utc() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let days = d.as_secs() / 86_400;
    Utc {
        mjd_day: 40_587 + days as i64,
        sec: (d.as_secs() % 86_400) as f64 + f64::from(d.subsec_nanos()) * 1e-9,
    }
    .to_string()
}

/// A record of how a run was made, rewritten atomically as it goes.
struct Provenance {
    /// `<out>.provenance.json`.
    path: PathBuf,
    /// The record so far.
    doc: serde_json::Value,
}

impl Provenance {
    /// A record of a run that is starting: version, command line, start
    /// time, scene file and any `--seed`.
    fn start(cfg_path: &Path, out_path: &Path, seed: Option<u64>) -> Self {
        let doc = serde_json::json!({
            "status": "running",
            "skytwin_version": env!("CARGO_PKG_VERSION"),
            "command": std::env::args().collect::<Vec<_>>(),
            "started_utc": now_utc(),
            "scene_file": file_record(cfg_path),
            "seed_override": seed,
        });
        Provenance {
            path: out_path.with_extension("provenance.json"),
            doc,
        }
    }

    /// Records the loaded inputs: the file's seed, the profile, the
    /// instrument file the scene names (read from the raw scene JSON, if
    /// any), the TLE and catalogue files, and the catalogue the profile says
    /// it needs, to compare with the one used.
    fn note_inputs(&mut self, cfg_path: &Path, cfg: &Config) {
        self.doc["seed"] = serde_json::json!(cfg.seed);
        self.doc["profile_name"] = serde_json::json!(cfg.profile.name);
        let inst = std::fs::read_to_string(cfg_path)
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| {
                v.get("instrument")
                    .and_then(|i| i.as_str().map(str::to_string))
            });
        if let Some(p) = inst {
            self.doc["profile_file"] = file_record(Path::new(&p));
        }
        if let Some(tf) = &cfg.scene.target_file {
            self.doc["target_file"] = file_record(&tf.path);
        }
        if let Some(c) = &cfg.scene.catalog {
            self.doc["catalog"] = serde_json::to_value(c).unwrap_or_default();
            if let Some(csv) = &c.csv {
                self.doc["catalog"]["csv"]["sha256"] = serde_json::json!(sha256(&csv.path));
            }
        }
        if let Some(r) = &cfg.profile.catalog {
            self.doc["profile_catalog"] = serde_json::to_value(r).unwrap_or_default();
        }
    }

    /// Records the end of the run: the outputs and their hashes, or the
    /// error.
    fn finish(&mut self, result: &Result<Vec<PathBuf>, String>) {
        self.doc["finished_utc"] = serde_json::json!(now_utc());
        match result {
            Ok(files) => {
                self.doc["status"] = serde_json::json!("complete");
                self.doc["outputs"] = files.iter().map(|f| file_record(f)).collect();
            }
            Err(e) => {
                self.doc["status"] = serde_json::json!("failed");
                self.doc["error"] = serde_json::json!(e);
            }
        }
    }

    /// Writes the record through a temporary file and a rename, so readers
    /// never see half a file. On failure the temporary file is removed: the
    /// record never existed or is already lost, and a stale temporary file
    /// would only mislead.
    ///
    /// # Errors
    ///
    /// When the record cannot be written or moved into place.
    fn write(&self) -> Result<(), String> {
        let tmp = self.path.with_extension("provenance.json.tmp");
        let text = serde_json::to_string_pretty(&self.doc).map_err(|e| e.to_string())?;
        let done = std::fs::write(&tmp, text + "\n")
            .and_then(|()| std::fs::rename(&tmp, &self.path))
            .map_err(path_err(&self.path));
        if done.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        done
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("skytwin: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A folder of its own for each test, removed first.
    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("skytwin-cli-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// The quick start writes into `out/`, which a fresh checkout lacks.
    #[test]
    fn a_missing_output_folder_is_created() {
        let d = scratch("folder");
        let out = d.join("out/nested/frame.fits");
        make_output_folder(&out).unwrap();
        assert!(d.join("out/nested").is_dir());
        make_output_folder(Path::new("frame.fits")).unwrap();
    }

    /// A provenance record that can't be written is an error, and leaves no
    /// temporary file behind.
    #[test]
    fn a_failed_provenance_write_is_an_error() {
        let d = scratch("prov");
        let out = d.join("pv.fits");
        std::fs::create_dir_all(d.join("pv.provenance.json")).unwrap();
        let prov = Provenance::start(Path::new("scene.json"), &out, None);
        let err = prov.write().unwrap_err();
        assert!(err.contains("pv.provenance.json"), "{err}");
        assert!(!d.join("pv.provenance.json.tmp").exists());
    }

    /// A one-frame sequence runs as a sequence: it writes its index, and
    /// its conditions apply.
    #[test]
    fn a_one_frame_sequence_runs_as_a_sequence() {
        let d = scratch("seq");
        std::fs::create_dir_all(&d).unwrap();
        let cfg = skytwin_core::profile::config_from_value(
            &serde_json::json!({
                "camera": { "width": 16, "height": 16 },
                "scene": { "time": { "start_utc": "2025-03-01T00:00:00" } },
                "sequence": { "frames": 1, "cadence_s": 10.0 }
            }),
            "test",
        )
        .unwrap();
        let opts = Opts {
            quiet: true,
            ..Opts::default()
        };
        generate(&cfg, 1, &d.join("one.fits"), opts).unwrap();
        assert!(d.join("one.sequence.json").is_file());
        assert!(d.join("one_000.fits").is_file());
    }

    /// A pointing walk needs both RA and Dec, and a sequence needs a time.
    #[test]
    fn sequence_inputs_are_checked() {
        let cfg = |scene: serde_json::Value, seq: serde_json::Value| {
            skytwin_core::profile::config_from_value(
                &serde_json::json!({ "scene": scene, "sequence": seq }),
                "test",
            )
        };
        let time = serde_json::json!({ "start_utc": "2025-03-01T00:00:00" });
        let walk = serde_json::json!({
            "frames": 2, "cadence_s": 1.0,
            "pointing_walk": { "step_arcsec": 1.0, "roll_step_deg": 0.0 }
        });
        let ra_only = serde_json::json!({ "time": time, "pointing": { "ra_deg": 10.0 } });
        assert!(cfg(ra_only, walk).unwrap_err().contains("ra_deg/dec_deg"));
        let one = serde_json::json!({ "frames": 1, "cadence_s": 1.0 });
        assert!(cfg(serde_json::json!({}), one)
            .unwrap_err()
            .contains("scene.time"));
    }

    /// A scene's `null` removes a camera key the instrument sets.
    #[test]
    fn a_camera_key_set_to_null_is_removed() {
        let cfg = skytwin_core::profile::config_from_value(
            &serde_json::json!({
                "instrument": { "camera": { "width": 16, "height": 16, "optical_axis_px": [1.0, 2.0] } },
                "camera": { "optical_axis_px": null }
            }),
            "test",
        )
        .unwrap();
        assert_eq!(cfg.profile.camera.optical_axis_px, None);
        assert_eq!(cfg.profile.camera.width, 16);
    }

    /// Two batches named `visit.g.fits` and `visit.r.fits` get two indexes.
    #[test]
    fn batches_differing_before_the_extension_keep_separate_indexes() {
        let g = index_path(Path::new("out/visit.g.fits"), "batch");
        let r = index_path(Path::new("out/visit.r.fits"), "batch");
        assert_eq!(g, PathBuf::from("out/visit.g.batch.json"));
        assert_eq!(r, PathBuf::from("out/visit.r.batch.json"));
        assert_eq!(
            index_path(Path::new("night.fits"), "sequence"),
            PathBuf::from("night.sequence.json")
        );
    }
}
