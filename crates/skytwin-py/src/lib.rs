//! Python bindings for the `skytwin` simulator: render frames and their truth in memory,
//! one at a time or as a stream rendered ahead on worker threads.
//!
//! The Python package (`python/skytwin`) wraps this module; arrays cross as
//! little-endian bytes and become numpy arrays there.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Condvar, Mutex};

use pyo3::exceptions::{PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use skytwin_core::fits::OutputType;
use skytwin_core::{config_from_value, load_config, Generator};

/// One rendered frame, ready to cross to Python.
struct Packed {
    /// Frame index in the batch.
    index: u64,
    /// The frame's seed.
    seed: u64,
    width: usize,
    height: usize,
    /// numpy dtype string of `image`.
    dtype: &'static str,
    /// Pixels, row-major, little-endian.
    image: Vec<u8>,
    /// Truth record as JSON.
    truth: String,
    /// Expected electrons per pixel (noiseless), little-endian `f32`, if asked.
    expected: Option<Vec<u8>>,
}

/// Converts each value with `f` and concatenates the bytes.
fn le_bytes<const N: usize>(values: &[f64], f: impl Fn(f64) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|&v| f(v)).collect()
}

/// Renders frame `k` and packs it as bytes, in the frame's output type.
fn pack(gen: &Generator, k: u64, expected: bool) -> Result<Packed, String> {
    let f = gen.render(k)?;
    let (dtype, image) = match f.output {
        OutputType::U16 => (
            "<u2",
            le_bytes(&f.adu, |v| (v.round() as u16).to_le_bytes()),
        ),
        OutputType::I16 => (
            "<i2",
            le_bytes(&f.adu, |v| (v.round() as i16).to_le_bytes()),
        ),
        OutputType::I32 => (
            "<i4",
            le_bytes(&f.adu, |v| (v.round() as i32).to_le_bytes()),
        ),
        OutputType::F32 => ("<f4", le_bytes(&f.adu, |v| (v as f32).to_le_bytes())),
    };
    let truth = serde_json::to_string(&f.truth).map_err(|e| e.to_string())?;
    let expected = expected.then(|| le_bytes(&f.expected_e, |v| (v as f32).to_le_bytes()));
    Ok(Packed {
        index: k,
        seed: gen.seed_of(k),
        width: f.width,
        height: f.height,
        dtype,
        image,
        truth,
        expected,
    })
}

/// A frame as Python receives it: index, seed, width, height, dtype, image
/// bytes, truth JSON and optional expected-electron bytes.
type Out<'py> = (
    u64,
    u64,
    usize,
    usize,
    &'static str,
    Bound<'py, PyBytes>,
    String,
    Option<Bound<'py, PyBytes>>,
);

/// Converts a packed frame to its Python tuple.
fn to_py(py: Python<'_>, p: Packed) -> Out<'_> {
    (
        p.index,
        p.seed,
        p.width,
        p.height,
        p.dtype,
        PyBytes::new(py, &p.image),
        p.truth,
        p.expected.map(|e| PyBytes::new(py, &e)),
    )
}

/// A loaded scene file: renders its frames.
#[pyclass(frozen, module = "skytwin._native")]
struct Native {
    gen: Arc<Generator>,
}

#[pymethods]
impl Native {
    /// Load a scene file.
    #[staticmethod]
    #[pyo3(signature = (path, seed=None))]
    fn from_path(py: Python<'_>, path: &str, seed: Option<u64>) -> PyResult<Self> {
        let gen = py
            .detach(|| {
                load_config(std::path::Path::new(path)).and_then(|c| Generator::new(c, seed))
            })
            .map_err(PyValueError::new_err)?;
        Ok(Native { gen: Arc::new(gen) })
    }

    /// Load a scene given as JSON text.
    #[staticmethod]
    #[pyo3(signature = (text, seed=None))]
    fn from_json(py: Python<'_>, text: &str, seed: Option<u64>) -> PyResult<Self> {
        let gen = py
            .detach(|| {
                let v = serde_json::from_str(text).map_err(|e| format!("scene: {e}"))?;
                config_from_value(&v, "scene").and_then(|c| Generator::new(c, seed))
            })
            .map_err(PyValueError::new_err)?;
        Ok(Native { gen: Arc::new(gen) })
    }

    /// Frames in the batch (1 without a batch block).
    fn frames(&self) -> usize {
        self.gen.frames()
    }

    /// Render frame `index` (the GIL is released while it renders).
    #[pyo3(signature = (index, expected=false))]
    fn render<'py>(&self, py: Python<'py>, index: u64, expected: bool) -> PyResult<Out<'py>> {
        let gen = self.gen.clone();
        let p = py
            .detach(|| pack(&gen, index, expected))
            .map_err(PyValueError::new_err)?;
        Ok(to_py(py, p))
    }

    /// Frames `start..start + count`, rendered ahead on `workers` threads
    /// (at most `prefetch` waiting), returned in order.
    #[pyo3(signature = (start, count, workers, prefetch, expected=false))]
    fn stream(
        &self,
        start: u64,
        count: u64,
        workers: usize,
        prefetch: usize,
        expected: bool,
    ) -> Stream {
        Stream::start(
            &self.gen,
            start,
            count,
            workers.max(1),
            prefetch.max(1),
            expected,
        )
    }
}

/// A frame index and its result.
type Item = (u64, Result<Packed, String>);

/// Frames rendered ahead on worker threads, handed out in order.
#[pyclass(module = "skytwin._native")]
struct Stream {
    /// Results from the workers; `None` once closed.
    rx: Option<Arc<Mutex<Receiver<Item>>>>,
    /// Frames that arrived ahead of their turn.
    pending: BTreeMap<u64, Result<Packed, String>>,
    /// Next index to hand out.
    next: u64,
    /// One past the last index.
    end: u64,
    /// Tells the workers to stop.
    stop: Arc<AtomicBool>,
    /// The next index to hand out, shared with the workers so none renders
    /// more than a window ahead of it.
    handed: Arc<(Mutex<u64>, Condvar)>,
}

impl Stream {
    /// Starts `workers` threads (no more than `count`) rendering frames
    /// `start..start + count` into a channel holding at most `prefetch`.
    ///
    /// A worker takes frame `k` only once `k` is less than `prefetch +
    /// workers` ahead of the next frame handed out, so frames that finish
    /// out of order wait in a bounded set however slow one frame is. A
    /// worker stops at the end, when told to, or when its send fails: a
    /// closed channel means the stream was dropped.
    fn start(
        gen: &Arc<Generator>,
        start: u64,
        count: u64,
        workers: usize,
        prefetch: usize,
        expected: bool,
    ) -> Self {
        let end = start.saturating_add(count);
        let (tx, rx) = sync_channel(prefetch);
        let counter = Arc::new(AtomicU64::new(start));
        let stop = Arc::new(AtomicBool::new(false));
        let handed = Arc::new((Mutex::new(start), Condvar::new()));
        let workers = workers.min(count.max(1) as usize);
        let window = (prefetch + workers) as u64;
        for _ in 0..workers {
            let (gen, tx, counter, stop, handed) = (
                Arc::clone(gen),
                tx.clone(),
                counter.clone(),
                stop.clone(),
                handed.clone(),
            );
            std::thread::spawn(move || loop {
                let k = counter.fetch_add(1, Ordering::Relaxed);
                if k >= end || stop.load(Ordering::Relaxed) {
                    break;
                }
                {
                    let (lock, cv) = &*handed;
                    let mut next = lock.lock().expect("lock");
                    while k >= *next + window && !stop.load(Ordering::Relaxed) {
                        next = cv.wait(next).expect("lock");
                    }
                }
                if stop.load(Ordering::Relaxed) || tx.send((k, pack(&gen, k, expected))).is_err() {
                    break;
                }
            });
        }
        Stream {
            rx: Some(Arc::new(Mutex::new(rx))),
            pending: BTreeMap::new(),
            next: start,
            end,
            stop,
            handed,
        }
    }
}

#[pymethods]
impl Stream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// The next frame in index order. Waits for the workers without the
    /// GIL, so the caller's other Python threads keep running.
    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Out<'py>> {
        if self.next >= self.end {
            return Err(PyStopIteration::new_err(()));
        }
        let want = self.next;
        let item = loop {
            if let Some(r) = self.pending.remove(&want) {
                break r;
            }
            let rx = self
                .rx
                .clone()
                .ok_or_else(|| PyStopIteration::new_err(()))?;
            match py.detach(move || rx.lock().expect("lock").recv()) {
                Ok((k, r)) => {
                    self.pending.insert(k, r);
                }
                Err(_) => return Err(PyValueError::new_err("stream workers stopped")),
            }
        };
        self.next += 1;
        {
            let (lock, cv) = &*self.handed;
            *lock.lock().expect("lock") = self.next;
            cv.notify_all();
        }
        match item {
            Ok(p) => Ok(to_py(py, p)),
            Err(e) => {
                self.close();
                Err(PyValueError::new_err(format!("frame {want}: {e}")))
            }
        }
    }

    /// Stop the workers (also done when the stream is dropped).
    fn close(&mut self) {
        self.halt();
        self.rx = None;
        self.pending.clear();
        self.next = self.end;
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        self.halt();
    }
}

impl Stream {
    /// Tells the workers to stop, waking any waiting for their turn. The
    /// flag is set under the lock so no worker misses the wake-up.
    fn halt(&self) {
        let (lock, cv) = &*self.handed;
        let _guard = lock.lock().expect("lock");
        self.stop.store(true, Ordering::Relaxed);
        cv.notify_all();
    }
}

/// The native module.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Native>()?;
    m.add_class::<Stream>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
