//! Foundational types shared by every `alphax` crate.
//!
//! Three things in here are load-bearing for the whole engine, in the sense
//! that getting them wrong invalidates every downstream number:
//!
//! - [`math`] — the float tolerance layer. Every fill decision in
//!   [`alphax-exec`](../alphax_exec/index.html) routes through it. Plain `==` / `<`
//!   on monetary floats is denied by a workspace lint.
//! - [`flex`] — zero-allocation broadcasting. Lets a scalar, a per-row vector,
//!   a per-column vector and a full matrix all be read through one call, with
//!   no materialization. This is what makes thousand-combination parameter
//!   sweeps fit in memory.
//! - [`columns`] — multi-level column labels. Parameters, symbols and factors
//!   all collapse into the column axis, so labels are the *only* way to
//!   attribute a result back to what produced it.
//!
//! [`columns`] and [`flex`] explain why the column axis is laid out the way it
//! is.

pub mod columns;
pub mod error;
pub mod flex;
pub mod math;
pub mod matrix;
pub mod random;

pub use error::{Error, Result};
