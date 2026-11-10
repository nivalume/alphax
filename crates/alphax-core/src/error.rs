//! Shared error type.
//!
//! Note the distinction this type draws, which runs through the whole engine:
//! an [`Error`] is an *invariant violation* — the caller handed us something
//! impossible (mismatched shapes, a level that does not exist, negative
//! cash). An order that simply cannot be filled is **not** an error; it is a
//! rejected order carrying a status code, returned through the normal path.
//!
//! Written by hand rather than derived, to keep the foundation crate free of
//! external dependencies.

use std::fmt;

/// Result alias used across the workspace.
pub type Result<T> = std::result::Result<T, Error>;

/// An invariant violation. Never used for rejected orders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Two arrays that had to broadcast together could not.
    Broadcast {
        /// Shape of the left operand.
        left: (usize, usize),
        /// Shape of the right operand.
        right: (usize, usize),
    },

    /// A flexible array's length did not match the axis it was declared against.
    FlexLen {
        /// Which axis the array was declared against.
        kind: &'static str,
        /// Length the caller supplied.
        actual: usize,
        /// Length the axis requires.
        expected: usize,
    },

    /// A column-label level was referenced by a name that does not exist.
    UnknownLevel {
        /// The name that was looked up.
        name: String,
        /// Comma-separated list of level names that do exist.
        available: String,
    },

    /// Column-label levels disagreed on how many columns they describe.
    LevelWidth {
        /// The offending level.
        name: String,
        /// Number of labels it carries.
        actual: usize,
        /// Number of columns the label set describes.
        expected: usize,
    },

    /// A numeric invariant the simulator relies on was violated.
    Invariant(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Broadcast { left, right } => {
                write!(f, "cannot broadcast {left:?} against {right:?}")
            }
            Self::FlexLen {
                kind,
                actual,
                expected,
            } => write!(f, "{kind} array has length {actual}, expected {expected}"),
            Self::UnknownLevel { name, available } => {
                write!(f, "no column level named {name:?} (have: {available})")
            }
            Self::LevelWidth {
                name,
                actual,
                expected,
            } => write!(
                f,
                "level {name:?} describes {actual} columns, expected {expected}"
            ),
            Self::Invariant(msg) => write!(f, "invariant violated: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_the_offending_values() {
        let e = Error::FlexLen {
            kind: "per-column",
            actual: 3,
            expected: 4,
        };
        let msg = e.to_string();
        assert!(msg.contains("per-column"), "{msg}");
        assert!(msg.contains('3'), "{msg}");
        assert!(msg.contains('4'), "{msg}");
    }

    #[test]
    fn unknown_level_lists_the_alternatives() {
        let e = Error::UnknownLevel {
            name: "windwo".to_owned(),
            available: "window, symbol".to_owned(),
        };
        let msg = e.to_string();
        assert!(msg.contains("windwo"), "{msg}");
        assert!(msg.contains("window, symbol"), "{msg}");
    }
}
