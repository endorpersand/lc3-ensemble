//! Error interface for this crate.
//! 
//! This module creates an [`Error`] trait implemented by error types in this crate.
//! 
//! The error trait extends the standard library's [`std::error::Error`] type by
//! adding a `help` method (to help assist the user in fixing the issue),
//! and a `span` method (to point to where the error occurs).

use std::borrow::Cow;

use logos::Span;

pub use crate::asm::AsmErr;
pub use crate::ast::OffsetNewErr;
pub use crate::parse::lex::LexErr;
pub use crate::parse::ParseErr;
pub use crate::sim::SimErr;

/// Unified error interface for all errors in this crate.
/// 
/// Note that the [`Display`] implementation is used for a brief message,
/// where as [`Error::help`] is used for any clarifying messages.
/// 
/// [`Display`]: std::fmt::Display
pub trait Error: std::error::Error {
    /// The range where this error occurs in source.
    /// 
    /// If this is not known, this can be set to `None`.
    fn span(&self) -> Option<crate::err::ErrSpan> {
        None
    }

    /// A clarifying message to help aid someone in how to fix the message.
    /// 
    /// If there is none to add, this can be set to `None`.
    fn help(&self) -> Option<Cow<str>>;
}

/// The possible source ranges for an error. 
/// 
/// This can be:
/// - one contiguous span,
/// - two contiguous spans, or
/// - three or more contiguous spans
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ErrSpan {
    /// One contiguous span.
    One(Span),
    /// Two contiguous spans.
    Two([Span; 2]),
    /// Three or more contiguous spans.
    /// 
    /// This should always have at least 3 elements.
    Many(Vec<Span>)
}
impl ErrSpan {
    /// Gets the first span.
    pub fn first(&self) -> Span {
        match self {
            ErrSpan::One(r)      => r.clone(),
            ErrSpan::Two([r, _]) => r.clone(),
            ErrSpan::Many(r)     => r.first().unwrap().clone(),
        }
    }

    /// Gets an iterator over all of the spans.
    pub fn iter(&self) -> impl Iterator<Item=&Span> {
        match self {
            ErrSpan::One(r)  => std::slice::from_ref(r).iter(),
            ErrSpan::Two(r)  => r.iter(),
            ErrSpan::Many(r) => r.iter(),
        }
    }
}
impl Extend<Span> for ErrSpan {
    fn extend<T: IntoIterator<Item = Span>>(&mut self, iter: T) {
        let mut iter = iter.into_iter();

        // It's 4:30AM and I thought it'd be funny.
        loop {
            match self {
                ErrSpan::One(r0) => {
                    let Some(r1) = iter.next() else { return };
                    let r0 = std::mem::replace(r0, 0..0);
                    *self = ErrSpan::Two([r0, r1]);
                },
                ErrSpan::Two([r0, r1]) => {
                    let Some(r2) = iter.next() else { return };
                    let r0 = std::mem::replace(r0, 0..0);
                    let r1 = std::mem::replace(r1, 0..0);
                    *self = ErrSpan::Many(vec![r0, r1, r2]);
                },
                ErrSpan::Many(mr) => {
                    mr.extend(iter);
                    return;
                },
            }
        }
    }
}
impl From<Span> for ErrSpan {
    fn from(value: Span) -> Self {
        ErrSpan::One(value)
    }
}
impl From<[Span; 2]> for ErrSpan {
    fn from(value: [Span; 2]) -> Self {
        ErrSpan::Two(value)
    }
}
impl From<Vec<Span>> for ErrSpan {
    fn from(value: Vec<Span>) -> Self {
        match Box::<[_; 1]>::try_from(value) {
            Ok(rbox) => {
                let [r] = *rbox;
                ErrSpan::One(r)
            },
            Err(value) => match Box::try_from(value) {
                Ok(rbox) => ErrSpan::Two(*rbox),
                Err(value) => ErrSpan::Many(value)
            }
        }
    }
}
impl From<Vec<ErrSpan>> for ErrSpan {
    fn from(value: Vec<ErrSpan>) -> Self {
        let mr: Vec<_> = value.into_iter()
            .flat_map(|r| match r {
                ErrSpan::One(r) => vec![r],
                ErrSpan::Two(r) => r.to_vec(),
                ErrSpan::Many(r) => r,
            })
            .collect();

        ErrSpan::from(mr)
    }
}
impl std::fmt::Debug for ErrSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::One(r)   => r.fmt(f),
            Self::Two(r)   => r.fmt(f),
            Self::Many(mr) => mr.fmt(f),
        }
    }
}