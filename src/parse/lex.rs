//! Tokenizing LC-3 assembly.
//! 
//! This module holds the tokens that characterize LC-3 assembly ([`Token`]).
//! This module is used by the parser to facilitate the conversion of
//! assembly source code into an AST.
//! 
//! The module's key data structure is the [`Token`] enum,
//! which lists all of the tokens of LC-3 assembly.

use std::num::IntErrorKind;

use logos::{Lexer, Logos};

/// A unit of information in LC3 source code.
#[derive(Debug, Logos, PartialEq, Eq)]
#[logos(skip r"[ \t]+", error = LexErr)]
pub enum Token {
    // Note, these regexes span over tokens that are technically invalid 
    // (e.g., 23trst matches for unsigned even though it shouldn't).
    // This is intended.
    // These regexes collect what would be considered one discernable unit
    // and validates it using the validator function.

    /// An unsigned numeric value (e.g., `9`, `#14`, `x7F`, etc.)
    #[regex(r"\d\w*", lex_unsigned_dec)]
    #[regex(r"#\d?\w*", lex_unsigned_dec)]
    #[regex(r"[Xx][\dA-Fa-f]\w*", lex_unsigned_hex)]
    Unsigned(u16),

    /// A signed numeric value (e.g., `-9`, `#-14`, `x-7F`, etc.)
    #[regex(r"-\w*", lex_signed_dec)]
    #[regex(r"#-\w*", lex_signed_dec)]
    #[regex(r"[Xx]-\w*", lex_signed_hex)]
    Signed(i16),

    /// A register value (i.e., `R0`-`R7`)
    #[regex(r"[Rr]\d+", lex_reg)]
    Reg(u8),

    /// An identifier.
    /// 
    /// This can refer to either:
    /// - a label (e.g., `IF`, `WHILE`, `ENDIF`, `IF1`)
    /// - an instruction (e.g. `ADD`, `AND`, `NOT`)
    ///
    /// This token type is case-insensitive. 
    #[regex(r"[A-Za-z_]\w*", |lx| lx.slice().parse::<Ident>().expect("should be infallible"))]
    Ident(Ident),

    /// A directive (e.g., `.orig`, `.end`).
    #[regex(r"\.[A-Za-z_]\w*", |lx| lx.slice()[1..].to_string())]
    Directive(String),

    /// A string literal (e.g., `"Hello!"`)
    #[token(r#"""#, lex_str_literal)]
    String(String),

    /// A colon, which can optionally appear after labels
    #[token(":")]
    Colon,

    /// A comma, which delineate operands of an instruction
    #[token(",")]
    Comma,

    /// A comment, which starts with a semicolon and spans the remaining part of the line.
    #[regex(r";.*")]
    Comment,

    /// A new line
    #[regex(r"\r?\n")]
    NewLine
}

macro_rules! ident_enum {
    ($($instr:ident),+) => {
        /// An identifier. 
        /// 
        /// This can refer to either:
        /// - a label (e.g., `IF`, `WHILE`, `ENDIF`, `IF1`)
        /// - an instruction (e.g. `ADD`, `AND`, `NOT`)
        ///
        /// This token type is case insensitive. 
        #[derive(Debug, PartialEq, Eq, Clone)]
        pub enum Ident {
            $(
                #[allow(missing_docs)]
                $instr
            ),+,
            #[allow(missing_docs)]
            Label(String)
        }

        impl std::str::FromStr for Ident {
            type Err = std::convert::Infallible;
        
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match &*s.to_uppercase() {
                    $(stringify!($instr) => Ok(Self::$instr)),*,
                    _ => Ok(Self::Label(s.to_string()))
                }
            }
        }

        impl std::fmt::Display for Ident {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(Self::$instr => f.write_str(stringify!($instr))),*,
                    Self::Label(id) => f.write_str(id)
                }
            }
        }
    };
}
ident_enum! {
    ADD, AND, NOT, BR, BRP, BRZ, BRZP, BRN, BRNP, BRNZ, BRNZP, 
    JMP, JSR, JSRR, LD, LDI, LDR, LEA, ST, STI, STR, TRAP, NOP, 
    RET, RTI, GETC, OUT, PUTC, PUTS, IN, PUTSP, HALT
}

/// Any errors raised in attempting to tokenize an input stream.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum LexErr {
    /// Numeric literal (unsigned dec, hex, and bin) cannot fit within the range of a u16
    DoesNotFitU16,
    /// Numeric literal (signed dec) cannot fit within the range of a i16
    DoesNotFitI16,
    /// Hex literal (starting with x) has invalid hex digits
    InvalidHex,
    /// Numeric literal could not be parsed as a decimal literal because it has invalid digits (i.e., not 0-9)
    InvalidNumeric,
    /// Hex literal (starting with x) doesn't have digits after it.
    InvalidHexEmpty,
    /// Numeric literal could not be parsed as a decimal literal because there are no digits in it (it's just # or #-)
    InvalidDecEmpty,
    /// Int parsing failed but the reason why is unknown
    UnknownIntErr,
    /// String literal is missing an end quotation mark.
    UnclosedStrLit,
    /// String literal is missing an end quotation mark.
    StrLitTooBig,
    /// Token had the format R\d, but \d isn't 0-7.
    InvalidReg,
    /// A symbol was used which is not allowed in LC3 assembly files
    #[default]
    InvalidSymbol
}
impl std::fmt::Display for LexErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LexErr::DoesNotFitU16   => f.write_str("numeric token does not fit 16-bit unsigned integer"),
            LexErr::DoesNotFitI16   => f.write_str("numeric token does not fit 16-bit signed integer"),
            LexErr::InvalidHex      => f.write_str("invalid hex literal"),
            LexErr::InvalidNumeric  => f.write_str("invalid decimal literal"),
            LexErr::InvalidHexEmpty => f.write_str("invalid hex literal"),
            LexErr::InvalidDecEmpty => f.write_str("invalid decimal literal"),
            LexErr::UnknownIntErr   => f.write_str("could not parse integer"),
            LexErr::UnclosedStrLit  => f.write_str("unclosed string literal"),
            LexErr::StrLitTooBig    => f.write_str("string literal is too large"),
            LexErr::InvalidReg      => f.write_str("invalid register"),
            LexErr::InvalidSymbol   => f.write_str("unrecognized symbol"),
        }
    }
}
impl std::error::Error for LexErr {}
impl crate::err::Error for LexErr {
    fn help(&self) -> Option<std::borrow::Cow<str>> {
        match self {
            LexErr::DoesNotFitU16    => Some(format!("the range for a 16-bit unsigned integer is [{}, {}]", u16::MIN, u16::MAX).into()),
            LexErr::DoesNotFitI16    => Some(format!("the range for a 16-bit signed integer is [{}, {}]", i16::MIN, i16::MAX).into()),
            LexErr::InvalidHex       => Some("a hex literal starts with 'x' and consists of 0-9, A-F".into()),
            LexErr::InvalidNumeric   => Some("a decimal literal only consists of digits 0-9".into()),
            LexErr::InvalidHexEmpty  => Some("there should be hex digits (0-9, A-F) here".into()),
            LexErr::InvalidDecEmpty  => Some("there should be digits (0-9) here".into()),
            LexErr::UnknownIntErr    => None,
            LexErr::UnclosedStrLit   => Some("add a quote to the end of the string literal".into()),
            LexErr::StrLitTooBig     => Some(format!("string literals are limited to at most {} characters", u16::MAX - 1).into()),
            LexErr::InvalidReg       => Some("this must be R0-R7".into()),
            LexErr::InvalidSymbol    => Some("this char does not occur in any token in LC-3 assembly".into()),
        }
    }
}
/// Helper that converts an int error kind to its corresponding LexErr, based on the provided inputs.
fn convert_int_error(
    e: &std::num::IntErrorKind, 
    invalid_digits_err: LexErr, 
    empty_err: LexErr, 
    overflow_err: LexErr, 
    src: &str
) -> LexErr {
    match e {
        IntErrorKind::Empty        => empty_err,
        IntErrorKind::InvalidDigit if src == "-" => empty_err,
        IntErrorKind::InvalidDigit => invalid_digits_err,
        IntErrorKind::PosOverflow  => overflow_err,
        IntErrorKind::NegOverflow  => overflow_err,
        IntErrorKind::Zero         => unreachable!("IntErrorKind::Zero should not be emitted in parsing u16"),
        _ => LexErr::UnknownIntErr,
    }
}
fn lex_unsigned_dec(lx: &Lexer<'_, Token>) -> Result<u16, LexErr> {
    let mut string = lx.slice();
    if lx.slice().starts_with('#') {
        string = &string[1..];
    }

    string.parse::<u16>()
        .map_err(|e| convert_int_error(e.kind(), LexErr::InvalidNumeric, LexErr::InvalidDecEmpty, LexErr::DoesNotFitU16, string))
}

fn lex_signed_dec(lx: &Lexer<'_, Token>) -> Result<i16, LexErr> {
    let mut string = lx.slice();
    if lx.slice().starts_with('#') {
        string = &string[1..];
    }

    string.parse::<i16>()
        .map_err(|e| convert_int_error(e.kind(), LexErr::InvalidNumeric, LexErr::InvalidDecEmpty, LexErr::DoesNotFitI16, string))
}
fn lex_unsigned_hex(lx: &Lexer<'_, Token>) -> Result<u16, LexErr> {
    let Some(hex) = lx.slice().strip_prefix(['X', 'x']) else {
        unreachable!("Lexer slice should have contained an X or x");
    };

    u16::from_str_radix(hex, 16)
        .map_err(|e| convert_int_error(e.kind(), LexErr::InvalidHex, LexErr::InvalidHexEmpty, LexErr::DoesNotFitU16, hex))
}
fn lex_signed_hex(lx: &Lexer<'_, Token>) -> Result<i16, LexErr> {
    let Some(hex) = lx.slice().strip_prefix(['X', 'x']) else {
        unreachable!("Lexer slice should have contained an X or x");
    };

    i16::from_str_radix(hex, 16)
        .map_err(|e| convert_int_error(e.kind(), LexErr::InvalidHex, LexErr::InvalidHexEmpty, LexErr::DoesNotFitI16, hex))
}
fn lex_reg(lx: &Lexer<'_, Token>) -> Result<u8, LexErr> {
    lx.slice()[1..].parse::<u8>().ok()
        .filter(|&r| r < 8)
        .ok_or(LexErr::InvalidReg)
}
fn lex_str_literal(lx: &mut Lexer<'_, Token>) -> Result<String, LexErr> {
    let rem = lx.remainder()
        .lines()
        .next()
        .unwrap_or("");

    // calculate the length of the string literal ignoring the quotes
    // consume tokens up to the end of the literal and including the unescaped quote
    let mlen = rem.match_indices('"')
        .map(|(n, _)| n)
        .find(|&n| !matches!(rem.get((n - 1)..(n + 1)), Some("\\\"")));
    
    match mlen {
        Some(len) => lx.bump(len + 1),
        None => {
            lx.bump(rem.len());
            return Err(LexErr::UnclosedStrLit);
        }
    }

    // get the string inside quotes:
    let mut remaining = &lx.slice()[1..(lx.slice().len() - 1)];
    let mut buf = String::with_capacity(remaining.len());

    // Look for escapes. Only a simple group of escapes are implemented.
    // (e.g., `\n`, `\r`, etc.)
    while let Some((left, right)) = remaining.split_once('\\') {
        buf.push_str(left);

        // this character is part of the escape:
        let esc = right.as_bytes()
            .first()
            .unwrap_or_else(|| unreachable!("expected character after escape")); // there always has to be one, cause last character is not \
        match esc {
            b'n'  => buf.push('\n'),
            b'r'  => buf.push('\r'),
            b't'  => buf.push('\t'),
            b'\\' => buf.push('\\'),
            b'0'  => buf.push('\0'),
            b'"'  => buf.push('\"'),
            &c => {
                buf.push('\\');
                buf.push(char::from(c));
            }
        }
        
        remaining = &right[1..];
    }
    buf.push_str(remaining);
    
    match buf.len() < usize::from(u16::MAX) {
        true  => Ok(buf),
        false => Err(LexErr::StrLitTooBig),
    }
}