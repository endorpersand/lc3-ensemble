//! Assembling assembly source ASTs into object files.
//! 
//! This module is used to convert source ASTs (`Vec<`[`Stmt`]`>`) into object files 
//! that can be executed by the simulator.
//! 
//! The assembler module notably consists of:
//! - [`assemble`] and [`assemble_debug`]: The main functions which assemble the statements into an object file.
//! - [`SymbolTable`]: a struct holding the symbol table, which stores location information for labels after the first assembler pass
//! - [`ObjectFile`]: a struct holding the object file, which can be loaded into the simulator and executed
//! 
//! [`Stmt`]: crate::ast::asm::Stmt

pub mod encoding;

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use logos::Span;

use crate::ast::asm::{AsmInstr, Directive, Stmt, StmtKind};
use crate::ast::sim::SimInstr;
use crate::ast::{IOffset, ImmOrReg, Offset, OffsetNewErr, PCOffset, Reg};
use crate::err::ErrSpan;

/// Assembles a assembly source code AST into an object file.
/// 
/// This function assembles the source AST *without* including debug symbols
/// in the object file.
/// See [`SymbolTable`] for more details about debug symbols.
/// 
/// # Example
/// ```
/// use lc3_ensemble::parse::parse_ast;
/// use lc3_ensemble::asm::assemble;
/// 
/// let src = "
///     .orig x3000
///     LABEL: HALT
///     .end
/// ";
/// let ast = parse_ast(src).unwrap();
/// 
/// let obj_file = assemble(ast);
/// assert!(obj_file.is_ok());
/// 
/// // Symbol table doesn't exist in object file:
/// let obj_file = obj_file.unwrap();
/// assert!(obj_file.symbol_table().is_none());
/// ```
pub fn assemble(ast: Vec<Stmt>) -> Result<ObjectFile, AsmErr> {
    let sym = SymbolTable::new(&ast, None)?;
    ObjectFile::new(ast, sym, false)
}
/// Assembles a assembly source code AST into an object file.
/// 
/// This function assembles the source AST *and* includes debug symbols
/// in the object file.
/// See [`SymbolTable`] for more details about debug symbols.
/// 
/// # Example
/// ```
/// use lc3_ensemble::parse::parse_ast;
/// use lc3_ensemble::asm::assemble_debug;
/// 
/// let src = "
///     .orig x3000
///     LABEL: HALT
///     .end
/// ";
/// let ast = parse_ast(src).unwrap();
/// 
/// let obj_file = assemble_debug(ast, src);
/// assert!(obj_file.is_ok());
/// 
/// // Symbol table does exist in object file:
/// let obj_file = obj_file.unwrap();
/// assert!(obj_file.symbol_table().is_some());
/// ```
pub fn assemble_debug(ast: Vec<Stmt>, src: &str) -> Result<ObjectFile, AsmErr> {
    let sym = SymbolTable::new(&ast, Some(src))?;
    ObjectFile::new(ast, sym, true)
}

/// Kinds of errors that can occur from assembling given assembly code.
/// 
/// See [`AsmErr`] for this error type with span information included.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum AsmErrKind {
    /// Cannot determine address of label (pass 1).
    UndetAddrLabel,
    /// Cannot determine address of instruction (pass 2).
    UndetAddrStmt,
    /// There was an `.orig` but no corresponding `.end` (pass 1).
    UnclosedOrig,
    /// There was an `.end` but no corresonding `.orig` (pass 1).
    UnopenedOrig,
    /// There was an `.orig` opened after another `.orig` (pass 1).
    OverlappingOrig,
    /// There were multiple labels of the same name (pass 1).
    OverlappingLabels,
    /// Block wraps memory (pass 2).
    WrappingBlock,
    /// Block writes to IO memory region (pass 2).
    BlockInIO,
    /// There are blocks that overlap ranges of memory (pass 2).
    OverlappingBlocks,
    /// Creating the offset to replace a label caused overflow (pass 2).
    OffsetNewErr(OffsetNewErr),
    /// Cannot find the offset with an external label (pass 2).
    OffsetExternal,
    /// Label did not have an assigned address (pass 2).
    CouldNotFindLabel,
}
impl std::fmt::Display for AsmErrKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UndetAddrLabel    => f.write_str("cannot determine address of label"),
            Self::UndetAddrStmt     => f.write_str("cannot determine address of statement"),
            Self::UnclosedOrig      => f.write_str(".orig directive was never closed"),
            Self::UnopenedOrig      => f.write_str(".end does not have associated .orig"),
            Self::OverlappingOrig   => f.write_str("cannot have an .orig inside another region"),
            Self::OverlappingLabels => f.write_str("label was defined multiple times"),
            Self::WrappingBlock     => f.write_str("block wraps around in memory"),
            Self::BlockInIO         => f.write_str("cannot write code into memory-mapped IO region"),
            Self::OverlappingBlocks => f.write_str("regions overlap in memory"),
            Self::OffsetNewErr(e)   => e.fmt(f),
            Self::OffsetExternal    => f.write_str("cannot use external label here"),
            Self::CouldNotFindLabel => f.write_str("label could not be found"),
        }
    }
}

/// Error from assembling given assembly code.
#[derive(Debug)]
pub struct AsmErr {
    /// The value with a span.
    pub kind: AsmErrKind,
    /// The span in the source associated with this value.
    pub span: ErrSpan
}
impl AsmErr {
    /// Creates a new [`AsmErr`].
    pub fn new<E: Into<ErrSpan>>(kind: AsmErrKind, span: E) -> Self {
        AsmErr { kind, span: span.into() }
    }
}
impl std::fmt::Display for AsmErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.kind.fmt(f)
    }
}
impl std::error::Error for AsmErr {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            AsmErrKind::OffsetNewErr(e) => Some(e),
            _ => None
        }
    }
}
impl crate::err::Error for AsmErr {
    fn span(&self) -> Option<crate::err::ErrSpan> {
        Some(self.span.clone())
    }

    fn help(&self) -> Option<std::borrow::Cow<str>> {
        match &self.kind {
            AsmErrKind::UndetAddrLabel    => Some("try moving this label inside of an .orig/.end block".into()),
            AsmErrKind::UndetAddrStmt     => Some("try moving this statement inside of an .orig/.end block".into()),
            AsmErrKind::UnclosedOrig      => Some("try adding an .end directive at the end of this block".into()),
            AsmErrKind::UnopenedOrig      => Some("try adding an .orig directive at the beginning of this block".into()),
            AsmErrKind::OverlappingOrig   => Some("try adding an .end directive at the end of the outer .orig block".into()),
            AsmErrKind::OverlappingLabels => Some("labels must be unique within a file, try renaming one of the labels".into()),
            AsmErrKind::OverlappingBlocks => Some("try moving the starting address of one of these regions".into()),
            AsmErrKind::WrappingBlock     => Some("user code typically starts at x3000 and is short enough to not wrap memory".into()),
            AsmErrKind::BlockInIO         => Some("try not doing that".into()),
            AsmErrKind::OffsetNewErr(e)   => e.help(),
            AsmErrKind::OffsetExternal    => Some("external labels cannot be an offset operand; try creating a .fill LABEL directive".into()),
            AsmErrKind::CouldNotFindLabel => Some("try adding this label before an instruction or directive".into()),
        }
    }
}

const IO_START: u16 = 0xFE00;

/// A mapping from line numbers to memory addresses (and vice-versa).
///
/// This is implemented as a sorted list of contiguous blocks, consisting of:
/// - The first source line number of the block, and
/// - The memory addresses of the block
/// 
/// For example,
/// ```text
/// 0 | .orig x3000
/// 1 |     AND R0, R0, #0
/// 2 |     ADD R0, R0, #5
/// 3 |     HALT
/// 4 | .end
/// 5 | 
/// 6 | .orig x4000
/// 7 |     .blkw 5
/// 8 |     .fill x9F9F
/// 9 | .end
/// ```
/// maps to `LineSymbolMap({1: [0x3000, 0x3001, 0x3002], 7: [0x4000, 0x4005]})`.
/// 
/// This data structure holds several invariants:
/// - Line numbers should never overlap.
/// - In a given block, the addresses should be in ascending order 
///     (this has to occur in a well-formed program because regions constitute contiguous, non-overlapping parts of memory).
/// 
/// If these invariants are not held, invalid behavior can occur.
#[derive(PartialEq, Eq, Clone)]
struct LineSymbolMap(BTreeMap<usize, Vec<u16>>);

impl LineSymbolMap {
    /// Creates a new line symbol table.
    /// 
    /// This takes an expanded list of line-memory address mappings and condenses it into 
    /// the internal [`LineSymbolMap`] format.
    /// 
    /// For example,
    /// 
    /// `[None, Some(0x3000), Some(0x3001), Some(0x3002), None, None, None, Some(0x4000), Some(0x4005)]` 
    /// condenses to `{1: [0x3000, 0x3001, 0x3002], 7: [0x4000, 0x4005]}`.
    /// 
    /// For a given block of contiguous `Some`s, the memory addresses should be sorted and accesses
    /// through `LineSymbolMap`'s methods assume the values are sorted.
    /// 
    /// If they are not sorted, incorrect behaviors may occur. Skill issue.
    fn new(lines: Vec<Option<u16>>) -> Option<Self> {
        let mut blocks = BTreeMap::new();
        let mut current = None;
        for (i, line) in lines.into_iter().enumerate() {
            match line {
                Some(addr) => current.get_or_insert_with(Vec::new).push(addr),
                None => if let Some(bl) = current.take() {
                    blocks.insert(i - bl.len(), bl);
                },
            }
        }

        Self::from_blocks(blocks)
    }

    fn from_blocks(blocks: impl IntoIterator<Item=(usize, Vec<u16>)>) -> Option<Self> {
        let mut bl: Vec<_> = blocks.into_iter().collect();

        bl.sort_by_key(|&(l, _)| l);
        
        // Check not overlapping:
        let not_overlapping = bl.windows(2).all(|win| {
            let [(ls, lb), (rs, _)] = win else { unreachable!() };
            ls + lb.len() <= *rs
        });

        match not_overlapping {
            true => {
                // Check every individual block is sorted:
                let sorted = bl.iter().all(|(_, lb)| {
                    lb.windows(2).all(|win| win[0] <= win[1])
                });

                sorted.then(|| Self(bl.into_iter().collect()))
            }
            false => None,
        }
    }

    /// Gets the memory address associated with this line, if it is present in the line symbol mapping.
    fn get(&self, line: usize) -> Option<u16> {
        // Find the block such that `line` falls within the source line number range of the block.
        let (start, block) = self.0.range(..=line).next_back()?;

        // Access the memory address.
        block.get(line - *start).copied()
    }

    /// Gets the source line number associated with this memory address, if it is present in the symbol table.
    fn find(&self, addr: u16) -> Option<usize> {
        self.0.iter()
            .find_map(|(start, words)| {
                // Find the block that contains the given address,
                // and then find the line index once it's found.
                words.binary_search(&addr)
                    .ok()
                    .map(|o| start + o)
            })
    }

    /// Gets an iterable representing the block mappings.
    fn block_iter(&self) -> impl Iterator<Item=(usize, &[u16])> + '_ {
        self.0.iter()
            .map(|(&i, words)| (i, words.as_slice()))
    }

    /// Gets an iterable representing the mapping of line numbers to addresses.
    fn iter(&self) -> impl Iterator<Item=(usize, u16)> + '_ {
        self.block_iter()
            .flat_map(|(i, words)| {
                words.iter()
                    .enumerate()
                    .map(move |(off, &addr)| (i + off, addr))
            })
    }
}
impl std::fmt::Debug for LineSymbolMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
        .entries(self.iter().map(|(i, v)| (i, Addr(v))))
        .finish()
    }
}
/// Struct holding the source string and contains helpers 
/// to index lines and to query position information from a source string.
#[derive(PartialEq, Eq, Clone)]
pub struct SourceInfo {
    /// The source code.
    src: String,
    /// The index of each new line in source code.
    nl_indices: Vec<usize>
}
impl std::fmt::Debug for SourceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceInfo")
            .field("nl_indices", &self.nl_indices)
            .finish_non_exhaustive()
    }
}
impl SourceInfo {
    /// Computes the source info from a given string.
    pub fn new(src: &str) -> Self {
        Self::from_string(src.to_string())
    }
    fn from_string(src: String) -> Self {
        // Index where each new line appears.
        let nl_indices: Vec<_> = src
            .match_indices('\n')
            .map(|(i, _)| i)
            .chain([src.len()])
            .collect();

        Self { src, nl_indices }
    }

    /// Returns the entire source.
    pub fn source(&self) -> &str {
        &self.src
    }

    /// Counts the number of lines in the source string.
    pub fn count_lines(&self) -> usize {
        // The first line, plus every line after (delimited by a new line)
        self.nl_indices.len()
    }

    /// Gets the character range for the provided line, including any whitespace
    /// and the newline character.
    /// 
    /// This returns None if line is not in the interval `[0, number of lines)`.
    fn raw_line_span(&self, line: usize) -> Option<Range<usize>> {
        // Implementation detail:
        // number of lines = self.nl_indices.len() + 1
        if !(0..self.count_lines()).contains(&line) {
            return None;
        };

        let start = match line {
            0 => 0,
            _ => self.nl_indices[line - 1] + 1
        };

        let eof = self.src.len();
        let end = match self.nl_indices.get(line) {
            Some(i) => (i + 1).min(eof), // incl NL, but don't go over EOF
            None => eof,
        };
        
        Some(start..end)
    }

    /// Gets the character range for the provided line, excluding any whitespace.
    /// 
    /// This returns None if line is not in the interval `[0, number of lines)`.
    pub fn line_span(&self, line: usize) -> Option<Range<usize>> {
        let Range { mut start, mut end } = self.raw_line_span(line)?;
        
        // shift line span by trim
        let line = &self.src[start..end];
        let end_trimmed = line.trim_end();
        end -= line.len() - end_trimmed.len();
        
        let line = end_trimmed;
        start += line.len() - line.trim_start().len();

        Some(start..end)
    }

    /// Reads a line from source.
    /// 
    /// This returns None if line is not in the interval `[0, number of lines)`.
    pub fn read_line(&self, line: usize) -> Option<&str> {
        self.line_span(line).map(|r| &self.src[r])
    }

    /// Gets the line number of the current position.
    fn get_line(&self, index: usize) -> usize {
        self.nl_indices.partition_point(|&start| start < index)
    }

    /// Calculates the line and character number for a given character index.
    /// 
    /// If the index exceeds the length of the string,
    /// the line number is given as the last line and the character number
    /// is given as the number of characters after the start of the line.
    pub fn get_pos_pair(&self, index: usize) -> (usize, usize) {
        let lno = self.get_line(index);

        let Range { start: lstart, .. } = self.raw_line_span(lno)
            .or_else(|| self.raw_line_span(self.nl_indices.len()))
            .unwrap_or(0..0);
        let cno = index - lstart;
        (lno, cno)
    }
}
impl From<&'_ str> for SourceInfo {
    fn from(value: &'_ str) -> Self {
        Self::new(value)
    }
}
impl From<String> for SourceInfo {
    fn from(value: String) -> Self {
        Self::from_string(value)
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Default, Debug)]
struct SymbolData {
    addr: u16,
    src_start: usize,
    external: bool
}
impl SymbolData {
    /// Calculates the source range of this symbol, given the name of the label.
    fn span(&self, label: &str) -> Range<usize> {
        self.src_start .. (self.src_start + label.len())
    }
}

/// Debug symbols.
#[derive(PartialEq, Eq, Debug, Clone)]
struct DebugSymbols {
    /// A mapping from each line with a statement in the source to an address.
    line_map: LineSymbolMap,

    /// Information about the source.
    src_info: SourceInfo
}
impl DebugSymbols {
    pub fn lookup_line(&self, line: usize) -> Option<u16> {
        self.line_map.get(line)
    }

    pub fn rev_lookup_line(&self, addr: u16) -> Option<usize> {
        self.line_map.find(addr)
    }

    /// Links debug symbols.
    pub fn link(mut a: Self, b: Self) -> Result<Self, AsmErr> {
        // TODO: This kinda just tacks files together into a single source.
        // Once object files have support for multiple ASM references, this should be cleaner.

        let lines = a.src_info.count_lines();

        // B doesn't overlap with A because ObjectFile check
        a.line_map.0.extend({
            b.line_map.0.into_iter()
                .map(|(k, v)| (k + lines, v))
        });

        a.src_info = SourceInfo::from_string(a.src_info.src + "\n" + &b.src_info.src);

        Ok(a)
    }
}
/// The symbol table created in the first assembler pass
/// that encodes source code mappings to memory addresses in the object file.
/// 
/// The symbol table consists of: 
/// - A mapping from source code labels to memory addresses.
/// - A mapping from source code line numbers to memory addresses (if debug symbols are enabled).
/// - The source text (if debug symbols are enabled).
/// 
/// Here is a table of the mappings that the symbol table provides:
/// 
/// | from ↓, to → | label                              | memory address                | source line/span                  |
/// |----------------|------------------------------------|-------------------------------|-----------------------------------|
/// | label          | -                                  | [`SymbolTable::lookup_label`] | [`SymbolTable::get_label_source`] |
/// | memory address | [`SymbolTable::rev_lookup_label`]  | -                             | [`SymbolTable::rev_lookup_line`]  |
/// | source line    | none                               | [`SymbolTable::lookup_line`]  | -                                 |
/// 
/// # Debug symbols
/// 
/// Debug symbols are optional data added to this symbol table which can help users debug their code.
/// 
/// Without debug symbols, the symbol table only consists of mappings from labels to spans (and vice-versa).
/// These are used to translate labels in source code to addresses during the assembly process.
/// After the completion of this assembly process, the [`SymbolTable`] is dropped and is not part of the resultant
/// [`ObjectFile`].
/// 
/// However, with debug symbols, this information persists in the resultant [`ObjectFile`], allowing
/// the label mappings to be accessed during simulation time. Additionally, more information from the source
/// text is available during simulation time:
/// - Mappings from source code line numbers to memory addresses
/// - Source code text (which grants access to line contents from a given line number; see [`SourceInfo`] for more details)
#[derive(PartialEq, Eq, Clone)]
pub struct SymbolTable {
    /// A mapping from label to address and span of the label.
    label_map: HashMap<String, SymbolData>,

    /// Relocation table.
    rel_map: HashMap<u16, String>,

    /// Debug symbols. If None, there were no debug symbols provided.
    debug_symbols: Option<DebugSymbols>,
}

impl SymbolTable {
    /// Creates a new symbol table.
    /// 
    /// This performs the first assembler pass, calculating the memory address of
    /// labels at each provided statement.
    /// 
    /// If a `src` argument is provided, debug symbols are also computed for the symbol table.
    /// 
    /// ## Example
    /// ```
    /// use lc3_ensemble::parse::parse_ast;
    /// use lc3_ensemble::asm::SymbolTable;
    /// 
    /// let src = "
    ///     .orig x3000
    ///     LABEL: HALT
    ///     .end
    /// ";
    /// let ast = parse_ast(src).unwrap();
    /// 
    /// // without debug symbols
    /// let sym = SymbolTable::new(&ast, None).unwrap();
    /// assert_eq!(sym.lookup_label("LABEL"), Some(0x3000));
    /// assert_eq!(sym.lookup_line(2), None);
    /// 
    /// // with debug symbols
    /// let sym = SymbolTable::new(&ast, Some(src)).unwrap();
    /// assert_eq!(sym.lookup_label("LABEL"), Some(0x3000));
    /// assert_eq!(sym.lookup_line(2), Some(0x3000));
    /// ```
    pub fn new(stmts: &[Stmt], src: Option<&str>) -> Result<Self, AsmErr> {
        struct Cursor {
            // The current location counter.
            lc: u16,
            // True if 0x10000.
            overflowed: bool,
            // The span of the .orig directive.
            block_orig: Span,
        }
        impl Cursor {
            fn new(lc: u16, block_orig: Span) -> Self {
                Self { lc, overflowed: false, block_orig }
            }
            /// Attempts to shift the LC forward by n word locations,
            /// failing if that would cause the LC to pass the IO region or
            /// overflow memory.
            fn shift(&mut self, n: u16) -> Result<(), AsmErrKind> {
                if n == 0 { return Ok(()); }

                match (self.overflowed, self.lc.checked_add(n)) {
                    (true, _) => Err(AsmErrKind::WrappingBlock),
                    (false, Some(new_lc)) if new_lc > IO_START => Err(AsmErrKind::BlockInIO),
                    (false, Some(new_lc)) => {
                        self.lc = new_lc;
                        Ok(())
                    },
                    (false, None) => {
                        let lc = std::mem::take(&mut self.lc);
                        self.overflowed = true;
                        // If aligns exactly, it can't be considered wrapping over
                        Err(match lc == n.wrapping_neg() {
                            true => AsmErrKind::BlockInIO,
                            false => AsmErrKind::WrappingBlock,
                        })
                    }
                }
            }
        }

        fn add_label(
            labels: &mut HashMap<String, SymbolData>, 
            label: &crate::ast::Label, 
            addr: u16,
            external: bool
        ) -> Result<(), AsmErr> {
            match labels.entry(label.name.to_uppercase()) {
                // Two labels with different addresses. Conflict.
                Entry::Occupied(e) if e.get().addr != addr => {
                    let span1 = e.get().span(e.key());
                    let span2 = label.span();
                    Err(AsmErr::new(AsmErrKind::OverlappingLabels, [span1, span2]))
                },
                // Two labels with same address. No conflict.
                Entry::Occupied(_) => Ok(()),
                // New label.
                Entry::Vacant(e) => {
                    e.insert(SymbolData { addr, src_start: label.span().start, external });
                    Ok(())
                }
            }
        }

        let mut cursor: Option<Cursor> = None;
        let mut label_map: HashMap<String, SymbolData> = HashMap::new();
        let mut rel_map = HashMap::new();
        let mut debug_sym = src.map(|s| {
            let src_info = SourceInfo::new(s);
            (vec![None; src_info.count_lines()], src_info)
        });

        for stmt in stmts {
            // Add labels if they exist
            if !stmt.labels.is_empty() {
                // If cursor does not exist, that means we're not in an .orig block,
                // so these labels don't have a known location
                let Some(cur) = cursor.as_ref() else {
                    let spans = stmt.labels.iter()
                        .map(|label| label.span())
                        .collect::<Vec<_>>();
                    
                    return Err(AsmErr::new(AsmErrKind::UndetAddrLabel, spans));
                };

                // Add labels
                for label in &stmt.labels {
                    add_label(&mut label_map, label, cur.lc, false)?;
                }
            }

            // Handle special directives:
            match &stmt.nucleus {
                StmtKind::Directive(Directive::Orig(addr)) => match cursor {
                    Some(cur) => return Err(AsmErr::new(AsmErrKind::OverlappingOrig, [cur.block_orig, stmt.span.clone()])),
                    None      => { cursor.replace(Cursor::new(addr.get(), stmt.span.clone())); },
                },
                StmtKind::Directive(Directive::End) => match cursor {
                    Some(_) => { cursor.take(); },
                    None    => return Err(AsmErr::new(AsmErrKind::UnopenedOrig, stmt.span.clone())),
                },
                StmtKind::Directive(Directive::External(label)) => {
                    // Arbitrarily chose 0 as a placeholder for externals
                    add_label(&mut label_map, label, 0, true)?;
                }
                StmtKind::Directive(Directive::Fill(PCOffset::Label(label))) => {
                    let label_text = label.name.to_uppercase();
                    if let Some(SymbolData { external: true, .. }) = label_map.get(&label_text) {
                        let Some(cur) = cursor.as_ref() else {
                            return Err(AsmErr::new(AsmErrKind::UndetAddrStmt, stmt.span.clone()));
                        };

                        rel_map.insert(cur.lc, label_text);
                    }
                },
                _ => {}
            };

            // If we're keeping track of the line counter currently (i.e., are inside of a .orig block):
            if let Some(cur) = &mut cursor {
                // Debug symbol:
                // Calculate which source code line is associated with the instruction the LC is currently pointing to
                // and add the mapping from line to instruction address.
                if let Some((lines, s)) = &mut debug_sym {
                    if !matches!(stmt.nucleus, StmtKind::Directive(Directive::Orig(_) | Directive::End)) {
                        let line_index = s.get_line(stmt.span.start);
                        lines[line_index].replace(cur.lc);
                    }
                }

                // Shift the LC forward
                match &stmt.nucleus {
                    StmtKind::Instr(_)     => cur.shift(1),
                    StmtKind::Directive(d) => cur.shift(d.word_len()),
                }.map_err(|e| AsmErr::new(e, stmt.span.clone()))?
            }
        }

        if let Some(cur) = cursor {
            return Err(AsmErr::new(AsmErrKind::UnclosedOrig, cur.block_orig));
        }
        
        let debug_symbols = debug_sym.map(|(lines, src_info)| DebugSymbols {
            line_map: LineSymbolMap::new(lines)
            .unwrap_or_else(|| {
                unreachable!("line symbol map's invariants should have been upheld during symbol table pass")
            }),
            src_info,
        });

        Ok(SymbolTable { label_map, rel_map, debug_symbols })
    }

    /// Gets the memory address of a given label (if it exists).
    /// 
    /// ## Example
    /// ```
    /// use lc3_ensemble::parse::parse_ast;
    /// use lc3_ensemble::asm::SymbolTable;
    /// 
    /// let src = "
    ///     .orig x3000
    ///     LOOP:
    ///         ADD R0, R0, #1
    ///         BR LOOP
    ///     LOOP2:
    ///         ADD R0, R0, #2
    ///         BR LOOP2
    ///     LOOP3:
    ///         ADD R0, R0, #3
    ///         BR LOOP3
    ///     .end
    /// ";
    /// let ast = parse_ast(src).unwrap();
    /// 
    /// let sym = SymbolTable::new(&ast, None).unwrap();
    /// assert_eq!(sym.lookup_label("LOOP"), Some(0x3000));
    /// assert_eq!(sym.lookup_label("LOOP2"), Some(0x3002));
    /// assert_eq!(sym.lookup_label("LOOP3"), Some(0x3004));
    /// assert_eq!(sym.lookup_label("LOOP_DE_LOOP"), None);
    /// ```
    pub fn lookup_label(&self, label: &str) -> Option<u16> {
        self.label_map.get(&label.to_uppercase()).map(|sym_data| sym_data.addr)
    }
    
    /// Gets the label at a given memory address (if it exists).
    /// 
    /// ## Example
    /// ```
    /// use lc3_ensemble::parse::parse_ast;
    /// use lc3_ensemble::asm::SymbolTable;
    /// 
    /// let src = "
    ///     .orig x3000
    ///     LOOP:
    ///         ADD R0, R0, #1
    ///         BR LOOP
    ///     LOOP2:
    ///         ADD R0, R0, #2
    ///         BR LOOP2
    ///     LOOP3:
    ///         ADD R0, R0, #3
    ///         BR LOOP3
    ///     .end
    /// ";
    /// let ast = parse_ast(src).unwrap();
    /// 
    /// let sym = SymbolTable::new(&ast, None).unwrap();
    /// assert_eq!(sym.rev_lookup_label(0x3000), Some("LOOP"));
    /// assert_eq!(sym.rev_lookup_label(0x3002), Some("LOOP2"));
    /// assert_eq!(sym.rev_lookup_label(0x3004), Some("LOOP3"));
    /// assert_eq!(sym.rev_lookup_label(0x2110), None);
    /// ```
    pub fn rev_lookup_label(&self, addr: u16) -> Option<&str> {
        let (label, _) = self.label_map.iter()
            .find(|&(_, sym_data)| sym_data.addr == addr)?;

        Some(label)
    }

    /// Gets the source span of a given label (if it exists).
    /// 
    /// ## Example
    /// ```
    /// use lc3_ensemble::parse::parse_ast;
    /// use lc3_ensemble::asm::SymbolTable;
    /// 
    /// let src = "
    ///     .orig x3000
    ///     LOOPY:
    ///         ADD R0, R0, #1
    ///         BR LOOPY
    ///     .end
    /// ";
    /// let ast = parse_ast(src).unwrap();
    /// 
    /// let sym = SymbolTable::new(&ast, None).unwrap();
    /// assert_eq!(sym.get_label_source("LOOPY"), Some(21..26));
    /// assert_eq!(sym.get_label_source("LOOP_DE_LOOP"), None);
    /// ```
    pub fn get_label_source(&self, label: &str) -> Option<Range<usize>> {
        self.label_map.get(label)
            .map(|data| data.span(label))
    }

    /// Gets the address of a given source line.
    /// 
    /// If debug symbols are not enabled, this unconditionally returns `None`.
    /// Note that each address is mapped to at most one source code line.
    /// 
    /// ## Example
    /// ```
    /// use lc3_ensemble::parse::parse_ast;
    /// use lc3_ensemble::asm::SymbolTable;
    /// 
    /// let src = "              ;;  0
    ///     .orig x3000          ;;  1
    ///     LOOP:                ;;  2
    ///         ADD R0, R0, #1   ;;  3
    ///         BR LOOP          ;;  4
    ///     .fill x9999          ;;  5
    ///     .blkw 10             ;;  6
    ///     LOOP2:               ;;  7
    ///         ADD R0, R0, #3   ;;  8
    ///         BR LOOP3         ;;  9
    ///     .end                 ;; 10
    /// ";
    /// let ast = parse_ast(src).unwrap();
    /// 
    /// // Debug symbols required:
    /// let sym = SymbolTable::new(&ast, Some(src)).unwrap();
    /// assert_eq!(sym.lookup_line(0),  None);
    /// assert_eq!(sym.lookup_line(1),  None);
    /// assert_eq!(sym.lookup_line(2),  None);
    /// assert_eq!(sym.lookup_line(3),  Some(0x3000));
    /// assert_eq!(sym.lookup_line(4),  Some(0x3001));
    /// assert_eq!(sym.lookup_line(5),  Some(0x3002));
    /// assert_eq!(sym.lookup_line(6),  Some(0x3003));
    /// assert_eq!(sym.lookup_line(7),  None);
    /// assert_eq!(sym.lookup_line(8),  Some(0x300D));
    /// assert_eq!(sym.lookup_line(9),  Some(0x300E));
    /// assert_eq!(sym.lookup_line(10), None);
    /// ```
    pub fn lookup_line(&self, line: usize) -> Option<u16> {
        self.debug_symbols.as_ref()?.lookup_line(line)
    }

    /// Gets the source line of a given memory address (if it exists.)
    /// 
    /// The result can be converted into a source span (range of characters encompassed by the instruction)
    /// using [`SymbolTable::source_info`] and [`SourceInfo::line_span`].
    /// 
    /// If debug symbols are not enabled, this unconditionally returns `None`.
    /// Note that each source code line is mapped to at most one address.
    /// 
    /// ## Example
    /// ```
    /// use lc3_ensemble::parse::parse_ast;
    /// use lc3_ensemble::asm::SymbolTable;
    /// 
    /// let src = "              ;;  0
    ///     .orig x3000          ;;  1
    ///     LOOP:                ;;  2
    ///         ADD R0, R0, #1   ;;  3
    ///         BR LOOP          ;;  4
    ///     .fill x9999          ;;  5
    ///     .blkw 10             ;;  6
    ///     LOOP2:               ;;  7
    ///         ADD R0, R0, #3   ;;  8
    ///         BR LOOP3         ;;  9
    ///     .end                 ;; 10
    /// ";
    /// let ast = parse_ast(src).unwrap();
    /// 
    /// // Debug symbols required:
    /// let sym = SymbolTable::new(&ast, Some(src)).unwrap();
    /// assert_eq!(sym.rev_lookup_line(0x3000),  Some(3));
    /// assert_eq!(sym.rev_lookup_line(0x3001),  Some(4));
    /// assert_eq!(sym.rev_lookup_line(0x3002),  Some(5));
    /// assert_eq!(sym.rev_lookup_line(0x3003),  Some(6));
    /// assert_eq!(sym.rev_lookup_line(0x3004),  None);
    /// assert_eq!(sym.rev_lookup_line(0x3005),  None);
    /// assert_eq!(sym.rev_lookup_line(0x3006),  None);
    /// assert_eq!(sym.rev_lookup_line(0x3007),  None);
    /// assert_eq!(sym.rev_lookup_line(0x3008),  None);
    /// assert_eq!(sym.rev_lookup_line(0x3009),  None);
    /// assert_eq!(sym.rev_lookup_line(0x300A),  None);
    /// assert_eq!(sym.rev_lookup_line(0x300B),  None);
    /// assert_eq!(sym.rev_lookup_line(0x300C),  None);
    /// assert_eq!(sym.rev_lookup_line(0x300D),  Some(8));
    /// assert_eq!(sym.rev_lookup_line(0x300E),  Some(9));
    /// assert_eq!(sym.rev_lookup_line(0x300F),  None);
    /// ```
    pub fn rev_lookup_line(&self, addr: u16) -> Option<usize> {
        self.debug_symbols.as_ref()?.rev_lookup_line(addr)
    }

    /// Reads the source info from this symbol table (if debug symbols are enabled).
    pub fn source_info(&self) -> Option<&SourceInfo> {
        self.debug_symbols.as_ref().map(|ds| &ds.src_info)
    }
    
    /// Gets an iterable of the mapping from labels to addresses.
    pub fn label_iter(&self) -> impl Iterator<Item=(&str, u16, bool)> + '_ {
        self.label_map.iter()
            .map(|(label, sym_data)| (&**label, sym_data.addr, sym_data.external))
    }

    /// Gets an iterable of the mapping from lines to addresses.
    /// 
    /// This iterator will be empty if debug symbols were not enabled.
    pub fn line_iter(&self) -> impl Iterator<Item=(usize, u16)> + '_ {
        self.debug_symbols.iter()
            .flat_map(|s| s.line_map.iter())
    }
}
impl std::fmt::Debug for SymbolTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct ClosureMap<R, F: Fn() -> R>(F);
        impl<K, V, R, F> std::fmt::Debug for ClosureMap<R, F> 
            where K: std::fmt::Debug,
                  V: std::fmt::Debug,
                  R: IntoIterator<Item=(K, V)>,
                  F: Fn() -> R
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_map()
                    .entries((self.0)())
                    .finish()
            }
        }

        f.debug_struct("SymbolTable")
            .field("label_map", &ClosureMap(|| {
                self.label_map.iter()
                    .map(|(k, data @ SymbolData { addr, .. })| {
                        (k, (Addr(*addr), data.span(k)))
                    })
            }))
            .field("debug_symbols", &self.debug_symbols)
            .finish()
    }
}

/// Replaces a [`PCOffset`] value with an [`Offset`] value by calculating the offset from a given label
/// (if this `PCOffset` represents a label).
fn replace_pc_offset<const N: u32>(off: PCOffset<i16, N>, pc: u16, sym: &SymbolTable) -> Result<IOffset<N>, AsmErr> {
    match off {
        PCOffset::Offset(off) => Ok(off),
        PCOffset::Label(label) => {
            // TODO: use sym.lookup_label
            match sym.label_map.get(&label.name.to_uppercase()) {
                Some(SymbolData { external: true, .. }) => Err(AsmErr::new(AsmErrKind::OffsetExternal, label.span())),
                Some(SymbolData { addr, .. }) => {
                    IOffset::new(addr.wrapping_sub(pc) as i16)
                        .map_err(|e| AsmErr::new(AsmErrKind::OffsetNewErr(e), label.span()))
                }
                None => Err(AsmErr::new(AsmErrKind::CouldNotFindLabel, label.span())),
            }
        },
    }
}

/// Checks if two ranges overlap.
/// 
/// This assumes (start <= end) for both ranges.
fn ranges_overlap<T: Ord>(a: Range<T>, b: Range<T>) -> bool {
    let Range { start: a_start, end: a_end } = a;
    let Range { start: b_start, end: b_end } = b;

    // Range not overlapping: a_start >= b_end || b_start >= a_end
    // This is just the inverse.
    a_start < b_end && b_start < a_end
}

impl AsmInstr {
    /// Converts an ASM instruction into a simulator instruction ([`SimInstr`])
    /// by resolving offsets and erasing aliases.
    /// 
    /// Parameters:
    /// - `pc`: PC increment
    /// - `sym`: The symbol table
    pub fn into_sim_instr(self, pc: u16, sym: &SymbolTable) -> Result<SimInstr, AsmErr> {
        match self {
            AsmInstr::ADD(dr, sr1, sr2) => Ok(SimInstr::ADD(dr, sr1, sr2)),
            AsmInstr::AND(dr, sr1, sr2) => Ok(SimInstr::AND(dr, sr1, sr2)),
            AsmInstr::BR(cc, off)       => Ok(SimInstr::BR(cc, replace_pc_offset(off, pc, sym)?)),
            AsmInstr::JMP(br)           => Ok(SimInstr::JMP(br)),
            AsmInstr::JSR(off)          => Ok(SimInstr::JSR(ImmOrReg::Imm(replace_pc_offset(off, pc, sym)?))),
            AsmInstr::JSRR(br)          => Ok(SimInstr::JSR(ImmOrReg::Reg(br))),
            AsmInstr::LD(dr, off)       => Ok(SimInstr::LD(dr, replace_pc_offset(off, pc, sym)?)),
            AsmInstr::LDI(dr, off)      => Ok(SimInstr::LDI(dr, replace_pc_offset(off, pc, sym)?)),
            AsmInstr::LDR(dr, br, off)  => Ok(SimInstr::LDR(dr, br, off)),
            AsmInstr::LEA(dr, off)      => Ok(SimInstr::LEA(dr, replace_pc_offset(off, pc, sym)?)),
            AsmInstr::NOT(dr, sr)       => Ok(SimInstr::NOT(dr, sr)),
            AsmInstr::RET               => Ok(SimInstr::JMP(Reg::R7)),
            AsmInstr::RTI               => Ok(SimInstr::RTI),
            AsmInstr::ST(sr, off)       => Ok(SimInstr::ST(sr, replace_pc_offset(off, pc, sym)?)),
            AsmInstr::STI(sr, off)      => Ok(SimInstr::STI(sr, replace_pc_offset(off, pc, sym)?)),
            AsmInstr::STR(sr, br, off)  => Ok(SimInstr::STR(sr, br, off)),
            AsmInstr::TRAP(vect)        => Ok(SimInstr::TRAP(vect)),
            AsmInstr::NOP(off)          => Ok(SimInstr::BR(0b000, replace_pc_offset(off, pc, sym)?)),
            AsmInstr::GETC              => Ok(SimInstr::TRAP(Offset::new_trunc(0x20))),
            AsmInstr::OUT               => Ok(SimInstr::TRAP(Offset::new_trunc(0x21))),
            AsmInstr::PUTC              => Ok(SimInstr::TRAP(Offset::new_trunc(0x21))),
            AsmInstr::PUTS              => Ok(SimInstr::TRAP(Offset::new_trunc(0x22))),
            AsmInstr::IN                => Ok(SimInstr::TRAP(Offset::new_trunc(0x23))),
            AsmInstr::PUTSP             => Ok(SimInstr::TRAP(Offset::new_trunc(0x24))),
            AsmInstr::HALT              => Ok(SimInstr::TRAP(Offset::new_trunc(0x25))),
        }
    }
}
impl Directive {
    /// How many words this directive takes up in memory.
    fn word_len(&self) -> u16 {
        match self {
            Directive::Orig(_)     => 0,
            Directive::Fill(_)     => 1,
            Directive::Blkw(n)     => n.get(),
            Directive::Stringz(s)  => s.len() as u16 + 1, // lex should assure that s + 1 <= 65535
            Directive::End         => 0,
            Directive::External(_) => 0,
        }
    }
}

/// An object file.
/// 
/// This is the final product after assembly source code is fully assembled.
/// This can be loaded in the simulator to run the assembled code.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ObjectFile {
    /// A mapping of each block's address to its corresponding data.
    /// 
    /// Invariants:
    /// - The blocks are sorted in order.
    /// - Blocks cannot wrap around in memory.
    /// - Blocks cannot write into xFE00-xFFFF.
    /// - As a corollary, block's length must fit in a `u16`.
    block_map: BTreeMap<u16, Vec<Option<u16>>>,

    /// Debug symbols.
    sym: Option<SymbolTable>
}
impl ObjectFile {
    /// Creates an empty object file.
    pub fn empty() -> Self {
        ObjectFile { block_map: BTreeMap::new(), sym: None }
    }

    /// Creates a new object file from an assembly AST and a symbol table.
    fn new(ast: Vec<Stmt>, sym: SymbolTable, debug: bool) -> Result<Self, AsmErr> {
        /// A singular block which represents a singular region in an object file.
        struct ObjBlock {
            /// Starting address of the block.
            start: u16,
            /// The words in the block.
            words: Vec<Option<u16>>,
            /// Span of the orig statement.
            /// 
            /// Used for error diagnostics in this function.
            orig_span: Range<usize>
        }

        impl ObjBlock {
            fn range(&self) -> Range<u16> {
                // Assumes no overflow and there cannot be more than u16::MAX words
                // Both of these invariants are asserted by `push` and `try_extend`.
                self.start .. (self.start + self.words.len() as u16)
            }

            fn push(&mut self, data: u16) {
                self.words.push(Some(data));
            }
            fn shift(&mut self, n: u16) {
                self.words.extend({
                    std::iter::repeat(None)
                        .take(usize::from(n))
                });
            }
            /// Writes the assembly for the given directive into the provided object block.
            fn write_directive(&mut self, directive: Directive, labels: &SymbolTable) -> Result<(), AsmErr> {
                match directive {
                    Directive::Orig(_) => {},
                    Directive::Fill(pc_offset) => {
                        let off = match pc_offset {
                            PCOffset::Offset(o) => o.get(),
                            PCOffset::Label(l)  => {
                                labels.lookup_label(&l.name)
                                    .ok_or_else(|| AsmErr::new(AsmErrKind::CouldNotFindLabel, l.span()))?
                            },
                        };

                        self.push(off);
                    },
                    Directive::Blkw(n) => self.shift(n.get()),
                    Directive::Stringz(n) => {
                        self.extend(n.bytes().map(u16::from));
                        self.push(0);
                    },
                    Directive::End => {},
                    Directive::External(_) => {},
                }

                Ok(())
            }
        }

        impl Extend<u16> for ObjBlock {
            fn extend<T: IntoIterator<Item = u16>>(&mut self, iter: T) {
                self.words.extend(iter.into_iter().map(Some));
            }
        }

        let mut block_map: BTreeMap<u16, ObjBlock> = BTreeMap::new();

        // PASS 2
        // Holding both the LC and currently writing block
        let mut current: Option<(u16, ObjBlock)> = None;

        for stmt in ast {
            match stmt.nucleus {
                StmtKind::Directive(Directive::Orig(off)) => {
                    debug_assert!(current.is_none());
                    
                    // Add new working block.
                    let addr = off.get();
                    current.replace((addr, ObjBlock { start: addr, orig_span: stmt.span, words: vec![] }));
                },
                StmtKind::Directive(Directive::End) => {
                    // The current block is complete, so take it out and append it to the block map.
                    let Some((_, block)) = current.take() else {
                        // unreachable (because pass 1 should've found it)
                        return Err(AsmErr::new(AsmErrKind::UnopenedOrig, stmt.span));
                    };

                    // only append if it's not empty:
                    if block.words.is_empty() { continue; }

                    // Check for overlap. Note this is probably overengineering:
                    let m_overlapping = [
                        block_map.range(..=block.start).next_back(), // previous block
                        block_map.range(block.start..).next(), // next block
                    ]
                        .into_iter()
                        .flatten()
                        .find(|(_, b)| ranges_overlap(block.range(), b.range()));

                    // If found overlapping block, raise error:
                    if let Some((_, overlapping_block)) = m_overlapping {
                        let span0 = block.orig_span;
                        let span1 = overlapping_block.orig_span.clone();

                        let order = match span0.start <= span1.start {
                            true  => [span0, span1],
                            false => [span1, span0],
                        };

                        return Err(AsmErr::new(AsmErrKind::OverlappingBlocks, order));
                    }

                    block_map.insert(block.start, block);
                },
                StmtKind::Directive(Directive::External(_)) => {},
                StmtKind::Directive(directive) => {
                    let Some((lc, block)) = &mut current else {
                        return Err(AsmErr::new(AsmErrKind::UndetAddrStmt, stmt.span));
                    };

                    let wl = directive.word_len();
                    block.write_directive(directive, &sym)?;
                    *lc = lc.wrapping_add(wl);
                },
                StmtKind::Instr(instr) => {
                    let Some((lc, block)) = &mut current else {
                        return Err(AsmErr::new(AsmErrKind::UndetAddrStmt, stmt.span));
                    };
                    let sim = instr.into_sim_instr(lc.wrapping_add(1), &sym)?;
                    block.push(sim.encode());
                    *lc = lc.wrapping_add(1);
                },
            }
        }

        let block_map = block_map.into_iter()
            .map(|(start, ObjBlock { words, .. })| (start, words))
            .collect();
        Ok(Self {
            block_map,
            sym: debug.then_some(sym),
        })
    }

    /// Gets a mutable reference to the value at the given address if defined in the object file.
    /// 
    /// If the data is uninitialized, this returns `Some(None)`.
    fn get_mut(&mut self, addr: u16) -> Option<&mut Option<u16>> {
        let (&start, block) = self.block_map.range_mut(..=addr).next_back()?;
        block.get_mut(addr.wrapping_sub(start) as usize)
    }

    /// Links two object files, combining them into one.
    /// 
    /// The linking algorithm is as follows:
    /// - The list of regions in both object files are merged into one.
    /// - Overlaps between regions are checked. If any are found, error.
    /// - For every symbol in the symbol table, this is added to the new symbol table.
    ///     - If any symbols appear more than once in different locations (and neither are external), error (duplicate labels).
    ///     - If any symbols appear more than once in different locations (and one is external), pull out any relocation entries (from `.LINKER_INFO`) for the external and match them.
    /// - Merge the remaining relocation table entries.
    pub fn link(mut a_obj: Self, b_obj: Self) -> Result<Self, AsmErr> {
        let Self { block_map: b_block_map, sym: b_sym } = b_obj;

        for (addr, block) in b_block_map {
            if a_obj.block_map.insert(addr, block).is_some() {
                return Err(AsmErr::new(AsmErrKind::OverlappingBlocks, []));
            }
        }

        let first = a_obj.block_map.iter();
        let mut second = a_obj.block_map.iter();
        second.next();
        if std::iter::zip(first, second).any(|((&a_st, a_bl), (&b_st, b_bl))| {
            let ar = a_st .. (a_st + a_bl.len() as u16);
            let br = b_st .. (b_st + b_bl.len() as u16);
            ranges_overlap(ar, br)
        }) {
            return Err(AsmErr::new(AsmErrKind::OverlappingBlocks, []));
        }

        // Merge symbol tables:
        let mut relocations = vec![];
        a_obj.sym = match (a_obj.sym, b_sym) {
            // If we have both symbol tables:
            (Some(mut a_sym), Some(b_sym)) => {
                let SymbolTable { label_map, rel_map, debug_symbols: b_debug_symbols } = b_sym;
                a_sym.debug_symbols = match (a_sym.debug_symbols, b_debug_symbols) {
                    (Some(ads), Some(bds)) => Some(DebugSymbols::link(ads, bds)?),
                    (m_ads, b_ads) => m_ads.or(b_ads)
                };

                // Cannot overlap due to the above overlapping blocks invariant.
                a_sym.rel_map.extend(rel_map);

                // For every label in symbol table B:
                for (label, b_sym_data) in label_map {
                    match a_sym.label_map.entry(label) {
                        Entry::Occupied(mut e) => {
                            let &a_sym_data = e.get();
                            match (a_sym_data.external, b_sym_data.external) {
                                // Two external labels
                                // Rel map entries are preserved, nothing changes.
                                (true, true) => {},

                                // One external label
                                // Bind all of the relocation entries corresponding to the external symbol 
                                // to the linked symbol
                                (true, false) | (false, true) => {
                                    // The address to link to.
                                    let linked_sym = match a_sym_data.external {
                                        true  => b_sym_data,
                                        false => a_sym_data
                                    };
                                    e.insert(linked_sym);

                                    // Split the relocation map to unmatching & matching labels.
                                    let (rel_addrs, new_rel_map) = a_sym.rel_map.into_iter()
                                        .partition(|(_, v)| v == e.key());
                                    a_sym.rel_map = new_rel_map;

                                    // Add matching labels to the "to relocate later" Vec.
                                    relocations.extend({
                                        rel_addrs.into_keys().map(|addr| (addr, linked_sym.addr))
                                    });
                                },

                                // No external labels
                                // If they point to the same addresses, nothing changes.
                                // If they point to different addresses, raise conflict.
                                (false, false) => if a_sym_data.addr != b_sym_data.addr {
                                    let a_span = a_sym_data.span(e.key());
                                    let b_span = b_sym_data.span(e.key());
                
                                    // TODO: this error does not have correct spans (since the files are different).
                                    return Err(AsmErr::new(AsmErrKind::OverlappingLabels, [a_span, b_span]));
                                }
                            }
                        },
                        Entry::Vacant(e) => { e.insert(b_sym_data); },
                    };
                }
                Some(a_sym)
            },
            (ma, mb) => ma.or(mb)
        };
        for (addr, linked_addr) in relocations {
            // TODO: handle case where the address needed is not found in block map
            // should really only occur from invalid manipulation of obj file
            a_obj.get_mut(addr)
                .unwrap_or_else(|| unreachable!("object file should have had address x{addr:04X} bound"))
                .replace(linked_addr);
        }

        Ok(a_obj)
    }

    /// Get an iterator over all of the blocks of the object file.
    pub(crate) fn block_iter(&self) -> impl Iterator<Item=(u16, &[Option<u16>])> {
        self.block_map.iter()
            .map(|(&addr, block)| (addr, block.as_slice()))
    }
    /// Checks if object file has any external symbols.
    pub(crate) fn get_external_symbol(&self) -> Option<&str> {
        self.symbol_table()
            .and_then(|s| s.label_map.iter().find(|(_, s)| s.external))
            .map(|(k, _)| k.as_str())
    }
    /// Gets an iterator over all of the memory locations defined in the object file.
    pub fn addr_iter(&self) -> impl Iterator<Item=(u16, Option<u16>)> + '_ {
        self.block_iter()
            .flat_map(|(addr, block)| {
                block.iter()
                    .enumerate()
                    .map(move |(i, &v)| (addr.wrapping_add(i as u16), v))
            })
    }
    /// Gets the symbol table if it is present in the object file.
    pub fn symbol_table(&self) -> Option<&SymbolTable> {
        self.sym.as_ref()
    }
}

/// Used for [`std::fmt::Debug`] purposes.
#[repr(transparent)]
struct Addr(u16);
impl std::fmt::Debug for Addr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "x{:04X}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;

    use crate::asm::encoding::TextFormat;
    use crate::asm::AsmErrKind;
    use crate::parse::parse_ast;

    use super::encoding::{BinaryFormat, ObjFileFormat};
    use super::{assemble_debug, AsmErr, ObjectFile};

    fn assemble_src(src: &str) -> Result<ObjectFile, AsmErr> {
        let ast = parse_ast(src).unwrap();
        assemble_debug(ast, src)
    }
    fn assert_asm_fail<T: std::fmt::Debug>(r: Result<T, AsmErr>, kind: AsmErrKind) {
        assert_eq!(r.unwrap_err().kind, kind);
    }

    #[test]
    fn test_sym_basic() {
        let src = "
        .orig x3000
            A ADD R0, R0, #0
            AND R0, R0, #1
            C ADD R0, R0, #0
            D LD R0, #-1
            HALT
            HALT
            HALT
            HALT
            E BR C
            HALT
            HALT
            HALT
            B JSR A
        .end
        ";

        let obj = assemble_src(src).unwrap();
        let sym = obj.symbol_table().unwrap();
        assert_eq!(sym.lookup_label("A"), Some(0x3000));
        assert_eq!(sym.lookup_label("C"), Some(0x3002));
        assert_eq!(sym.lookup_label("D"), Some(0x3003));
        assert_eq!(sym.lookup_label("E"), Some(0x3008));
        assert_eq!(sym.lookup_label("B"), Some(0x300C));
    }

    #[test]
    fn test_region_overlap() {
        // Two orig blocks, one after another
        let src = "
        .orig x3000
            HALT
            HALT
            HALT
            HALT
        .end

        .orig x3002
            HALT
        .end
        ";

        let obj = assemble_src(src);
        assert_asm_fail(obj, AsmErrKind::OverlappingBlocks);

        // Two orig blocks, one before another
        let src = "
        .orig x3002
            HALT
        .end

        .orig x3000
            HALT
            HALT
            HALT
            HALT
        .end
        ";

        let obj = assemble_src(src);
        assert_asm_fail(obj, AsmErrKind::OverlappingBlocks);

        // Two orig blocks, one empty
        let src = "
        .orig x3000
            HALT
            HALT
            HALT
            HALT
        .end

        .orig x3002
        .end
        ";
        assemble_src(src).unwrap();

        // Two orig blocks, one empty
        let src = "
        .orig x3002
        .end

        .orig x3000
            HALT
            HALT
            HALT
            HALT
        .end
        ";

        assemble_src(src).unwrap();
    }

    #[test]
    fn test_writing_into_io() {
        // write empty blocks
        let src = "
            .orig xFE00
            .end
        ";
        assemble_src(src).unwrap();

        let src = "
            .orig xFE02
            .end
        ";
        assemble_src(src).unwrap();

        // write actual block
        let src = "
            .orig xFE00
                AND R0, R0, #0
            .end
        ";
        let obj = assemble_src(src);
        assert_asm_fail(obj, AsmErrKind::BlockInIO);
    }

    #[test]
    fn test_big_blocks() {
        // big BLKW
        let src = "
            .orig x3000
            .blkw xFFFF
            .end
        ";

        let obj = assemble_src(src);
        assert_asm_fail(obj, AsmErrKind::WrappingBlock);

        // Bunch of .fill:
        let mut src = String::from(".orig x0000\n");
        for i in 0x0000..=0xFFFF {
            writeln!(src, ".fill x{i:04X}").unwrap();
        }
        writeln!(src, ".end").unwrap();

        let obj = assemble_src(&src);
        assert_asm_fail(obj, AsmErrKind::BlockInIO);

        // perfectly aligns
        let src = "
            .orig xFFFF
            .blkw 1
            .end
        ";
        let obj = assemble_src(src);
        assert_asm_fail(obj, AsmErrKind::BlockInIO);

        // perfectly aligns 2
        let src = "
            .orig x3000
            .blkw xD000
            .end
        ";
        let obj = assemble_src(src);
        assert_asm_fail(obj, AsmErrKind::BlockInIO);

        // big BLKW
        let src = "
            .orig x3000
            .blkw xFFFF
            .blkw xFFFF
            .blkw xFFFF
            .end
        ";
        let obj = assemble_src(src);
        assert_asm_fail(obj, AsmErrKind::WrappingBlock);

        // perfectly aligns and then does schenanigans
        let src = "
            .orig x3000
            LABEL1 .blkw xD000
            .fill x0000
            .fill x0001
            LABEL2 .fill x0002
            .fill x0003
            .end
        ";
        // Should error. Don't really care which error.
        assemble_src(src).unwrap_err();
    }

    fn assert_obj_equal(deser: &mut ObjectFile, expected: &ObjectFile, m: &str) {
        let deser_src = deser.sym.as_mut()
            .and_then(|s| s.debug_symbols.as_mut())
            .map(|s| &mut s.src_info.src)
            .expect("deserialized object file has no source");

        let expected_src = expected.sym.as_ref()
            .and_then(|s| s.debug_symbols.as_ref())
            .map(|s| &s.src_info.src)
            .expect("expected object file has no source");

        let deser_lines = deser_src.trim().lines().map(str::trim);
        let expected_lines = expected_src.trim().lines().map(str::trim);
        
        assert!(deser_lines.eq(expected_lines), "lines should have matched");
        
        let mut buf = expected_src.to_string();
        std::mem::swap(deser_src, &mut buf);
        assert_eq!(deser, expected, "{m}");

        // Revert change
        let deser_src = deser.sym.as_mut()
            .and_then(|s| s.debug_symbols.as_mut())
            .map(|s| &mut s.src_info.src)
            .expect("deserialized object file has no source");
        std::mem::swap(deser_src, &mut buf);
    }

    #[test]
    fn test_ser_deser() {
        let src = "
            .orig x3000
                AND R0, R0, #0
                ADD R0, R0, #15
                MINUS_R0 NOT R1, R0
                ADD R1, R1, #1
                HALT
            .end
        ";

        let obj = assemble_src(src).unwrap();
        
        // Binary format
        let ser = BinaryFormat::serialize(&obj);
        let mut de = BinaryFormat::deserialize(&ser).expect("binary encoding should've been parseable");
        assert_obj_equal(&mut de, &obj, "binary encoding could not be roundtripped");

        // Text format
        let ser = TextFormat::serialize(&obj);
        let mut de = TextFormat::deserialize(&ser).expect("text encoding should've been parseable");
        assert_obj_equal(&mut de, &obj, "text encoding could not be roundtripped");
    }
    
    #[test]
    fn test_ser_deser_crlf() {
        // With CRLF:
        let src = "\r
            .orig x3000\r
                AND R0, R0, #0\r
                ADD R0, R0, #15\r
                MINUS_R0 NOT R1, R0\r
                ADD R1, R1, #1\r
                HALT\r
            .end\r
        ";

        let obj = assemble_src(src).unwrap();
        
        // Text format
        let ser = TextFormat::serialize(&obj);
        let mut de = TextFormat::deserialize(&ser).expect("text encoding should've been parseable");
        assert_obj_equal(&mut de, &obj, "text encoding could not be roundtripped");
    }

    #[test]
    fn test_basic_link() {
        let library = "
            .orig x5000
                ADDER ADD R2, R0, R1
                RET
            .end
        ";

        let program = "
            .external ADDER

            .orig x4000
                LD R0, A
                LD R1, B

                LD R3, ADDER_ADDR
                JSRR R3

                HALT

                A .fill 10
                B .fill 20
                ADDER_ADDR .fill ADDER
            .end
        ";

        let lib_obj = assemble_src(library).unwrap();
        let prog_obj = assemble_src(program).unwrap();
        ObjectFile::link(lib_obj, prog_obj).unwrap();
    }
    #[test]
    fn test_link_overlapping_labels() {
        let library = "
            .orig x5000
                LOOP BR LOOP
                RET
            .end
        ";

        let program = "
            .orig x4000
                LOOP BR LOOP
                RET
            .end
        ";

        let lib_obj = assemble_src(library).unwrap();
        let prog_obj = assemble_src(program).unwrap();
        let link_obj = ObjectFile::link(lib_obj, prog_obj);
        assert_asm_fail(link_obj, AsmErrKind::OverlappingLabels);
    }

    #[test]
    fn test_link_overlapping_blocks() {
        // Overlapping blocks, but not identically overlapping
        let library = "
            .orig x3000
                ADD R0, R0, #0
                ADD R0, R0, #1
                ADD R0, R0, #2
                ADD R0, R0, #3
            .end
        ";

        let program = "
            .orig x3001
                ADD R0, R0, #0
                ADD R0, R0, #1
                ADD R0, R0, #2
                ADD R0, R0, #3
            .end
        ";

        let lib_obj = assemble_src(library).unwrap();
        let prog_obj = assemble_src(program).unwrap();
        let link_obj = ObjectFile::link(lib_obj, prog_obj);
        assert_asm_fail(link_obj, AsmErrKind::OverlappingBlocks);
        
        // Overlapping blocks and identically overlapping
        let library = "
            .orig x3000
                ADD R0, R0, #0
                ADD R0, R0, #1
                ADD R0, R0, #2
                ADD R0, R0, #3
            .end
        ";

        let program = "
            .orig x3000
                ADD R0, R0, #0
                ADD R0, R0, #1
                ADD R0, R0, #2
                ADD R0, R0, #3
            .end
        ";

        let lib_obj = assemble_src(library).unwrap();
        let prog_obj = assemble_src(program).unwrap();
        let link_obj = ObjectFile::link(lib_obj, prog_obj);
        assert_asm_fail(link_obj, AsmErrKind::OverlappingBlocks);
        
        // Contiguous blocks
        let library = "
            .orig x3000
                ADD R0, R0, #0
                ADD R0, R0, #1
                ADD R0, R0, #2
                ADD R0, R0, #3
            .end
        ";

        let program = "
            .orig x3004
                ADD R0, R0, #4
                ADD R0, R0, #5
                ADD R0, R0, #6
                ADD R0, R0, #7
            .end
        ";

        let lib_obj = assemble_src(library).unwrap();
        let prog_obj = assemble_src(program).unwrap();
        ObjectFile::link(lib_obj, prog_obj).unwrap();
    }

    #[test]
    fn test_link_order_agnostic() {
        let library = "
            .orig x5000
                ;; very functional MULTIPLY subroutine
                MULTIPLY RET
            .end
        ";
        let program1 = "
            .external MULTIPLY

            .orig x3000
                LD R1, MADDR1
                JSRR R1
                HALT

                MADDR1 .fill MULTIPLY
            .end
        ";
        let program2 = "
            .external MULTIPLY

            .orig x4000
                LD R2, MADDR2
                JSRR R2
                HALT

                MADDR2 .fill MULTIPLY
            .end
        ";

        let lib_obj = assemble_src(library).unwrap();
        let prog1_obj = assemble_src(program1).unwrap();
        let prog2_obj = assemble_src(program2).unwrap();
        
        // (lib + prog1) + prog2:
        let link_obj = ObjectFile::link(lib_obj.clone(), prog1_obj.clone()).unwrap();
        ObjectFile::link(link_obj.clone(), prog2_obj.clone())
            .expect("(lib + prog1) + prog2 should've succeeded");
        // prog2 + (lib + prog1):
        ObjectFile::link(prog2_obj.clone(), link_obj.clone())
            .expect("prog2 + (lib + prog1) should've succeeded");
        
        // (prog1 + lib) + prog2:
        let link_obj = ObjectFile::link(prog1_obj.clone(), lib_obj.clone()).unwrap();
        ObjectFile::link(link_obj.clone(), prog2_obj.clone())
            .expect("(prog1 + lib) + prog2 should've succeeded");
        // prog2 + (prog1 + lib):
        ObjectFile::link(prog2_obj.clone(), link_obj.clone())
            .expect("prog2 + (prog1 + lib) should've succeeded");
        
        // (prog1 + prog2) + lib:
        let link_obj = ObjectFile::link(prog1_obj.clone(), prog2_obj.clone()).unwrap();
        ObjectFile::link(link_obj.clone(), lib_obj.clone())
            .expect("(prog1 + prog2) + lib should've succeeded");
        // lib + (prog1 + prog2):
        ObjectFile::link(lib_obj.clone(), link_obj.clone())
            .expect("lib + (prog1 + prog2) should've succeeded");
    }

    #[test]
    fn test_link_external_place() {
        let program = "
            .external X
            .orig x3000
                .fill X
            .end
        ";
        assemble_src(program).unwrap();

        let program = "
            .external X
            .orig x3000
                LD R0, X
            .end
        ";
        assert_asm_fail(assemble_src(program), AsmErrKind::OffsetExternal);

        let program = "
            .external X
            .orig x3000
                JSR X
            .end
        ";
        assert_asm_fail(assemble_src(program), AsmErrKind::OffsetExternal);

        let program = "
            .external X
            .orig x3000
                BR X
            .end
        ";
        assert_asm_fail(assemble_src(program), AsmErrKind::OffsetExternal);
    }
}