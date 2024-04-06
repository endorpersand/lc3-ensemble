//! Assembling assembly source ASTs into object files.
//! 
//! This module is used to convert source ASTs (`Vec<`[`Stmt`]`>`) into object files 
//! that can be executed by the simulator.
//! 
//! The assembler module notably consists of:
//! - [`assemble`]: The main function which assembles the statements into an object file.
//! - [`SymbolTable`]: a struct holding the symbol table, which stores location information for labels after the first assembler pass
//! - [`ObjectFile`]: a struct holding the object file, which can be loaded into the simulator and executed
//! 
//! [`Stmt`]: crate::ast::asm::Stmt

mod encoding;

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use logos::Span;

use crate::ast::asm::{AsmInstr, Directive, Stmt, StmtKind};
use crate::ast::sim::SimInstr;
use crate::ast::{IOffset, ImmOrReg, Offset, OffsetNewErr, PCOffset, Reg};
use crate::err::ErrSpan;


/// Assembles a assembly source code AST into an object file.
pub fn assemble(ast: Vec<Stmt>) -> Result<ObjectFile, AsmErr> {
    let sym = SymbolTable::new(&ast, None)?;
    create_obj_file(ast, sym, false)
}
/// Assembles a assembly source code AST into an object file.
/// 
/// This also registers debug symbols to the object file.
pub fn assemble_debug(ast: Vec<Stmt>, src: &str) -> Result<ObjectFile, AsmErr> {
    let sym = SymbolTable::new(&ast, Some(src))?;
    create_obj_file(ast, sym, true)
}

fn create_obj_file(ast: Vec<Stmt>, sym: SymbolTable, debug: bool) -> Result<ObjectFile, AsmErr> {
    let mut obj = ObjectFile::new();

    // PASS 2
    // Holding both the LC and currently writing block
    let mut current: Option<(u16, ObjBlock)> = None;

    for stmt in ast {
        match stmt.nucleus {
            StmtKind::Directive(Directive::Orig(off)) => {
                debug_assert!(current.is_none());
                
                // Add new working block.
                let addr = off.get();
                current.replace((addr + 1, ObjBlock { start: addr, orig_span: stmt.span, words: vec![] }));
            },
            StmtKind::Directive(Directive::End) => {
                // The current block is complete, so take it out and push it into the object file.
                let Some((_, ObjBlock { start, orig_span: start_span, words })) = current.take() else {
                    // unreachable (because pass 1 should've found it)
                    return Err(AsmErr::new(AsmErrKind::UnopenedOrig, stmt.span));
                };
                obj.push(start, start_span, words)?;
            },
            StmtKind::Directive(directive) => {
                let Some((lc, block)) = &mut current else {
                    return Err(AsmErr::new(AsmErrKind::UndetAddrStmt, stmt.span));
                };

                let wl = directive.word_len();
                directive.write_directive(&sym, block)?;
                *lc = lc.wrapping_add(wl);
            },
            StmtKind::Instr(instr) => {
                let Some((lc, block)) = &mut current else {
                    return Err(AsmErr::new(AsmErrKind::UndetAddrStmt, stmt.span));
                };
                let sim = instr.into_sim_instr(*lc, &sym)?;
                block.push(sim.encode());
                *lc = lc.wrapping_add(1);
            },
        }
    }

    if debug {
        obj.set_symbol_table(sym);
    }
    Ok(obj)
}

/// Kinds of errors that can occur from assembling given assembly code.
/// 
/// Error with span information is [`AsmErr`].
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
    /// There are blocks that overlap ranges of memory (pass 2).
    OverlappingBlocks,
    /// Creating the offset to replace a label caused overflow (pass 2).
    OffsetNewErr(OffsetNewErr),
    /// Label did not have an assigned address (pass 2).
    CouldNotFindLabel,
    /// Block is way too large (pass 2).
    ExcessiveBlock,
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
            Self::OverlappingBlocks => f.write_str("regions overlap in memory"),
            Self::OffsetNewErr(e)   => e.fmt(f),
            Self::CouldNotFindLabel => f.write_str("label could not be found"),
            Self::ExcessiveBlock    => write!(f, "block is larger than {} words", (1 << 16)),
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
            AsmErrKind::OffsetNewErr(e)   => e.help(),
            AsmErrKind::CouldNotFindLabel => Some("try adding the label before an instruction or directive".into()),
            AsmErrKind::ExcessiveBlock    => Some("try not doing that".into()),
        }
    }
}

/// A symbol table that maps source line numbers to memory addresses.
struct LineSymbolMap(Vec<(usize, Vec<u16>)>);

impl LineSymbolMap {
    /// Creates a new line symbol table.
    fn new(lines: Vec<Option<u16>>) -> Self {
        let mut blocks = vec![];
        let mut current = None;
        for (i, line) in lines.into_iter().enumerate() {
            match line {
                Some(addr) => current.get_or_insert_with(Vec::new).push(addr),
                None => if let Some(bl) = current.take() {
                    blocks.push((i - bl.len(), bl));
                },
            }
        }

        Self(blocks)
    }

    /// Gets the memory address associated with this line, if it is present in the symbol table.
    fn get(&self, line: usize) -> Option<u16> {
        use std::cmp::Ordering;

        let index = self.0.binary_search_by(|(start, words)| {
            match *start <= line {
                false => Ordering::Less,
                true  => match line < *start + words.len() {
                    true  => Ordering::Equal,
                    false => Ordering::Greater,
                },
            }
        }).ok()?;

        let (start, block) = &self.0[index];
        block.get(line - *start).copied()
    }

    /// Gets the source line number associated with this memory address, if it is present in the symbol table.
    fn find(&self, addr: u16) -> Option<usize> {
        self.0.iter()
            .find_map(|(start, words)| {
                words.binary_search(&addr)
                    .ok()
                    .map(|o| start + o)
            })
    }

    /// Gets an iterable representing the mapping of line numbers to addresses.
    fn iter(&self) -> impl Iterator<Item=(usize, u16)> + '_ {
        self.0.iter()
            .flat_map(|(i, words)| {
                words.iter()
                    .enumerate()
                    .map(move |(off, &addr)| (i + off, addr))
            })
    }
}

/// Details some encoding information about the source.
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
    /// Returns the entire source.
    pub fn source(&self) -> &str {
        &self.src
    }

    /// Gets the character range for the provided line, including any whitespace.
    /// 
    /// This returns None if line is not in the interval `[0, number of lines)`.
    fn raw_line_span(&self, line: usize) -> Option<Range<usize>> {
        // Implementation detail:
        // number of lines = self.nl_indices.len() + 1
        if !(0..=self.nl_indices.len()).contains(&line) {
            return None;
        };

        let end = match self.nl_indices.get(line) {
            Some(&n) => n,
            None     => self.src.len(),
        };
        
        let start = match line == 0 {
            false => self.nl_indices[line - 1] + 1,
            true  => 0,
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
    pub fn read_line(&self, line: usize) -> Option<&str> {
        Some(&self.src[self.line_span(line)?])
    }

    /// Calculates the line and character number for a given character index.
    /// 
    /// If the index exceeds the length of the string,
    /// the line number is given as the last line and the character number
    /// is given as the number of characters after the start of the line.
    pub fn get_pos_pair(&self, index: usize) -> (usize, usize) {
        let lno = self.nl_indices.partition_point(|&start| start < index);

        let Range { start: lstart, .. } = self.raw_line_span(lno)
            .or_else(|| self.raw_line_span(self.nl_indices.len()))
            .unwrap_or(0..0);
        let cno = index - lstart;
        (lno, cno)
    }
}

/// The symbol table created in the first assembler pass
/// that encodes source code mappings to memory addresses in the object file.
/// 
/// The symbol table consists of: 
/// - A mapping from source code labels to memory addresses.
/// - A mapping from source code line numbers to memory addresses (if debug symbols are enabled, see [`SymbolTable::new`]).
/// - The source text (if debug symbols are enabled, see [`SymbolTable::new`]).
pub struct SymbolTable {
    /// A mapping from label to address and span of the label.
    label_map: HashMap<String, (u16, usize)>,

    /// A mapping from each line with a statement in the source to an address.
    line_map: LineSymbolMap,

    /// Information about the source.
    src_info: Option<SourceInfo>,
}

impl SymbolTable {
    /// Creates a new symbol table.
    /// 
    /// This performs the first assembler pass, calculating the memory address of
    /// labels at each provided statement.
    /// 
    /// If a `src` argument is provided, this also computes debug symbols and includes them
    /// within this `SymbolTable` struct. These debug symbols include:
    /// - Mappings from source code line numbers to memory addresses
    /// - Source code text (and more detailed calculations with source code, see [`SourceInfo`] for more details)
    pub fn new(stmts: &[Stmt], src: Option<&str>) -> Result<Self, AsmErr> {
        struct Cursor {
            // The current location counter.
            lc: u16,
            // The length of the current block being read.
            block_len: u16,
            // The span of the .orig directive.
            block_orig: Span,
        }
        impl Cursor {
            /// Attempts to shift the LC forward by n word locations,
            /// failing if that would overflow the size of the block.
            /// 
            /// This returns if it was successful.
            fn shift(&mut self, n: u16) -> bool {
                let Some(new_len) = self.block_len.checked_add(n) else { return false };

                self.lc = self.lc.wrapping_add(n);
                self.block_len = new_len;
                true
            }
        }

        // Index where each new line appears.
        let nl_indices: Vec<_> = src.unwrap_or("")
            .match_indices('\n')
            .map(|(i, _)| i)
            .collect();

        let mut lc: Option<Cursor> = None;
        let mut labels: HashMap<String, (u16, Span)> = HashMap::new();
        let mut lines = vec![];
        lines.resize(nl_indices.len() + 1, None);

        for stmt in stmts {
            // Add labels if they exist
            if !stmt.labels.is_empty() {
                let Some(cur) = lc.as_ref() else {
                    let spans = stmt.labels.iter()
                        .map(|label| label.span())
                        .collect::<Vec<_>>();
                    
                    return Err(AsmErr::new(AsmErrKind::UndetAddrLabel, spans));
                };

                for label in &stmt.labels {
                    match labels.entry(label.name.to_uppercase()) {
                        Entry::Occupied(e) => {
                            let (_, span1) = e.get();
                            return Err(AsmErr::new(AsmErrKind::OverlappingLabels, [span1.clone(), label.span()]))
                        },
                        Entry::Vacant(e) => e.insert((cur.lc, label.span())),
                    };
                }
            }

            // Handle .orig, .end cases:
            match &stmt.nucleus {
                StmtKind::Directive(Directive::Orig(addr)) => match lc {
                    Some(cur) => return Err(AsmErr::new(AsmErrKind::OverlappingOrig, [cur.block_orig, stmt.span.clone()])),
                    None      => { lc.replace(Cursor { lc: addr.get(), block_len: 0, block_orig: stmt.span.clone() }); },
                },
                StmtKind::Directive(Directive::End) => match lc {
                    Some(_) => { lc.take(); },
                    None    => return Err(AsmErr::new(AsmErrKind::UnopenedOrig, stmt.span.clone())),
                },
                _ => {}
            };

            // Shift the location counter, and link the source line with the LC.
            if let Some(cur) = &mut lc {
                if src.is_some() {
                    // Calculate line index and put it in self.lines.
                    if !matches!(stmt.nucleus, StmtKind::Directive(Directive::Orig(_) | Directive::End)) {
                        let line_index = nl_indices.partition_point(|&start| start < stmt.span.start);
                        lines[line_index].replace(cur.lc);
                    }
                }

                let success = match &stmt.nucleus {
                    StmtKind::Instr(_)     => cur.shift(1),
                    StmtKind::Directive(d) => cur.shift(d.word_len()),
                };

                if !success { return Err(AsmErr::new(AsmErrKind::ExcessiveBlock, cur.block_orig.clone())) }
            }
        }

        match lc {
            None => Ok(SymbolTable {
                label_map: labels.into_iter().map(|(k, (addr, span))| (k, (addr, span.start))).collect(),
                line_map: LineSymbolMap::new(lines),
                src_info: src.map(|s| SourceInfo {
                    src: s.to_string(),
                    nl_indices,
                })
            }),
            Some(cur) => Err(AsmErr::new(AsmErrKind::UnclosedOrig, cur.block_orig)),
        }
    }

    /// Gets the memory address of a given label (if it exists).
    pub fn get_label(&self, label: &str) -> Option<u16> {
        self.label_map.get(&label.to_uppercase()).map(|&(addr, _)| addr)
    }

    /// Gets the source span of a given label (if it exists).
    pub fn find_label_source(&self, label: &str) -> Option<Range<usize>> {
        let &(_, start) = self.label_map.get(label)?;
        Some(start..(start + label.len()))
    }

    /// Gets an iterable of the mapping from labels to addresses.
    pub fn label_iter(&self) -> impl Iterator<Item=(&str, u16)> + '_ {
        self.label_map.iter()
            .map(|(label, &(addr, _))| (&**label, addr))
    }

    /// Gets the address of a given source line.
    /// 
    /// If debug symbols are not enabled, this unconditionally returns `None`.
    pub fn get_line(&self, line: usize) -> Option<u16> {
        self.line_map.get(line)
    }

    /// Gets the source line of a given memory address (if it exists.)
    /// 
    /// The result can be converted into a source span (range of characters encompassed by the instruction)
    /// using [`SymbolTable::source_info`] and [`SourceInfo::line_span`].
    /// 
    /// If debug symbols are not enabled, this unconditionally returns `None`.
    pub fn find_line_source(&self, addr: u16) -> Option<usize> {
        self.line_map.find(addr)
    }

    /// Reads the source info from this symbol table (if debug symbols are enabled).
    pub fn source_info(&self) -> Option<&SourceInfo> {
        self.src_info.as_ref()
    }
}
impl std::fmt::Debug for SymbolTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::cell::Cell;

        #[repr(transparent)]
        struct Addr(u16);
        impl std::fmt::Debug for Addr {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "x{:04X}", self.0)
            }
        }

        struct Map<M>(Cell<Option<M>>);
        impl<M> Map<M> {
            fn new(m: M) -> Self {
                Map(Cell::new(Some(m)))
            }
        }
        impl<K: std::fmt::Debug, V: std::fmt::Debug, M: IntoIterator<Item=(K, V)>> std::fmt::Debug for Map<M> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_map()
                    .entries(self.0.take().unwrap())
                    .finish()
            }
        }

        f.debug_struct("SymbolTable")
            .field("label_map", &Map::new({
                self.label_map.iter()
                    .map(|(k, &(addr, start))| (k, (Addr(addr), start..(start + k.len()))))
            }))
            .field("line_map", &Map::new({
                self.line_map.iter()
                    .map(|(i, v)| (i, Addr(v)))
            }))
            .field("source_info", &self.src_info)
            .finish()
    }
}

/// Replaces a [`PCOffset`] value with an [`Offset`] value by calculating the offset from a given label
/// (if this `PCOffset` represents a label).
fn replace_pc_offset<const N: u32>(off: PCOffset<i16, N>, lc: u16, sym: &SymbolTable) -> Result<IOffset<N>, AsmErr> {
    match off {
        PCOffset::Offset(off) => Ok(off),
        PCOffset::Label(label) => {
            let Some(loc) = sym.get_label(&label.name) else { return Err(AsmErr::new(AsmErrKind::CouldNotFindLabel, label.span())) };
            IOffset::new(loc.wrapping_sub(lc) as i16)
                .map_err(|e| AsmErr::new(AsmErrKind::OffsetNewErr(e), label.span()))
        },
    }
}

impl AsmInstr {
    /// Converts an ASM instruction into a simulator instruction ([`SimInstr`])
    /// by resolving offsets and erasing aliases.
    pub fn into_sim_instr(self, lc: u16, sym: &SymbolTable) -> Result<SimInstr, AsmErr> {
        match self {
            AsmInstr::ADD(dr, sr1, sr2) => Ok(SimInstr::ADD(dr, sr1, sr2)),
            AsmInstr::AND(dr, sr1, sr2) => Ok(SimInstr::AND(dr, sr1, sr2)),
            AsmInstr::BR(cc, off)       => Ok(SimInstr::BR(cc, replace_pc_offset(off, lc, sym)?)),
            AsmInstr::JMP(br)           => Ok(SimInstr::JMP(br)),
            AsmInstr::JSR(off)          => Ok(SimInstr::JSR(ImmOrReg::Imm(replace_pc_offset(off, lc, sym)?))),
            AsmInstr::JSRR(br)          => Ok(SimInstr::JSR(ImmOrReg::Reg(br))),
            AsmInstr::LD(dr, off)       => Ok(SimInstr::LD(dr, replace_pc_offset(off, lc, sym)?)),
            AsmInstr::LDI(dr, off)      => Ok(SimInstr::LDI(dr, replace_pc_offset(off, lc, sym)?)),
            AsmInstr::LDR(dr, br, off)  => Ok(SimInstr::LDR(dr, br, off)),
            AsmInstr::LEA(dr, off)      => Ok(SimInstr::LEA(dr, replace_pc_offset(off, lc, sym)?)),
            AsmInstr::NOT(dr, sr)       => Ok(SimInstr::NOT(dr, sr)),
            AsmInstr::RET               => Ok(SimInstr::JMP(Reg(0b111))),
            AsmInstr::RTI               => Ok(SimInstr::RTI),
            AsmInstr::ST(sr, off)       => Ok(SimInstr::ST(sr, replace_pc_offset(off, lc, sym)?)),
            AsmInstr::STI(sr, off)      => Ok(SimInstr::STI(sr, replace_pc_offset(off, lc, sym)?)),
            AsmInstr::STR(sr, br, off)  => Ok(SimInstr::STR(sr, br, off)),
            AsmInstr::TRAP(vect)        => Ok(SimInstr::TRAP(vect)),
            AsmInstr::NOP(off)          => Ok(SimInstr::BR(0b000, replace_pc_offset(off, lc, sym)?)),
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
            Directive::Orig(_)    => 0,
            Directive::Fill(_)    => 1,
            Directive::Blkw(n)    => n.get(),
            Directive::Stringz(s) => s.len() as u16 + 1, // lex should assure that s + 1 <= 65535
            Directive::End        => 0,
        }
    }

    /// Writes the assembly for the given directive into the provided object block.
    /// 
    /// This also returns the total number of memory locations written.
    fn write_directive(self, labels: &SymbolTable, block: &mut ObjBlock) -> Result<(), AsmErr> {
        match self {
            Directive::Orig(_) => {},
            Directive::Fill(pc_offset) => {
                let off = match pc_offset {
                    PCOffset::Offset(o) => o.get(),
                    PCOffset::Label(l)  => {
                        labels.get_label(&l.name)
                            .ok_or_else(|| AsmErr::new(AsmErrKind::CouldNotFindLabel, l.span()))?
                    },
                };

                block.push(off);
            },
            Directive::Blkw(n) => block.shift(n.get()),
            Directive::Stringz(n) => {
                block.extend(n.bytes().map(u16::from));
                block.push(0);
            },
            Directive::End => {},
        }

        Ok(())
    }
}

/// A singular block which represents a singular region in an object file.
struct ObjBlock {
    /// Starting address of the block.
    start: u16,
    /// Span of the orig statement.
    orig_span: Range<usize>,
    /// The words in the block.
    words: Vec<Option<u16>>
}
impl ObjBlock {
    fn push(&mut self, data: u16) {
        self.words.push(Some(data));
    }
    fn shift(&mut self, n: u16) {
        self.words.extend({
            std::iter::repeat(None)
                .take(usize::from(n))
        });
    }
}
impl Extend<u16> for ObjBlock {
    fn extend<T: IntoIterator<Item = u16>>(&mut self, iter: T) {
        self.words.extend(iter.into_iter().map(Some));
    }
}

/// An object file.
/// 
/// This is the final product after assembly source code is fully assembled.
/// This can be loaded in the simulator to run the assembled code.
#[derive(Debug)]
pub struct ObjectFile {
    /// A mapping of each block's address to its corresponding data and 
    /// the span of the .orig statement that starts the block.
    /// 
    /// Note that the length of a block should fit in a `u16`, so the
    /// block can be a maximum of 65535 words.
    block_map: BTreeMap<u16, (Vec<Option<u16>>, Span)>,

    /// Debug symbols.
    sym: Option<SymbolTable>
}
impl ObjectFile {
    /// Creates a new, empty [`ObjectFile`].
    pub fn new() -> Self {
        ObjectFile {
            block_map: BTreeMap::new(),
            sym: None
        }
    }

    /// Add a new block to the object file, writing the provided words (`words`) at the provided address (`start`).
    /// 
    /// This will error if this block overlaps with another block already present in the object file.
    pub fn push(&mut self, start: u16, start_span: Range<usize>, words: Vec<Option<u16>>) -> Result<(), AsmErr> {
        // Only add to object file if non-empty:
        if !words.is_empty() {
            // Find previous block and ensure no overlap:
            let prev_block = self.block_map.range(..=start).next_back()
                .or_else(|| self.block_map.last_key_value());

            if let Some((&prev_start, (prev_words, prev_span))) = prev_block {
                // check if this block overlaps with the previous block
                if (start.wrapping_sub(prev_start) as usize) < prev_words.len() {
                    return Err(AsmErr::new(AsmErrKind::OverlappingBlocks, [prev_span.clone(), start_span]));
                }
            }

            // No overlap, so we can add it:
            self.block_map.insert(start, (words, start_span));
        }

        Ok(())
    }

    /// Get an iterator over all of the blocks of the object file.
    pub fn iter(&self) -> impl Iterator<Item=(u16, &[Option<u16>])> {
        self.block_map.iter()
            .map(|(&addr, (block, _))| (addr, block.as_slice()))
    }

    /// Counts the number of blocks in this object file.
    pub fn len(&self) -> usize {
        self.block_map.len()
    }
    /// Returns whether there are blocks in this object file.
    pub fn is_empty(&self) -> bool {
        self.block_map.is_empty()
    }

    fn set_symbol_table(&mut self, sym: SymbolTable) {
        self.sym.replace(sym);
    }
    /// Gets the symbol table if it is present in the object file.
    pub fn symbol_table(&self) -> Option<&SymbolTable> {
        self.sym.as_ref()
    }
}

impl Default for ObjectFile {
    fn default() -> Self {
        Self::new()
    }
}