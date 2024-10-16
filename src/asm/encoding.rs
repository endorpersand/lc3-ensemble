//! Formatters which can read and write memory object files into disk.
//! 
//! The [`ObjFileFormat`] trait describes an implementation of reading/writing object files into disk.
//! This module provides an implementation of the trait:
//! - [`BinaryFormat`]: A binary representation of object file data
//! - [`TextFormat`]: A text representation of object file data

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

use super::{DebugSymbols, ObjectFile, SymbolData, SymbolTable};

/// A trait defining object file formats.
// This trait might be an abuse of notation/namespacing, so oops.
pub trait ObjFileFormat {
    /// Representation of the serialized format.
    /// 
    /// For binary formats, `[u8]` should be used.
    /// For text-based formats,`str` should be used.
    type Stream: ToOwned + ?Sized;
    /// Serializes into the stream format.
    fn serialize(o: &ObjectFile) -> <Self::Stream as ToOwned>::Owned;
    /// Deserializes from the stream format, returning `None`
    /// if an error occurred during deserialization.
    fn deserialize(i: &Self::Stream) -> Option<ObjectFile>;
}

// BINARY!
/// A binary format of object file data.
pub struct BinaryFormat;

const BFMT_MAGIC: &[u8] = b"obj\x21\x10";
const BFMT_VER: &[u8] = b"\x00\x01";
impl ObjFileFormat for BinaryFormat {
    type Stream = [u8];

    fn serialize(o: &ObjectFile) -> <Self::Stream as ToOwned>::Owned {
        // Object file specification:
        //
        // Object file consists of a header and an arbitrary number of data blocks.
        // 
        // The header consists of:
        // - The magic number (b"obj\x21\x10")
        //      Coincidentally, `x21` is `!`, so opening this file with read "obj!"
        //      That's fun.
        // - The version (2 bytes)
        //      Note that this is really arbitrary and backwards-incompatible changes
        //      may occur without version upgrades.
        //      The version will likely only upgrade if for some extenuating circumstance,
        //      the exact object file format of a previous iteration must be maintained (never).
        //
        // Data is divided into discrete chunks, which start with one of:
        // - 0x00: assembled bytecode segment
        // - 0x01: label symbol table entry
        // - 0x02: line symbol table entry
        // - 0x03: source code information
        // - 0x04: relocation map entry
        //
        // Block 0x00 consists of:
        // - the identifier byte 0x00 (1 byte)
        // - address where block starts (2 bytes)
        // - length of the block (2 bytes)
        // - the .orig span (16 bytes)
        // - the array of words (3n bytes)
        //    - each word is either 0xFF???? (initialized data) or 0x000000 (uninitialized data)
        //
        // Block 0x01 consists of:
        // - the identifier byte 0x01 (1 byte)
        // - address of the label (2 bytes)
        // - whether the label is external (1 byte)
        // - the start of the label in source (8 bytes)
        // - the length of the label's name (8 bytes)
        // - the label (n bytes)
        //
        // Block 0x02 consists of:
        // - the identifier byte 0x02 (1 byte)
        // - the source line number (8 bytes)
        // - length of contiguous block (2 bytes)
        // - the contiguous block (2n bytes)
        // 
        // Block 0x03 consists of:
        // - the identifier byte 0x03 (1 byte)
        // - the length of the source code (8 bytes)
        // - the source code (n bytes)
        //
        // Block 0x04 consists of:
        // - the identifier byte 0x04 (1 byte)
        // - the address to replace at (2 bytes)
        // - the length of the label's name (8 bytes)
        // - the label (n bytes)

        let mut bytes = BFMT_MAGIC.to_vec();
        bytes.extend_from_slice(BFMT_VER);

        for (addr, data) in o.block_iter() {
            bytes.push(0x00);
            bytes.extend(u16::to_le_bytes(addr));
            bytes.extend(u16::to_le_bytes(data.len() as u16));
            for &word in data {
                if let Some(val) = word {
                    bytes.push(0xFF);
                    bytes.extend(u16::to_le_bytes(val));
                } else {
                    bytes.extend([0x00; 3]);
                }
            }
        }

        if let Some(sym) = &o.sym {
            for (label, &super::SymbolData { addr, src_start, external }) in sym.label_map.iter() {
                bytes.push(0x01);
                bytes.extend(u16::to_le_bytes(addr));
                bytes.push(u8::from(external));
                bytes.extend(u64::to_le_bytes(src_start as u64));
                bytes.extend(u64::to_le_bytes(label.len() as u64));
                bytes.extend_from_slice(label.as_bytes());
            }

            if let Some(debug_sym) = &sym.debug_symbols {
                for (lno, data) in debug_sym.line_map.block_iter() {
                    bytes.push(0x02);
                    bytes.extend(u64::to_le_bytes(lno as u64));
                    bytes.extend(u16::to_le_bytes(data.len() as u16));
                    for &word in data {
                        bytes.extend(u16::to_le_bytes(word));
                    }
                }

                let src = &debug_sym.src_info.src;
                bytes.push(0x03);
                bytes.extend(u64::to_le_bytes(src.len() as u64));
                bytes.extend_from_slice(src.as_bytes());
            }

            for (&addr, label) in &sym.rel_map {
                bytes.push(0x04);
                bytes.extend(u16::to_be_bytes(addr));
                bytes.extend(u64::to_le_bytes(label.len() as u64));
                bytes.extend_from_slice(label.as_bytes());
            }
        }

        bytes
    }

    fn deserialize(mut vec: &Self::Stream) -> Option<ObjectFile> {
        let mut block_map  = BTreeMap::new();
        let mut label_map  = HashMap::new();
        let mut rel_map    = HashMap::new();
        let mut debug_sym  = None::<(BTreeMap<_, _>, String)>;

        vec = vec.strip_prefix(BFMT_MAGIC)?
            .strip_prefix(BFMT_VER)?;

        while let Some((ident_byte, rest)) = vec.split_first() {
            vec = rest;
            match ident_byte {
                0x00 => {
                    let addr     = u16::from_le_bytes(take::<2>(&mut vec)?);
                    let data_len = u16::from_le_bytes(take::<2>(&mut vec)?);

                    let data = map_chunks::<_, 3>(take_slice(&mut vec, 3 * usize::from(data_len))?, 
                        |[init, rest @ ..]| (init == 0xFF).then(|| u16::from_le_bytes(rest))
                    );

                    block_map.insert(addr, data);
                },
                0x01 => {
                    let addr      = u16::from_le_bytes(take::<2>(&mut vec)?);
                    let external  = u8::from_le_bytes(take::<1>(&mut vec)?) != 0;
                    let src_start = u64::from_le_bytes(take::<8>(&mut vec)?) as usize;
                    let str_len   = u64::from_le_bytes(take::<8>(&mut vec)?) as usize;
                    let string    = String::from_utf8(take_slice(&mut vec, str_len)?.to_vec()).ok()?;

                    label_map.insert(string, SymbolData { addr, src_start, external });
                },
                0x02 => {
                    let (line_map, _) = debug_sym.get_or_insert_with(Default::default);
                    let lno      = u64::from_le_bytes(take::<8>(&mut vec)?) as usize;
                    let data_len = u16::from_le_bytes(take::<2>(&mut vec)?);
                    let data     = map_chunks::<_, 2>(take_slice(&mut vec, 2 * usize::from(data_len))?, u16::from_le_bytes);
                    
                    // Assert line map has sorted data without duplicates,
                    // as LineSymbolMap depends on the block being sorted
                    // and assumes no duplicates (since no two lines map to the same address)
                    assert_sorted_no_dup(&data)?;
                    line_map.insert(lno, data);
                },
                0x03 => {
                    let (_, src) = debug_sym.get_or_insert_with(Default::default);

                    let src_len = u64::from_le_bytes(take::<8>(&mut vec)?) as usize;
                    let obj_src = std::str::from_utf8(take_slice(&mut vec, src_len)?).ok()?;
                    src.push_str(obj_src);
                },
                0x04 => {
                    let addr = u16::from_le_bytes(take::<2>(&mut vec)?);
                    let label_len = u64::from_le_bytes(take::<8>(&mut vec)?) as usize;
                    let label = String::from_utf8(take_slice(&mut vec, label_len)?.to_vec()).ok()?;

                    rel_map.insert(addr, label);
                },
                _ => return None
            }
        }

        let debug_symbols = match debug_sym {
            Some((line_map, src)) => Some(DebugSymbols {
                // Error should propagate to deser
                line_map: super::LineSymbolMap::from_blocks(line_map)?,
                src_info: super::SourceInfo::from_string(src),
            }),
            None => None,
        };
        let sym = (!label_map.is_empty() || debug_symbols.is_some())
            .then_some(SymbolTable { label_map, debug_symbols, rel_map });
        Some(ObjectFile {
            block_map,
            sym,
        })
    }
}

fn take<const N: usize>(data: &mut &[u8]) -> Option<[u8; N]> {
    take_slice(data, N)
        .map(|slice| <[_; N]>::try_from(slice).unwrap())
}
fn take_slice<'a>(data: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    let (left, right) = try_split_at(data, n)?;
    *data = right;
    Some(left)
}
fn try_split_at(data: &[u8], n: usize) -> Option<(&[u8], &[u8])> {
    if n > data.len() { return None; }
    Some(data.split_at(n))
}
fn map_chunks<T, const N: usize>(data: &[u8], f: impl FnMut([u8; N]) -> T) -> Vec<T> {
    assert_eq!(data.len() % N, 0);
    
    data.chunks_exact(N)
        .map(|c| <[_; N]>::try_from(c).unwrap())
        .map(f)
        .collect()
}
fn assert_sorted_no_dup<T: Ord>(data: &[T]) -> Option<()> {
    data.windows(2)
        .map(|w| <&[_; 2]>::try_from(w).unwrap())
        .all(|[l, r]| l < r)
        .then_some(())
}

// TEXT!
/// A text-based format of object file data.
pub struct TextFormat;

const TFMT_MAGIC: &str = "LC-3 OBJ FILE";
const TFMT_UNINIT: &str = "????";
const TABLE_DIV: &str = " | ";

impl ObjFileFormat for TextFormat {
    type Stream = str;

    fn serialize(o: &ObjectFile) -> <Self::Stream as ToOwned>::Owned {
        // Text format specification.
        //
        // ```text
        // LC-3 OBJ FILE
        // 
        // .TEXT
        // <start address in hex>
        // <length of segment in dec>
        // <instruction in hex>
        // <...>
        //
        // .SYMBOL
        // ADDR | EXT | LABEL
        // 0000 |   0 | FOO  
        // 0001 |   0 | BAR  
        // 0002 |   1 | BAZ  
        // ...
        //
        // .LINKER_INFO
        // ADDR | LABEL
        // 0002 | BAZ
        // 0003 | BAZ
        // 0005 | BAZ
        // 
        // .DEBUG
        // LABEL | INDEX
        // FOO   | 35
        // BAR   | 94
        // ====================
        // LINE | ADDR | SOURCE
        // 0    | 9090 | ......
        // 1    | 9091 | ......
        // ====================
        // ...
        // // Support for comments, as well.
        // ```
        //
        // For a given `instruction in hex`, it prints the ASCII-hex encoding, returning ???? if uninitialized.
        fn _ser(o: &ObjectFile) -> Result<String, std::fmt::Error> {
            use std::fmt::Write;
            let mut buf = String::new();

            writeln!(buf, "{TFMT_MAGIC}")?;
            writeln!(buf)?;
            
            writeln!(buf, ".TEXT")?;
            for (addr, block) in o.block_iter() {
                writeln!(buf, "{addr:04X}")?;
                writeln!(buf, "{}", block.len())?;
                for &m_instr in block {
                    match m_instr {
                        Some(instr) => writeln!(buf, "{instr:04X}")?,
                        None => writeln!(buf, "{TFMT_UNINIT}")?,
                    }
                }
            }
            writeln!(buf)?;

            if let Some(sym) = &o.sym {
                writeln!(buf, ".SYMBOL")?;
                if !sym.label_map.is_empty() {
                    let mut sym_entries: Vec<_> = sym.label_iter().collect();
                    sym_entries.sort_by_key(|&(name, addr, ext)| (addr, name, ext));

                    writeln!(buf, "ADDR{0}EXT{0}LABEL", TABLE_DIV)?;
                    for (label, addr, external) in sym_entries {
                        writeln!(buf, "{addr:04X}{0}{1:3}{0}{label}", TABLE_DIV, u8::from(external))?;
                    }
                }
                writeln!(buf)?;
                
                writeln!(buf, ".LINKER_INFO")?;
                if !sym.rel_map.is_empty() {
                    let mut rel_entries: Vec<_> = sym.rel_map.iter().collect();
                    rel_entries.sort_by_key(|&(&addr, label)| (addr, label));

                    writeln!(buf, "ADDR{0}LABEL", TABLE_DIV)?;
                    for (addr, label) in rel_entries {
                        writeln!(buf, "{addr:04X}{0}{label}", TABLE_DIV)?;
                    }
                }
                writeln!(buf)?;

                writeln!(buf, ".DEBUG")?;
                writeln!(buf, "// DEBUG SYMBOLS FOR LC3TOOLS")?;
                writeln!(buf)?;

                // Display label to index mapping
                const LABEL: &str = "LABEL";
                const INDEX: &str = "INDEX";
                
                if !sym.label_map.is_empty() {
                    // Calculate label & index column lengths
                    let (label_col, index_col) = sym.label_map.iter()
                        .map(|(s, sym_data)| (s.len(), count_digits(sym_data.src_start)))
                        .fold(
                            (LABEL.len(), INDEX.len()), 
                            |(lc, ic), (lx, ix)| (lc.max(lx), ic.max(ix))
                        );

                    // Display!
                    writeln!(buf, "{LABEL:1$}{0}{INDEX:2$}", TABLE_DIV, label_col, index_col)?;
                    for (label, SymbolData { src_start, .. }) in sym.label_map.iter() {
                        writeln!(buf, "{label:1$}{0}{src_start:2$}", TABLE_DIV, label_col, index_col)?;
                    }
                }
                writeln!(buf, "====================")?;

                if let Some(DebugSymbols { line_map, src_info }) = &sym.debug_symbols {
                    // Create line table
                    let mut line_table = BTreeMap::from_iter({
                        src_info.nl_indices.iter().enumerate()
                            .map(|(lno, &idx)| (lno, (Some(idx), None)))
                    });
                    
                    for (start_line, block) in line_map.block_iter() {
                        for (i, &addr) in block.iter().enumerate() {
                            let (_, entry_addr) = line_table.entry(start_line.wrapping_add(i)).or_default();
                            entry_addr.replace(addr);
                        }
                    }
    
                    // Display line table
                    const LINE: &str = "LINE";
                    if !line_table.is_empty() {
                        // Compute line & index column length
                        let (mut last_line, mut last_index) = (None, None);
                        for (&line, &(index, _)) in line_table.iter().rev() {
                            if last_line.is_none() { last_line.replace(line); }
                            if last_index.is_none() { last_index = index; }
    
                            if last_line.is_some() && last_index.is_some() {
                                break;
                            }
                        }
                        let line_col = LINE.len().max(count_digits(last_line.unwrap_or(0)));
    
                        // Display!
                        writeln!(buf, "{LINE:1$}{0}ADDR{0}SOURCE", TABLE_DIV, line_col)?;
                        for (line, (_, m_addr)) in line_table {
                            write!(buf, "{line:0$}", line_col)?;
                            write!(buf, "{TABLE_DIV}")?;
                            match m_addr {
                                Some(addr) => write!(buf, "{addr:04X}"),
                                None => write!(buf, "{TFMT_UNINIT}")
                            }?;
                            write!(buf, "{TABLE_DIV}")?;
    
                            // Line:
                            let src_line = src_info.raw_line_span(line)
                                .and_then(|r| src_info.source().get(r))
                                .unwrap_or("");
                            write!(buf, "{src_line}")?;
    
                            writeln!(buf)?;
                        }
                    }
    
                    writeln!(buf, "====================")?;
                }
            }


            Ok(buf)
        }

        _ser(o).unwrap()
    }

    fn deserialize(string: &Self::Stream) -> Option<ObjectFile> {
        // Warning: spaghetti.

        let mut block_map  = BTreeMap::new();
        let mut label_map  = HashMap::<_, SymbolData>::new();
        let mut rel_map    = HashMap::new();
        let mut debug_sym  = None::<(Vec<_>, String)>;

        // Read all of the non-empty lines:
        let mut lines = string.trim().lines()
            .map(|l| {
                l.split_once("//").map_or(l, |(left, _)| left) // remove comments
            })
            .filter(|&l| !l.trim().is_empty());
        if lines.next() != Some(TFMT_MAGIC) { return None };

        let mut line_groups = vec![];
        for line in lines {
            if line.starts_with('.') {
                line_groups.push(vec![line]);
            } else {
                line_groups.last_mut()?.push(line);
            }
        }
        for group in line_groups {
            let [header, rest @ ..] = &*group else { return None };
            match *header {
                ".TEXT" => {
                    let mut it = rest.iter();
                    while let Some(orig_hex) = it.next() {
                        let orig = hex2u16(orig_hex)?;
                        let block_len = it.next()?.parse::<u16>().ok()?;

                        // Get and parse block of hex:
                        let block: Vec<_> = it.by_ref().take(usize::from(block_len))
                            .copied()
                            .map(maybe_hex2u16)
                            .collect::<Option<_>>()?;

                        if block.len() != usize::from(block_len) { return None; }
                        match block_map.entry(orig) {
                            std::collections::btree_map::Entry::Vacant(e) => e.insert(block),
                            std::collections::btree_map::Entry::Occupied(_) => return None,
                        };
                    }
                },
                ".SYMBOL" => {
                    let table = parse_table(rest, ["ADDR", "EXT", "LABEL"], |[addr_hex, ext, label], _| {
                        let addr = hex2u16(addr_hex)?;
                        let ext = ext.parse::<u8>().ok()? != 0;
                        Some((addr, ext, label))
                    }, true)?;

                    for (addr, ext, label) in table {
                        // TODO: what happens if .SYMBOL label + .DEBUG label mismatch
                        let entry = label_map.entry(label.to_string()).or_default();
                        
                        entry.addr = addr;
                        entry.external = ext;
                    }
                },
                ".LINKER_INFO" => {
                    let table = parse_table(rest, ["ADDR", "LABEL"], |[addr_hex, label], _| {
                        let addr = hex2u16(addr_hex)?;
                        Some((addr, label.to_string()))
                    }, true)?;

                    rel_map.extend(table);
                },
                ".DEBUG" => if !rest.is_empty() {
                    let split_pos = rest.iter().position(|l| l.starts_with('='))?;
                    if !rest.last()?.starts_with('=') { return None; }
                    let (label_src, [_, line_src @ .., _]) = rest.split_at(split_pos) else { unreachable!("divider should be present") };

                    let label_table = parse_table(label_src, ["LABEL", "INDEX"], |[label, index_str], _| {
                        let index = index_str.parse().ok()?;
                        Some((label, index))
                    }, true)?;
                    for (label, index) in label_table {
                        label_map.entry(label.to_string()).or_default().src_start = index;
                    }

                    let mut line_table = parse_table(line_src, ["LINE", "ADDR", "SOURCE"], |cols, i| {
                        let [rest @ .., source_str] = cols;
                        let [line_str, addr_str] = rest.map(str::trim);
                        
                        if line_str.parse::<usize>().ok()? != i { return None; }
                        let m_addr = maybe_hex2u16(addr_str)?;
                        Some((m_addr, source_str))
                    }, false)?;

                    if let Some((last_m_addr, last_line)) = line_table.pop() {
                        let (line_map, src) = debug_sym.get_or_insert_with(Default::default);

                        for (m_addr, line) in line_table {
                            line_map.push(m_addr);
                            writeln!(src, "{line}").unwrap();
                        }
                        
                        line_map.push(last_m_addr);
                        write!(src, "{last_line}").unwrap();
                    }
                },
                _ => return None
            }
        }

        let debug_symbols = match debug_sym {
            Some((line_map, src)) => Some(DebugSymbols {
                // propagate error to deser
                line_map: super::LineSymbolMap::new(line_map)?,
                src_info: super::SourceInfo::from_string(src),
            }),
            None => None,
        };
        let sym = (!label_map.is_empty() || debug_symbols.is_some())
            .then_some(SymbolTable { label_map, debug_symbols, rel_map });
        Some(ObjectFile {
            block_map,
            sym,
        })
    }
}

fn count_digits(n: usize) -> usize {
    (n.checked_ilog10().unwrap_or(0) + 1) as usize
}
fn hex2u16(s: &str) -> Option<u16> {
    match s.len() == 4 {
        true => u16::from_str_radix(s, 16).ok(),
        false => None
    }
}
fn maybe_hex2u16(s: &str) -> Option<Option<u16>> {
    match s {
        TFMT_UNINIT => Some(None),
        s => hex2u16(s).map(Some)
    }
}

fn parse_header(line: &str, columns: &[&str]) -> Option<()> {
    line.splitn(columns.len(), TABLE_DIV)
        .map(str::trim)
        .eq(columns.iter().copied())
        .then_some(())
}
fn parse_row<'a, T, const N: usize>(line: &'a str, f: impl FnOnce([&'a str; N]) -> Option<T>) -> Option<T> {
    let mut segments: Vec<_> = line
        .splitn(N, TABLE_DIV)
        .collect();
    segments.resize(N, "");
    let segments = *<Box<[_; N]>>::try_from(segments).ok()?;
    f(segments)
}
fn parse_table<'a, T, const N: usize>(
    contents: &[&'a str],
    columns: [&str; N], 
    mut row_parser: impl FnMut([&'a str; N], usize) -> Option<T>,
    trim: bool
) -> Option<Vec<T>> {
    // Accept empty tables:
    let Some((header, body)) = contents.split_first() else {
        return Some(vec![])
    };

    let trimfn = |s| match trim {
        true  => str::trim(s),
        false => s,
    };
    parse_header(header, &columns)?;
    body.iter()
        .enumerate()
        .map(|(i, l)| parse_row(l, |r| row_parser(r.map(trimfn), i)))
        .collect()
}