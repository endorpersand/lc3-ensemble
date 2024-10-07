//! Formatters which can read and write memory object files into disk.
//! 
//! The [`ObjFileFormat`] trait describes an implementation of reading/writing object files into disk.
//! This module provides an implementation of the trait:
//! - [`BinaryFormat`]: A binary representation of object file data

use std::collections::{BTreeMap, HashMap};

use super::{ObjectFile, SymbolTable};

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
        // - the length of the line indices table (8 bytes)
        // - the line indices table (8n bytes)
        // - the length of the source code (8 bytes)
        // the source code (n bytes)

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
            for (label, &(addr, span_start)) in sym.label_map.iter() {
                bytes.push(0x01);
                bytes.extend(u16::to_le_bytes(addr));
                bytes.extend(u64::to_le_bytes(span_start as u64));
                bytes.extend(u64::to_le_bytes(label.len() as u64));
                bytes.extend_from_slice(label.as_bytes());
            }

            for (lno, data) in sym.line_map.0.iter() {
                bytes.push(0x02);
                bytes.extend(u64::to_le_bytes(*lno as u64));
                bytes.extend(u16::to_le_bytes(data.len() as u16));
                for &word in data {
                    bytes.extend(u16::to_le_bytes(word));
                }
            }

            if let Some(src) = &sym.src_info {
                bytes.push(0x03);
                bytes.extend(u64::to_le_bytes(src.nl_indices.len() as u64));
                for &index in &src.nl_indices {
                    bytes.extend(u64::to_le_bytes(index as u64));
                }
                bytes.extend(u64::to_le_bytes(src.src.len() as u64));
                bytes.extend_from_slice(src.src.as_bytes());
            }
        }

        bytes
    }

    fn deserialize(mut vec: &Self::Stream) -> Option<ObjectFile> {
        let mut block_map  = BTreeMap::new();
        let mut label_map  = HashMap::new();
        let mut line_map   = BTreeMap::new();
        let mut nl_indices = None;
        let mut src = None;

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
                    let addr       = u16::from_le_bytes(take::<2>(&mut vec)?);
                    let span_start = u64::from_le_bytes(take::<8>(&mut vec)?) as usize;
                    let str_len    = u64::from_le_bytes(take::<8>(&mut vec)?) as usize;
                    let string     = String::from_utf8(take_slice(&mut vec, str_len)?.to_vec()).ok()?;

                    label_map.insert(string, (addr, span_start));
                },
                0x02 => {
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
                    let ref_li  = nl_indices.get_or_insert(vec![]);
                    let ref_src = src.get_or_insert(String::new());

                    let li_len = u64::from_le_bytes(take::<8>(&mut vec)?) as usize;
                    let li     = map_chunks::<_, 8>(take_slice(&mut vec, 8 * li_len)?, |arr| u64::from_le_bytes(arr) as usize);
                    ref_li.extend(li);

                    // Assert line indices has sorted data without duplicates,
                    // as calculations with nl_indices depends on the list being sorted
                    // and assumes no duplicates (since no two lines have the new line at the same place)
                    assert_sorted_no_dup(ref_li)?;

                    let src_len = u64::from_le_bytes(take::<8>(&mut vec)?) as usize;
                    let obj_src = std::str::from_utf8(take_slice(&mut vec, src_len)?).ok()?;
                    ref_src.push_str(obj_src);
                }
                _ => return None
            }
        }

        let sym = match !label_map.is_empty() || !line_map.is_empty() || nl_indices.is_some() {
            true => Some(SymbolTable {
                label_map,
                src_info: match !line_map.is_empty() || nl_indices.is_some() {
                    true  => Some(super::SourceInfo {
                        src: src?, 
                        nl_indices: nl_indices?,
                    }),
                    false => None,
                },
                line_map: super::LineSymbolMap(line_map)
            }),
            false => None,
        };
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