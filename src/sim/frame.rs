//! The frame stack and call frame management.
//! 
//! This module exposes:
//! - [`FrameStack`]: The frame stack used by the Simulator.
//! - [`Frame`]: All the data from a given frame.
//! - [`ParameterList`]: An enum which defines the signature of a subroutine or trap.

use std::collections::HashMap;

use crate::ast::reg_consts::{R0, R6};
use crate::ast::Reg;

use super::mem::{Mem, RegFile, Word};


/// A list of parameters, used to define the signature of a subroutine or trap.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ParameterList {
    /// A parameter list defined with standard LC-3 calling convention.
    /// 
    /// If a subroutine is defined with this parameter list variant,
    /// arguments are pulled from the stack at FP+4 to FP+4+n.
    /// 
    /// The `params` field defines the names of the parameters accepted by this
    /// subroutine or trap.
    /// 
    /// This variant can be readily created with [`ParameterList::with_calling_convention`].
    CallingConvention {
        /// Names for the parameters.
        params: Vec<String>
    },

    /// A parameter list defined by pass-by-register calling convention.
    /// 
    /// If a subroutine is defined with this parameter list variant,
    /// arguments are pulled from registers.
    /// 
    /// The `params` field defines the name of the parameters accepted by this
    /// subroutine or trap and the register where the argument is located.
    /// 
    /// The `ret` field defines which register the return value is located in
    /// (if it exists).
    /// 
    /// This variant can be readily created with [`ParameterList::with_pass_by_register`].
    PassByRegister {
        /// Names for the parameters and the register the parameter is located at.
        params: Vec<(String, Reg)>,
        /// The register to store the return value in (if there is one).
        ret: Option<Reg>
    }
}
impl ParameterList {
    /// Creates a new standard LC-3 calling convention parameter list.
    pub fn with_calling_convention(params: &[&str]) -> Self {
        let params = params.iter()
            .map(|name| name.to_string())
            .collect();

        Self::CallingConvention { params }
    }

    /// Creates a new pass-by-register parameter list.
    pub fn with_pass_by_register(params: &[(&str, Reg)], ret: Option<Reg>) -> Self {
        let params = params.iter()
            .map(|&(name, reg)| (name.to_string(), reg))
            .collect();

        Self::PassByRegister { params, ret }
    }

    /// Compute the arguments of this parameter list.
    fn get_arguments(&self, regs: &RegFile, mem: &Mem, fp: u16) -> Vec<Word> {
        match self {
            ParameterList::CallingConvention { params } => {
                (0..params.len())
                    .map(|i| fp.wrapping_add(4).wrapping_add(i as u16))
                    .map(|addr| *mem.get_raw(addr))
                    .collect()
            },
            ParameterList::PassByRegister { params, ret: _ } => {
                params.iter()
                    .map(|&(_, r)| regs[r])
                    .collect()
            },
        }
    }
}
impl std::fmt::Debug for ParameterList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CallingConvention { params } => {
                f.write_str("fn[stdcc](")?;
                if let Some((first, rest)) = params.split_first() {
                    f.write_str(first)?;
                    for param in rest {
                        f.write_str(", ")?;
                        f.write_str(param)?;
                    }
                }
                f.write_str(") -> _")?;
                Ok(())
            },
            Self::PassByRegister { params, ret } => {
                f.write_str("fn[pass by reg](")?;
                if let Some(((first_param, first_reg), rest)) = params.split_first() {
                    write!(f, "{first_param} = {first_reg}")?;
                    for (param, reg) in rest {
                        write!(f, ", {param} = {reg}")?;
                    }
                }
                f.write_str(")")?;
                if let Some(ret) = ret {
                    write!(f, " -> {ret}")?;
                }
                Ok(())
            },
        }
    }
}

/// Where this frame came from.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FrameType {
    /// Frame came from a subroutine call.
    Subroutine,
    /// Frame came from a trap call.
    Trap,
    /// Frame came from an interrupt.
    Interrupt
}
/// A frame entry, which defines all the known information about a frame.
/// 
/// This information is only exposed by the Simulator if the `debug_frames` flag is enabled.
#[derive(Debug, Clone)]
pub struct Frame {
    /// The memory location of the caller instruction.
    pub caller_addr: u16,

    /// The memory location of the start of the callee subroutine.
    /// 
    /// For subroutines, this should point to the start of the callee subroutine.
    /// For traps and interrupts, this should point to where the entry exists within their respective tables
    /// (in other words, this value should be `0x00`-`0xFF` for traps and `0x100`-`0x1FF` for interrupts).
    pub callee_addr: u16,

    /// Whether this frame is from a subroutine call, trap call, or interrupt.
    pub frame_type: FrameType,

    /// The address of the current frame pointer.
    /// 
    /// This is only `Some` when:
    /// - the callee's definition is defined, and
    /// - the callee's definition is defined with the standard LC-3 calling convention variant
    pub frame_ptr: Option<Word>,

    /// The arguments of the call.
    /// 
    /// If the callee's definition is defined, this is a list of the arguments used in the call.
    /// Otherwise, this is an empty Vec.
    pub arguments: Vec<Word>
}
/// The stack of call frames.
/// 
/// This struct is used within the Simulator to keep track of the frames of subroutine/trap calls.
/// The amount of information it keeps track of depends on the `debug_frames` flag of the Simulator.
/// - If the `debug_frames` flag is true, this keeps track of a Vec of [`Frame`]s, which contains a large set of information about each frame.
/// - If the `debug_frames` flag is false, this only keeps track of the number of frames traversed.
#[derive(Debug)]
pub struct FrameStack {
    /// The number of frames traversed.
    /// 
    /// At top level execution, `frame_no` == 0.
    /// Every subroutine/trap call (i.e., JSR, JSRR, TRAP instr.) increments this value,
    /// and every return call (i.e., RET, RTI, JMP R7 instr.) decrements this value.
    frame_no: u64,

    /// Function signatures for all traps.
    trap_defns: HashMap<u8, ParameterList>,

    /// Function signatures for all subroutines.
    /// 
    /// The simulator does not compute this by default.
    /// It has to be defined externally by the [`FrameStack::set_subroutine_def`] method.
    sr_defns: HashMap<u16, ParameterList>,

    /// The frames.
    /// 
    /// If `None`, this means frames are not being tracked and frame information is ignored.
    /// If `Some`, frames will be added and removed as subroutines/traps are entered and exited.
    frames: Option<Vec<Frame>>
}

impl FrameStack {
    /// Creates a new frame stack.
    pub(super) fn new(debug_frames: bool) -> Self {
        Self {
            frame_no: 0,
            trap_defns: HashMap::from_iter([
                (0x20, ParameterList::with_pass_by_register(&[], Some(R0))),
                (0x21, ParameterList::with_pass_by_register(&[("char", R0)], None)),
                (0x22, ParameterList::with_pass_by_register(&[("addr", R0)], None)),
                (0x23, ParameterList::with_pass_by_register(&[], Some(R0))),
                (0x24, ParameterList::with_pass_by_register(&[("addr", R0)], None)),
                (0x25, ParameterList::with_pass_by_register(&[], None)),
            ]),
            sr_defns: Default::default(),
            frames: debug_frames.then(Vec::new)
        }
    }

    /// Gets the parameter definition for a trap (if it is defined).
    pub fn get_trap_def(&self, vect: u8) -> Option<&ParameterList> {
        self.trap_defns.get(&vect)
    }
    /// Gets the parameter definition for a subroutine (if it is defined).
    /// 
    /// Note that the simulator does not automatically make subroutine definitions.
    /// Subroutine definitions have to be manually set by the [`FrameStack::set_subroutine_def`] method.
    pub fn get_subroutine_def(&self, addr: u16) -> Option<&ParameterList> {
        self.sr_defns.get(&addr)
    }
    /// Sets the parameter definition for a subroutine.
    /// 
    /// This will overwrite any preexisting definition for a given subroutine.
    pub fn set_subroutine_def(&mut self, addr: u16, params: ParameterList) {
        self.sr_defns.insert(addr, params);
    }
    /// Gets the current number of frames entered.
    pub fn len(&self) -> u64 {
        self.frame_no
    }

    /// Tests whether the frame stack is at top level execution.
    pub fn is_empty(&self) -> bool {
        self.frame_no == 0
    }

    /// Gets the list of current frames (if debug frames are enabled).
    pub fn frames(&self) -> Option<&[Frame]> {
        self.frames.as_deref()
    }

    /// Pushes a new frame to the frame stack.
    /// 
    /// This should be called at the instruction where a subroutine or trap call occurs.
    /// 
    /// Note that the `callee` parameter depends on the type of frame:
    /// - For subroutines, the `callee` parameter represents the start of the subroutine.
    /// - For traps and interrupts, the `callee` parameter represents the vect (0x00-0xFF for traps, 0x100-0x1FF for interrupts).
    pub(super) fn push_frame(&mut self, caller: u16, callee: u16, frame_type: FrameType, regs: &RegFile, mem: &Mem) {
        self.frame_no += 1;
        if let Some(frames) = self.frames.as_mut() {
            let m_plist = match frame_type {
                FrameType::Subroutine => self.sr_defns.get(&callee),
                FrameType::Trap => {
                    u8::try_from(callee).ok()
                        .and_then(|addr| self.trap_defns.get(&addr))
                },
                FrameType::Interrupt => {
                    // FIXME: Interrupt semantics are not well defined.
                    self.sr_defns.get(&callee)
                },
            };
            
            let (fp, args) = match m_plist {
                Some(plist @ ParameterList::CallingConvention { .. }) => {
                    // Strictness: We'll let the simulator handle strictness around this,
                    // because we don't want to trigger an error if frame information is incorrect.
                    let fp = regs[R6] + Word::new_init(4);
                    (Some(fp), plist.get_arguments(regs, mem, fp.get()))
                },
                Some(plist @ ParameterList::PassByRegister { .. }) => {
                    // pass by register doesn't use the fp parameter
                    // so it doesn't matter value is used for the fp arg
                    (None, plist.get_arguments(regs, mem, 0))
                },
                None => (None, vec![])
            };

            frames.push(Frame {
                caller_addr: caller,
                callee_addr: callee,
                frame_type,
                frame_ptr: fp,
                arguments: args,
            })
        }
    }
    
    /// Pops a frame from the frame stack.
    /// 
    /// This should be called at the instruction where a return occurs.
    pub(super) fn pop_frame(&mut self) {
        self.frame_no = self.frame_no.saturating_sub(1);
        if let Some(frames) = self.frames.as_mut() {
            frames.pop();
        }
    }
}
impl Default for FrameStack {
    fn default() -> Self {
        Self::new(false)
    }
}