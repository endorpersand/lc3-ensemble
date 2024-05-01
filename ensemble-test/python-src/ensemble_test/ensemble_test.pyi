# This is a stub file.
# It is used to provide useful type annotations on IDEs (e.g., VS Code).
#
# It is not automatically updated, so it has to be manually updated
# every time new classes or methods are declared in the Rust backend.
#
# Not properly updating this file doesn't affect its ability to be used
# in a Python project, it's just that there won't be type annotations
# from the IDE.

class Simulator:
    def __new__(cls): pass

    # Initializing machine state
    def init(self, fill: MemoryFillType, value: int | None = None) -> int: pass

    # Loading simulator code
    def load_file(self, src_fp: str) -> None: pass
    def load_code(self, src: str) -> None: pass

    # Simulation
    def run(self, limit: int | None = None) -> None: pass
    def step_in(self) -> None: pass
    def step_out(self) -> None: pass
    def step_over(self) -> None: pass

    # Memory access
    def read_mem(self, addr: int, *, privileged: bool = True, strict: bool = False) -> int: pass
    def write_mem(self, addr: int, val: int, *, privileged: bool = True, strict: bool = False) -> None: pass
    def get_mem(self, addr: int) -> int: pass
    def set_mem(self, addr: int, val: int) -> None: pass

    # Register access
    @property
    def r0(self) -> int: pass
    @r0.setter
    def r0(self, value: int): pass
    @property
    def r1(self) -> int: pass
    @r1.setter
    def r1(self, value: int): pass
    @property
    def r2(self) -> int: pass
    @r2.setter
    def r2(self, value: int): pass
    @property
    def r3(self) -> int: pass
    @r3.setter
    def r3(self, value: int): pass
    @property
    def r4(self) -> int: pass
    @r4.setter
    def r4(self, value: int): pass
    @property
    def r5(self) -> int: pass
    @r5.setter
    def r5(self, value: int): pass
    @property
    def r6(self) -> int: pass
    @r6.setter
    def r6(self, value: int): pass
    @property
    def r7(self) -> int: pass
    @r7.setter
    def r7(self, value: int): pass
    def get_reg(self, index: int) -> int: pass
    def set_reg(self, index: int, val: int) -> None: pass

    # Label lookup
    def lookup(self, label: str) -> int | None: pass
    def reverse_lookup(self, addr: int) -> str | None: pass

    # Breakpoints
    def add_breakpoint(self, break_loc: int | str): pass
    def remove_breakpoint(self, break_loc: int | str): pass
    
    @property
    def breakpoints(self) -> dict[int, None]: pass
    # Miscellaneous access
    @property
    def n(self) -> bool: pass
    @property
    def z(self) -> bool: pass
    @property
    def p(self) -> bool: pass

    @property
    def pc(self) -> int: pass
    @pc.setter
    def pc(self, addr: int) -> None: pass

    @property
    def executions(self) -> int: pass

    # Configuration settings
    @property
    def use_real_halt(self) -> bool: pass
    @use_real_halt.setter
    def use_real_halt(self, status: bool) -> None: pass
    
    @property
    def strict_mem_accesses(self) -> bool: pass
    @strict_mem_accesses.setter
    def strict_mem_accesses(self, status: bool) -> None: pass
    
    # I/O
    @property
    def input(self) -> str: pass
    @input.setter
    def input(self, input: str) -> None: pass
    
    @property
    def output(self) -> str: pass
    @output.setter
    def output(self, output: str) -> None: pass
    
    # from pylc3
    # TODO: determine if these are  necessary
    # @property
    # def max_call_stack_size(self) -> int: pass
    # @max_call_stack_size.setter
    # def max_call_stack_size(self, size: int) -> None: pass

    # def disassemble(self, addr: int, level: int) -> str: pass
    # def disassemble_data(self, addr: int, level: int) -> str: pass
    
    # def add_subroutine_info(self, subroutine_label: str, n_params: int) -> bool: pass
    
    # def first_level_calls(self) -> list[None]: pass
    # def first_level_traps(self) -> list[None]: pass

class LoadError(ValueError):
    pass
class SimError(ValueError):
    pass
class MemoryFillType:
    Random: MemoryFillType
    Single: MemoryFillType