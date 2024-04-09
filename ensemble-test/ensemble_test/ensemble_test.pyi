class Simulator:
    def __new__(cls): pass
    def init(self, src_fp: str) -> None: pass
    def load(self,
        src_fp: str,
        disable_plugins: bool = False,
        process_debug_comments: bool = True,
        multiple_errors: bool = True,
        enable_warnings: bool = False,
        warnings_as_errors: bool = False
    ) -> None: pass
    def load_code(self, lc3_code: str) -> bool: pass
    def run(self, n: int | None) -> None: pass
    def step(self) -> None: pass
    def back(self) -> None: pass
    def rewind(self, n: int | None) -> None: pass
    def finish(self) -> None: pass
    def next_line(self, n: int | None) -> None: pass
    def prev_line(self, n: int | None) -> None: pass
    def read_mem(self, addr: int) -> int: pass
    def write_mem(self, addr: int, val: int) -> None: pass
    def lookup(self, label: str) -> int: pass
    def reverse_lookup(self, addr: int) -> str: pass
    def add_label(self, label: str, addr: int) -> bool: pass
    def delete_label(self, label: str) -> None: pass
    def get_mem(self, addr: int) -> int: pass
    def set_mem(self, addr: int, val: int) -> None: pass
    def disassemble(self, addr: int, level: int) -> str: pass
    def disassemble_data(self, addr: int, level: int) -> str: pass
    def add_breakpoint_by_addr(
        self, 
        addr: int, 
        condition: str = "1", 
        times: int = -1, 
        label: str = ""
    ) -> bool: pass
    def add_breakpoint_by_label(
        self, 
        label: int, 
        condition: str = "1",
        times: int = -1,
        bp_label: str = ""
    ) -> bool: pass
    def add_watchpoint_by_reg_or_addr(
        self, 
        is_reg: bool, 
        data: int, 
        condition: str = "1", 
        times: int = -1, 
        label: str = ""
    ) -> bool: pass
    def add_watchpoint_by_label(
        self, 
        label: str, 
        condition: str = "1", 
        times: int = -1, 
        wp_label: str = ""
    ) -> bool: pass
    def add_blackbox_by_label(
        self, 
        label: str, 
        condition: str = "1", 
        bb_label: str = ""
    ) -> bool: pass
    def add_blackbox_by_addr(
        self, 
        addr: int, 
        condition: str = "1", 
        bb_label: str = ""
    ) -> bool: pass
    def remove_breakpoint_by_addr(self, addr: int) -> bool: pass
    def remove_breakpoint_by_label(self, label: str) -> bool: pass
    def remove_watchpoint_by_reg_or_addr(self, is_reg: bool, data: int) -> bool: pass
    def remove_watchpoint_by_label(self, label: str) -> bool: pass
    def remove_blackbox_by_addr(self, addr: int) -> bool: pass
    def remove_blackbox_by_label(self, label: str) -> bool: pass
    def add_subroutine_info(self, subroutine_label: str, n_params: int) -> bool: pass
    def seed(self, seed: int) -> None: pass
    def random(self) -> int: pass
    def get_reg(self, index: int) -> int: pass
    def set_reg(self, index: int, val: int) -> None: pass
    
    def has_halted(self) -> None: pass
    def comment(self) -> str: pass
    
    def setup_replay(self, file: str, replay_str: str) -> None: pass
    def describe_replay(self, replay_str: str) -> None: pass

    def first_level_calls(self) -> list[None]: pass
    def first_level_traps(self) -> list[None]: pass

    # Simulator properties
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
    @property
    def memory_ops(self) -> dict[int, None]: pass
    @property
    def breakpoints(self) -> dict[int, None]: pass
    @property
    def blackboxes(self) -> dict[int, None]: pass
    @property
    def memory_watchpoints(self) -> dict[int, None]: pass
    @property
    def register_watchpoints(self) -> dict[int, None]: pass

    @property
    def max_undo_stack_size(self) -> int: pass
    @max_undo_stack_size.setter
    def max_undo_stack_size(self, size: int) -> None: pass
    
    @property
    def max_call_stack_size(self) -> int: pass
    @max_call_stack_size.setter
    def max_call_stack_size(self, size: int) -> None: pass
    
    @property
    def true_traps(self) -> bool: pass
    @true_traps.setter
    def true_traps(self, status: bool) -> None: pass
    
    @property
    def lc3_version(self) -> int: pass
    @lc3_version.setter
    def lc3_version(self, version: int) -> None: pass
    
    @property
    def interrupts(self) -> bool: pass
    @interrupts.setter
    def interrupts(self, status: bool) -> None: pass
    def enable_keyboard_interrupt(self) -> None: pass
    
    @property
    def keyboard_interrupt_delay(self) -> int: pass
    @keyboard_interrupt_delay.setter
    def keyboard_interrupt_delay(self, delay: int) -> None: pass
    
    @property
    def strict_execution(self) -> bool: pass
    @strict_execution.setter
    def strict_execution(self, status: bool) -> None: pass
    
    @property
    def input(self) -> str: pass
    @input.setter
    def input(self, input: str) -> None: pass
    
    @property
    def output(self) -> str: pass
    @output.setter
    def output(self, output: str) -> None: pass
    
    @property
    def warnings(self) -> str: pass
    @warnings.setter
    def warnings(self, warnings: str) -> None: pass

class LoadError(ValueError):
    pass
class SimError(ValueError):
    pass

class MemoryFillStrategy:
    fill_with_value: MemoryFillStrategy
    single_random_value_fill: MemoryFillStrategy
    completely_random: MemoryFillStrategy