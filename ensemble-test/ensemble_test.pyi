class Simulator:
    def __new__(self): pass
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
    def get_n(self) -> bool: pass
    def get_z(self) -> bool: pass
    def get_p(self) -> bool: pass
    def get_pc(self) -> int: pass
    def set_pc(self, addr: int) -> None: pass
    def has_halted(self) -> None: pass
    def get_executions(self) -> int: pass
    def get_memory_ops(self) -> dict[int, None]: pass
    def comment(self) -> str: pass
    def get_breakpoints(self) -> dict[int, None]: pass
    def get_blackboxes(self) -> dict[int, None]: pass
    def get_memory_watchpoints(self) -> dict[int, None]: pass
    def get_register_watchpoints(self) -> dict[int, None]: pass
    def get_max_undo_stack_size(self) -> int: pass
    def set_max_undo_stack_size(self, size: int) -> None: pass
    def get_max_call_stack_size(self) -> int: pass
    def set_max_call_stack_size(self, size: int) -> None: pass
    def get_true_traps(self) -> bool: pass
    def set_true_traps(self, status: bool) -> None: pass
    def get_lc3_version(self) -> int: pass
    def set_lc3_version(self, version: int) -> None: pass
    def get_interrupts(self) -> bool: pass
    def set_interrupts(self, status: bool) -> None: pass
    def enable_keyboard_interrupt(self) -> None: pass
    def get_keyboard_interrupt_delay(self) -> int: pass
    def set_keyboard_intget_keyboard_interrupt_delay(self, delay: int) -> None: pass
    def get_strict_execution(self) -> bool: pass
    def set_strict_execution(self, status: bool) -> None: pass
    def setup_replay(self, file: str, replay_str: str) -> None: pass
    def describe_replay(self, replay_str: str) -> None: pass
    def get_input(self) -> str: pass
    def set_input(self, input: str) -> None: pass
    def get_output(self) -> str: pass
    def set_output(self, output: str) -> None: pass
    def get_warnings(self) -> str: pass
    def set_warnings(self, warnings: str) -> None: pass
    def first_level_calls(self) -> list[None]: pass
    def first_level_traps(self) -> list[None]: pass