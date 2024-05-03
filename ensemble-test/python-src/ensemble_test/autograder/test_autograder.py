import unittest
from . import LC3UnitTestCase

class TestLC3Sample(LC3UnitTestCase):
    def test_reg(self):
        self.sim.load_code("""
            .orig x3000
            AND R0, R0, #0
            ADD R0, R0, #1
            ADD R0, R0, R0
            ADD R0, R0, R0
            ADD R0, R0, R0
            ADD R0, R0, R0
            ADD R0, R0, R0
            HALT
            .end
        """)
        self.sim.run()
        self.assertReg(0, 32)
    
    def test_value(self):
        self.sim.load_code("""
            .orig x3000
            ONE:   .fill x1111
            TWO:   .fill x2222
            three: .fill x3333
            four:  .fill x4444
            .end
        """)

        self.assertMemValue("ONE",   0x1111)
        self.assertMemValue("two",   0x2222)
        self.assertMemValue("THREE", 0x3333)
        self.assertMemValue("four",  0x4444)
        
        # doesn't exist
        with self.assertRaises(ValueError):
            self.assertMemValue("five", 0x5555)
    
    def test_array(self):
        self.sim.load_code("""
            .orig x3000
            ARRAY: .fill x1234
                   .fill xCDE5
                   .fill xB0F6
                   .fill xA987
            .end
        """)

        self.assertArray("ARRAY", [0x1234, 0xCDE5, 0xB0F6, 0xA987])

        with self.assertRaises(AssertionError):
            self.assertArray("ARRAY", [0x0000, 0x0000, 0x0000, 0x0000])

    def test_str(self):
        self.sim.load_code("""
            .orig x3000
            GOOD_STRINGY: 
                .stringz "HELLO!"
            .end
        """)
        self.assertString("GOOD_STRINGY", "HELLO!")

    def test_str_failures(self):
        self.sim.load_code("""
            .orig x3000
            GOOD_STRINGY: 
                .stringz "HELLO!"
            BAD_STRINGY:
                .stringz "GOODBYE."
            WORSE_STRINGY:
                .fill xF0
                .fill x9F
                .fill x98
                .fill x94
                .fill 0
            .end
        """)
        # non-ascii string test
        with self.assertRaises(ValueError) as e:
            self.assertString("BAD_STRINGY", "\U0001F614")

        # non-byte
        with self.assertRaises(AssertionError) as e:
            self.assertString("WORSE_STRINGY", "HELLO")
        self.assertIn("Found invalid ASCII byte", str(e.exception))

        # mismatch test
        with self.assertRaises(AssertionError) as e:
            self.assertString("BAD_STRINGY", "GOODBYE?")
            self.assertIn("did not match expected", str(e.exception))

        # mismatch test
        with self.assertRaises(AssertionError) as e:
            self.assertString("BAD_STRINGY", "GOOBYEEE")
            self.assertIn("did not match expected", str(e.exception))

        # early cut test
        with self.assertRaises(AssertionError) as e:
            self.assertString("BAD_STRINGY", "GOODBYE...?")
            self.assertIn("shorter than expected", str(e.exception))
        
        # late cut test
        with self.assertRaises(AssertionError) as e:
            self.assertString("BAD_STRINGY", "GOOD")
            self.assertIn("longer than expected", str(e.exception))
        

    def test_output(self):
        self.sim.load_code("""
            .orig x3000
            LD R1, _126
            NOT R1, R1
            ADD R1, R1, #1 ;; R1 = -126
            
            LD R0, _32
            LOOP:
                ADD R2, R0, R1
                BRp ENDLOOP
                PUTC
                ADD R0, R0, #1
            BR LOOP
            ENDLOOP:
            HALT
                           
            _32:  .fill 32
            _126: .fill 126
            .end
        """)
        self.sim.run()
        self.assertConsoleOutput(''.join([chr(i) for i in range(32, 127)]))

    def test_pc(self):
        self.sim.load_code("""
            .orig x3000
                NOP
                NOP
                NOP
                HALT
            .end
        """)

        self.sim.run()
        self.assertPC(0x3003)
    
    def test_cc(self):
        self.sim.load_code("""
            .orig x3000
                AND R0, R0, #0
                ADD R0, R0, #-1
                HALT
            .end
        """)
        self.sim.run()

        # cc failure
        with self.assertRaises(ValueError):
            self.assertCondCode("q") # type: ignore
        
        # cc success
        self.assertCondCode("n")

if __name__ == "__main__":
    unittest.main()