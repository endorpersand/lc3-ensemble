;; sumtorial(R0):
;;     if R0 <= 0: return 0
;;     return R0 + sumtorial(R0 - 1)

.orig x3000
    LD R6, STACK_FP
    LD R0, N
    JSR SUMTORIAL
    HALT
    
    STACK_FP: .fill xD000
    N: .fill 10

    SUMTORIAL:
        ADD R6, R6, #-1
        STR R7, R6, #0 ; push R7

        ADD R0, R0, #0
        BRp ENDIF
            AND R0, R0, #0
            BR RETURN
        ENDIF:
        ADD R1, R0, #0
        ADD R0, R0, #-1

        ADD R6, R6, #-1
        STR R1, R6, #0 ; push R1
        JSR SUMTORIAL
        LDR R1, R6, #0
        ADD R6, R6, #1 ; pop R1

        ADD R0, R1, R0

        RETURN:
        LDR R7, R6, #0
        ADD R6, R6, #1 ; pop R7
        RET
.end