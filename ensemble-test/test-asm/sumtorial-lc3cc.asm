;; sumtorial(n):
;;     if n <= 0: return 0
;;     return n + sumtorial(n - 1)

.orig x3000
    LD R6, STACK_FP
    LD R0, N

    ADD R6, R6, #-1
    STR R0, R6, #0 ;; push R0
    JSR SUMTORIAL
    LDR R0, R6, #0
    ADD R6, R6, #2
    HALT
    
    STACK_FP: .fill xD000
    N: .fill 10

    SUMTORIAL:
        ADD R6, R6, #-1 ;; alloca RV
        ADD R6, R6, #-1
        STR R7, R6, #0  ;; push R7
        ADD R6, R6, #-1
        STR R5, R6, #0  ;; push R5
        
        ADD R5, R6, #-1 ;; set current FP
        
        ADD R6, R6, #-1
        STR R4, R6, #0  ;; push R4
        ADD R6, R6, #-1
        STR R3, R6, #0  ;; push R3
        ADD R6, R6, #-1
        STR R2, R6, #0  ;; push R2
        ADD R6, R6, #-1
        STR R1, R6, #0  ;; push R1
        ADD R6, R6, #-1
        STR R0, R6, #0  ;; push R0
        
        ;; R0 = ret
        ;; R1 = n
        LDR R1, R5, #4
        BRp ENDIF ;; unless n > 0
            AND R0, R0, #0
            BR RETURN
        ENDIF:
        ADD R0, R1, #0
        ADD R1, R1, #-1
        
        ADD R6, R6, #-1
        STR R1, R6, #0  ;; push arg
        JSR SUMTORIAL
        LDR R1, R6, #0 ;; R1 = sumtorial(n - 1)
        ADD R6, R6, #2
        ADD R0, R0, R1 ;; R0 = n + sumtorial(n - 1)

        RETURN:
        STR R0, R5, #3
        LDR R0, R6, #0
        LDR R1, R6, #1
        LDR R2, R6, #2
        LDR R3, R6, #3
        LDR R4, R6, #4
        ADD R6, R5, #1 ;; pop R0-R4

        LDR R5, R6, #0 ;; pop R5
        ADD R6, R6, #1
        LDR R7, R6, #0 ;; pop R7
        ADD R6, R6, #1

        RET
.end