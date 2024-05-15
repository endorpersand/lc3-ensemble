.orig x3000
    LD R6, STACK_PTR
    AND R0, R0, #0
    ADD R6, R6, #-1
    STR R0, R6, #0
    JSR X
    LDR R0, R6, #0
    ADD R6, R6, #2
    HALT

STACK_PTR .fill xD000
X:
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
    
    LDR R0, R5, #4
    ADD R0, R0, #1
    ADD R6, R6, #-1
    STR R0, R6, #0
    JSR Y
    LDR R0, R6, #0
    ADD R6, R6, #2
    
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

Y:
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
    
    LDR R0, R5, #4
    ADD R0, R0, #1

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

    ;; RET
.end