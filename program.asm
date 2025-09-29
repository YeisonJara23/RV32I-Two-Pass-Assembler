
    # Programa de ejemplo RV32I
    .text
start:
    nop
    li a0, 42          # a0 = 42
    li t0, 0x12345678  # 32-bit immediate via LUI/ADDI
    mv a1, a0
    not a2, a0
    neg a3, a0

    # saltos con etiquetas
    beqz a0, end
    j mid

mid:
    addi a0, a0, -1
    bnez a0, mid

end:
    ret

    .data
msg:
    .asciz "hola"
val:
    .word 0xDEADBEEF
