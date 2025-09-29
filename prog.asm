.text
start:
  nop
  li a0, 42
  addi a1, a0, 1
  beq a1, a0, done
  add  a2, a0, a1
done:
  jalr x0, x1, 0   # ret
