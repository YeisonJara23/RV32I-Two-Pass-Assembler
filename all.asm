.text
  add  x1,x2,x3
  addi x1,x2,10
  sw   x3,16(x2)
  lui  x1,0x12345
  jal  x1, L
  beq  x1,x2, L
L:
  nop
