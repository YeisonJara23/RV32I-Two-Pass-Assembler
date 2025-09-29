# Calcula suma = 1 + 2 + ... + n en t0
li a0, 5            # n
li t0, 0            # suma
li t1, 1            # i = 1
loop:
  blt a0, t1, end   # if (i > n) goto end
  add t0, t0, t1    # suma += i
  addi t1, t1, 1    # i++
  beq x0, x0, loop  # goto loop
end:
  ret               # jalr x0, ra, 0
