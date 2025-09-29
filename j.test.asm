# 21 bits con signo (±1,048,576 aprox.) y múltiplo de 2.
# 1) Fuera de rango:
# jal x0, 2000000    # ERROR ESPERADO: fuera de 21 bits con signo

# 2) Desalineado:
# jal x0, 3          # ERROR ESPERADO: no múltiplo de 2
nop
