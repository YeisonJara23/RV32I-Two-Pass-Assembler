Ensamblador RV32I para RISC-V

https://img.shields.io/badge/Python-3.6+-blue.svg
https://img.shields.io/badge/Architecture-RISC--V-green.svg
https://img.shields.io/badge/License-MIT-yellow.svg

Un ensamblador educativo de dos pasadas para el conjunto de instrucciones RV32I de RISC-V, diseñado para facilitar el aprendizaje de arquitectura de computadoras y programación a bajo nivel.

Un ensamblador educativo de dos pasadas para el conjunto de instrucciones RV32I de RISC-V, diseñado para facilitar el aprendizaje de arquitectura de computadoras y programación a bajo nivel.

📋 Tabla de Contenidos
🎯 ¿Qué es un Ensamblador?
🚀 Características
📦 Instalación
🛠️ Uso
📝 Sintaxis del Lenguaje Ensamblador
🔧 Formatos de Salida
📚 Ejemplos
🏗️ Estructura del Proyecto
🤝 Contribuciones
📄 Licencia

🎯 ¿Qué es un Ensamblador?
Concepto Básico
Un ensamblador es un programa que traduce código escrito en lenguaje ensamblador (legible para humanos) a código máquina (binario ejecutable por el procesador).

Analogía: Imagina que el ensamblador es un traductor profesional que convierte instrucciones en español a un idioma que solo la computadora entiende (ceros y unos).

¿Por qué es Necesario?
Los procesadores solo entienden código binario (0s y 1s)

Escribir directamente en binario es extremadamente difícil para los humanos

El lenguaje ensamblador proporciona mnemónicos (abreviaturas) para las instrucciones

El ensamblador automatiza la conversión y evita errores

🚀 Características
✨ Funcionalidades Principales
Característica	Descripción
✅ Soporte RV32I Completo	Todas las instrucciones base del conjunto RV32I
🔄 Dos Pasadas	Análisis robusto con detección temprana de errores
🎭 Pseudoinstrucciones	Sintaxis abreviada para código más legible
🏷️ Manejo de Símbolos	Etiquetas y direcciones simbólicas
📊 Múltiples Salidas	.hex, .bin, .lst en diferentes formatos
🎯 Pseudoinstrucciones Soportadas
Pseudoinstrucción	Instrucciones Reales	Descripción
nop	addi x0, x0, 0	No operation
mv rd, rs	addi rd, rs, 0	Mover registro
not rd, rs	xori rd, rs, -1	Complemento a uno
neg rd, rs	sub rd, x0, rs	Negación
j etiqueta	jal x0, etiqueta	Salto incondicional
ret	jalr x0, x1, 0	Return from subroutine
🔧 Directivas del Ensamblador
Directiva	Descripción
.text	Cambia al segmento de código
.data	Cambia al segmento de datos
.word valor	Reserva una palabra (32 bits)
.byte valor	Reserva un byte
.asciz "cadena"	Cadena terminada en null
.align n	Alinea a 2^n bytes
📦 Instalación
Prerrequisitos
Python 3.6 o superior

No se requieren dependencias externas

🐍 Instalación Rápida
