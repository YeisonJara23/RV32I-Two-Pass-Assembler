#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N1.py — Ensamblador RV32I de dos pasadas con:
  - .text / .data (bases configurables)
  - %hi() / %lo() (opción B) y tokenizador que no los rompe
  - Expansión de pseudoinstrucciones (la/li/mv/not/neg/seqz/.../call/tail)
  - Salidas .hex y .bin en FORMATO TABLA (PC | BIN 32b en nibbles | HEX | ; Fuente)
  - Listado .lst con PC, asm, HEX y campos del formato (R/I/S/B/U/J)

Uso:
  python N1.py program.asm program.hex program.bin
    [--text-base 0x0] [--data-base 0x10000000]
    [--list program.lst] [--show-list N] [--dump-symbols]
    [--raw-hex] [--raw-bin]   # (opcional) salidas planas

Este código es educativo y cubre el conjunto base RV32I sin compresión.
"""

import argparse  # Para procesar argumentos de línea de comandos
import os        # Para operaciones del sistema de archivos
import re        # Para expresiones regulares (procesamiento de texto)
from typing import List, Tuple, Dict  # Para tipado de datos (mejor legibilidad)

# ---------------------------------------------------------------------------
# Utilidades básicas para el ensamblador
# ---------------------------------------------------------------------------

# Diccionario de alias para registros: nombres alternativos -> números de registro
REG_ALIASES = {
    'zero':0, 'ra':1, 'sp':2, 'gp':3, 'tp':4,        # Registros especiales
    't0':5, 't1':6, 't2':7,                          # Registros temporales
    's0':8, 'fp':8, 's1':9,                          # Registros guardados y frame pointer
    'a0':10, 'a1':11, 'a2':12, 'a3':13, 'a4':14, 'a5':15, 'a6':16, 'a7':17,  # Argumentos
    's2':18, 's3':19, 's4':20, 's5':21, 's6':22, 's7':23, 's8':24, 's9':25, 's10':26, 's11':27,  # Guardados
    't3':28, 't4':29, 't5':30, 't6':31,              # Temporales adicionales
}

def reg_number(tok: str) -> int:
    """
    Convierte un token de registro (nombre o número) a su número numérico.
    Ejemplo: 'x5' -> 5, 't0' -> 5, 'a0' -> 10
    """
    tok = tok.strip()  # Elimina espacios en blanco al inicio y final
    if re.fullmatch(r'x([0-9]|[12][0-9]|3[01])', tok):  # Si es formato x0-x31
        return int(tok[1:])  # Extrae el número después de la 'x'
    if tok in REG_ALIASES:  # Si es un alias conocido
        return REG_ALIASES[tok]  # Devuelve el número correspondiente
    raise ValueError(f"Registro inválido: {tok}")  # Error si no es válido

def parse_imm(s: str) -> int:
    """
    Convierte una cadena que representa un número inmediato a entero.
    Soporta decimal, hexadecimal (0x) y binario (0b), con o sin signo.
    """
    s = s.strip().lower().replace('_','')  # Normaliza: minúsculas, sin guiones bajos
    if s.startswith('-0x'): return -int(s[3:],16)  # Hexadecimal negativo
    if s.startswith('0x'):  return  int(s,16)      # Hexadecimal positivo
    if s.startswith('-0b'): return -int(s[3:],2)   # Binario negativo
    if s.startswith('0b'):  return  int(s,2)       # Binario positivo
    return int(s,10)  # Decimal (por defecto)

def sign_extend(value: int, bits: int) -> int:
    """
    Extiende el signo de un número de 'bits' bits a 32 bits.
    Ejemplo: sign_extend(0xFFF, 12) -> 0xFFFFFFFF (para -1 en complemento a 2)
    """
    mask = (1 << bits) - 1  # Máscara para los bits menos significativos
    value &= mask  # Asegura que solo tenemos los bits especificados
    if value & (1 << (bits - 1)):  # Si el bit de signo está activado
        value -= (1 << bits)  # Convierte a negativo en complemento a 2
    return value

def check_range_signed(v: int, bits: int, ctx: str):
    """
    Verifica que un valor con signo esté dentro del rango para 'bits' bits.
    Ejemplo: para 12 bits, rango es -2048 a 2047
    """
    lo = -(1 << (bits-1))  # Límite inferior
    hi = (1 << (bits-1)) - 1  # Límite superior
    if v < lo or v > hi:  # Si está fuera de rango
        raise ValueError(f"Inmediato fuera de rango para {ctx}: {v} no cabe en {bits} bits (signed)")

def check_range_unsigned(v: int, bits: int, ctx: str):
    """
    Verifica que un valor sin signo esté dentro del rango para 'bits' bits.
    Ejemplo: para 5 bits, rango es 0 a 31
    """
    if v < 0 or v > ((1<<bits)-1):  # Si es negativo o excede el máximo
        raise ValueError(f"Valor fuera de rango para {ctx}: {v} no cabe en {bits} bits (unsigned)")

def u32(x:int)->int: 
    """
    Convierte un número a formato de 32 bits sin signo (trunca a 32 bits).
    """
    return x & 0xFFFFFFFF  # Máscara para mantener solo 32 bits

# ---------------------------------------------------------------------------
# Manejo de %hi() / %lo() para direcciones de 32 bits
# ---------------------------------------------------------------------------

# Expresión regular para detectar %hi(etiqueta) o %lo(etiqueta)
RE_HILO = re.compile(r'%(?P<kind>hi|lo)\((?P<sym>[A-Za-z_]\w*)\)$')

def eval_hilo(tok: str, symbols: dict) -> int:
    """
    Evalúa %hi(sym) o %lo(sym) usando la tabla de símbolos.
    %hi(sym) = (dirección + 0x800) >> 12  (parte alta con redondeo para LUI)
    %lo(sym) = dirección[11:0] con signo (parte baja para ADDI/loads/stores)
    """
    m = RE_HILO.fullmatch(tok.strip())  # Intenta hacer match con el patrón %hi/%lo
    if not m:  # Si no coincide
        raise ValueError("no hilo")  # No es un patrón hi/lo
    sym = m.group('sym')  # Extrae el nombre del símbolo
    if sym not in symbols:  # Si el símbolo no está definido
        raise ValueError(f"Etiqueta indefinida: {sym}")
    addr = symbols[sym]  # Obtiene la dirección del símbolo
    kind = m.group('kind')  # 'hi' o 'lo'
    low = addr & 0xFFF  # Extrae los 12 bits bajos (0-11)
    if kind == 'hi':  # Para %hi
        return (addr + 0x800) >> 12  # Parte alta con redondeo
    else:  # Para %lo
        return low - 0x1000 if (low & 0x800) else low  # Parte baja con extensión de signo

# ---------------------------------------------------------------------------
# Mapas de instrucciones RV32I por formato
# ---------------------------------------------------------------------------

# Instrucciones tipo R (register): opcode, funct3, funct7
R_FUNCTS = {
    'add':  ('0110011', '000', '0000000'),  # Suma
    'sub':  ('0110011', '000', '0100000'),  # Resta
    'sll':  ('0110011', '001', '0000000'),  # Shift lógico izquierdo
    'slt':  ('0110011', '010', '0000000'),  # Set less than (con signo)
    'sltu': ('0110011', '011', '0000000'),  # Set less than unsigned
    'xor':  ('0110011', '100', '0000000'),  # XOR
    'srl':  ('0110011', '101', '0000000'),  # Shift lógico derecho
    'sra':  ('0110011', '101', '0100000'),  # Shift aritmético derecho
    'or':   ('0110011', '110', '0000000'),  # OR
    'and':  ('0110011', '111', '0000000'),  # AND
}

# Instrucciones tipo I ALU: opcode, funct3
I_ALU_FUNCTS = {
    'addi': ('0010011', '000'),  # Add immediate
    'slti': ('0010011', '010'),  # Set less than immediate (con signo)
    'sltiu':('0010011', '011'),  # Set less than immediate unsigned
    'xori': ('0010011', '100'),  # XOR immediate
    'ori':  ('0010011', '110'),  # OR immediate
    'andi': ('0010011', '111'),  # AND immediate
}

# Instrucciones tipo I shift: opcode, funct3, funct7
I_SHIFT_FUNCTS = {
    'slli': ('0010011', '001', '0000000'),  # Shift left logical immediate
    'srli': ('0010011', '101', '0000000'),  # Shift right logical immediate
    'srai': ('0010011', '101', '0100000'),  # Shift right arithmetic immediate
}

# Instrucciones de carga: opcode, funct3
LOAD_FUNCTS = {
    'lb':  ('0000011','000'),  # Load byte
    'lh':  ('0000011','001'),  # Load halfword
    'lw':  ('0000011','010'),  # Load word
    'lbu': ('0000011','100'),  # Load byte unsigned
    'lhu': ('0000011','101'),  # Load halfword unsigned
}

# Instrucciones de almacenamiento: opcode, funct3
STORE_FUNCTS = {
    'sb': ('0100011','000'),  # Store byte
    'sh': ('0100011','001'),  # Store halfword
    'sw': ('0100011','010'),  # Store word
}

# Instrucciones de bifurcación: opcode, funct3
BR_FUNCTS = {
    'beq':  ('1100011','000'),  # Branch if equal
    'bne':  ('1100011','001'),  # Branch if not equal
    'blt':  ('1100011','100'),  # Branch if less than (signed)
    'bge':  ('1100011','101'),  # Branch if greater or equal (signed)
    'bltu': ('1100011','110'),  # Branch if less than unsigned
    'bgeu': ('1100011','111'),  # Branch if greater or equal unsigned
}

# Instrucciones tipo U: opcode
U_FUNCTS = { 
    'lui': '0110111',    # Load upper immediate
    'auipc': '0010111'   # Add upper immediate to PC
}

# Instrucciones tipo J: opcode
J_FUNCTS = { 'jal': '1101111' }  # Jump and link

# Instrucciones JALR: opcode, funct3
JALR_FUNCTS = { 'jalr': ('1100111','000') }  # Jump and link register

# ---------------------------------------------------------------------------
# Parser y tokenizador
# ---------------------------------------------------------------------------

# Expresiones regulares para procesamiento de código fuente
COMMENT_RE = re.compile(r'(;|#|//).*')  # Detecta comentarios
LABEL_RE   = re.compile(r'^\s*([A-Za-z_]\w*):')  # Detecta etiquetas

def strip_comment(line:str)->str:
    """
    Elimina comentarios de una línea de código.
    """
    m = COMMENT_RE.search(line)  # Busca patrón de comentario
    return line if not m else line[:m.start()]  # Devuelve línea sin comentario

def tokenize_operands(op_str:str):
    """
    Tokeniza operandos, preservando %hi(sym)/%lo(sym) como un solo token.
    Ejemplo: "t0, %hi(etiqueta), x1" -> ['t0', '%hi(etiqueta)', 'x1']
    """
    raw = [t for t in re.split(r'[,\s()]+', op_str.strip()) if t]  # Divide por comas/espacios/paréntesis
    toks = []  # Lista de tokens resultantes
    i = 0
    while i < len(raw):
        if raw[i] in ('%hi','%lo') and i+1 < len(raw) and re.fullmatch(r'[A-Za-z_]\w*', raw[i+1]):
            # Si encuentra %hi o %lo seguido de un símbolo válido
            toks.append(f"{raw[i]}({raw[i+1]})")  # Combina en un token
            i += 2  # Avanza dos posiciones
        else:
            toks.append(raw[i])  # Agrega token normal
            i += 1  # Avanza una posición
    return toks

# ---------------------------------------------------------------------------
# Representación Intermedia (IR) para líneas de código
# ---------------------------------------------------------------------------

class LineIR:
    """
    Representa una línea de código en la representación intermedia.
    Almacena información necesaria para las dos pasadas del ensamblador.
    """
    def __init__(self, seg, addr, kind, content, raw):
        self.seg = seg          # Segmento: 'text' o 'data'
        self.addr = addr        # Dirección en memoria
        self.kind = kind        # Tipo: 'instr' o 'data'
        self.content = content  # Contenido: (mnemónico, operandos, número línea)
        self.raw = raw          # Línea original (sin comentarios)

# ---------------------------------------------------------------------------
# Clase principal del Ensamblador
# ---------------------------------------------------------------------------

class Assembler:
    """
    Ensamblador RV32I de dos pasadas.
    Pasada 1: Construye tabla de símbolos y representación intermedia.
    Pasada 2: Genera código máquina expandiendo pseudoinstrucciones.
    """
    
    def __init__(self, text_base=0x00000000, data_base=0x10000000):
        """
        Inicializa el ensamblador con bases de texto y datos.
        """
        self.text_base = text_base  # Dirección base del segmento de texto (código)
        self.data_base = data_base  # Dirección base del segmento de datos
        self.reset()  # Inicializa estado interno

    def reset(self):
        """
        Reinicia el estado interno del ensamblador para un nuevo ensamblado.
        """
        self.symbols: Dict[str,int] = {}  # Tabla de símbolos: nombre -> dirección
        self.lines: List[LineIR] = []     # Líneas en representación intermedia
        self.seg = 'text'                 # Segmento actual ('text' o 'data')
        self.pc_text = self.text_base     # Contador de programa para texto
        self.pc_data = 0                  # Contador de programa para datos (relativo)
        self.errors: List[str] = []       # Lista de errores encontrados

    def current_addr(self):
        """
        Devuelve la dirección actual según el segmento activo.
        """
        return self.pc_text if self.seg=='text' else (self.data_base + self.pc_data)

    def bump(self, n):
        """
        Incrementa el contador de programa actual por 'n' bytes.
        """
        if self.seg=='text': 
            self.pc_text += n  # Incrementa PC de texto
        else:                
            self.pc_data += n  # Incrementa PC de datos

    # ---------------------- Pasada 1: Análisis y construcción de símbolos ----------------------
    
    def pass1(self, src_lines):
        """
        Primera pasada: procesa el código fuente, construye tabla de símbolos y IR.
        Retorna True si no hay errores, False en caso contrario.
        """
        self.reset()  # Reinicia estado interno
        for ln, line in enumerate(src_lines, start=1):  # Itera por líneas con número
            raw = strip_comment(line).strip()  # Elimina comentarios y espacios
            if not raw: continue  # Salta líneas vacías

            # Procesa etiquetas al inicio de la línea
            while True:
                m = LABEL_RE.match(raw)  # Busca patrón de etiqueta
                if not m: break  # Si no hay más etiquetas, sale del bucle
                label = m.group(1)  # Extrae nombre de etiqueta
                if label in self.symbols:  # Si ya existe
                    self.errors.append(f"Línea {ln}: Etiqueta duplicada: {label}")
                else:
                    self.symbols[label] = self.current_addr()  # Agrega a tabla de símbolos
                raw = raw[m.end():].lstrip()  # Elimina etiqueta procesada
                if not raw: break  # Si no queda nada, termina
            if not raw: continue  # Si después de etiquetas queda vacío, siguiente línea

            # Procesa directivas del ensamblador
            if raw.startswith('.'):
                d = raw.split()[0].lower()  # Extrae nombre de directiva
                if d == '.text': 
                    self.seg = 'text'  # Cambia a segmento texto
                elif d == '.data': 
                    self.seg = 'data'  # Cambia a segmento datos
                elif d == '.word':
                    args = [v for v in raw[len('.word'):].split(',') if v.strip()]  # Extrae argumentos
                    for _ in args: 
                        self.bump(4)  # Cada palabra ocupa 4 bytes
                elif d == '.byte':
                    args = [v for v in raw[len('.byte'):].split(',') if v.strip()]  # Extrae argumentos
                    for _ in args: 
                        self.bump(1)  # Cada byte ocupa 1 byte
                elif d in ('.asciz','.string'):
                    rest = raw.split(None,1)[1] if len(raw.split())>1 else ''  # Extrae resto de línea
                    m = re.match(r'"(.*)"', rest)  # Busca cadena entre comillas
                    if not m:
                        self.errors.append(f"Línea {ln}: Cadena inválida en {d}")
                    else:
                        s = m.group(1).encode('utf-8') + b'\x00'  # Convierte a bytes + terminador nulo
                        self.bump(len(s))  # Incrementa PC por longitud de cadena
                elif d == '.align':
                    rest = raw.split()  # Divide línea en partes
                    if len(rest)<2:
                        self.errors.append(f"Línea {ln}: .align requiere argumento")
                    else:
                        n = int(rest[1],10)  # Obtiene valor de alineación
                        align = 1<<n  # Calcula alineación (2^n)
                        addr = self.current_addr()  # Dirección actual
                        while addr % align != 0:  # Mientras no esté alineado
                            self.bump(1)  # Añade padding
                            addr += 1
                continue  # Las directivas no generan IR de instrucción

            # Procesa instrucciones y pseudoinstrucciones
            parts = raw.split(None,1)  # Divide en mnemónico y operandos
            mnem = parts[0].lower()  # Mnemónico en minúsculas
            ops  = tokenize_operands(parts[1]) if len(parts)>1 else []  # Tokeniza operandos
            nwords = self.expansion_length(mnem, ops)  # Calcula cuántas palabras ocupa
            # Crea entrada en representación intermedia
            self.lines.append(LineIR(self.seg, self.current_addr(), 'instr', (mnem, ops, ln), raw))
            if self.seg == 'text':  # Solo incrementa PC para segmento texto
                self.bump(4*nwords)  # Cada instrucción ocupa 4 bytes

        return len(self.errors)==0  # True si no hay errores

    def expansion_length(self, mnem, ops) -> int:
        """
        Calcula cuántas palabras de 32 bits ocupa una instrucción después de expandir pseudoinstrucciones.
        """
        if mnem in ('la',):  # load address siempre ocupa 2 instrucciones
            return 2
        if mnem == 'li':  # load immediate
            try:
                v = parse_imm(ops[1])  # Intenta parsear el inmediato
                return 1 if -2048<=v<=2047 else 2  # 1 si cabe en 12 bits, 2 si no
            except Exception: 
                return 2  # Si hay error, asume 2 instrucciones
        # Cargas/almacenamientos con sintaxis abreviada ocupan 2 instrucciones
        if (mnem in ('lb','lh','lw','lbu','lhu') and len(ops)==2) or \
           (mnem in ('sb','sh','sw')  and len(ops)==2):
            return 2
        # Pseudoinstrucciones que se expanden a 1 instrucción real
        if mnem in ('nop','mv','not','neg','seqz','snez','sltz','sgtz',
                    'beqz','bnez','blez','bgez','bltz','bgtz',
                    'bgt','ble','bgtu','bleu','j','jal','jr','jalr','ret','call','tail'):
            return 1
        return 1  # Por defecto, 1 instrucción

    # ---------------------- Pasada 2: Generación de código máquina ----------------------
    
    def pass2(self):
        """
        Segunda pasada: genera código máquina a partir de la representación intermedia.
        Retorna lista de tuplas (dirección, palabra, mnemónico, operandos, línea original).
        """
        machine: List[Tuple[int,int,str,List[str],str]] = []  # Código máquina resultante
        for item in self.lines:  # Itera por líneas en IR
            if item.kind!='instr' or item.seg!='text':  # Solo instrucciones en segmento texto
                continue
            mnem, ops, ln = item.content  # Extrae mnemónico, operandos y número de línea
            try:
                # Expande pseudoinstrucciones recursivamente
                for base_mnem, base_ops in self.expand_recursive(mnem, ops, item.addr):
                    # Codifica instrucción base a palabra de 32 bits
                    word = self.encode_base(base_mnem, base_ops, item.addr)
                    # Agrega a lista de código máquina
                    machine.append((item.addr, word, base_mnem, base_ops, item.raw.strip()))
                    item.addr += 4  # Avanza a siguiente dirección
            except Exception as e:
                self.errors.append(f"Línea {ln}: {e}")  # Captura y registra errores
        return machine

    def expand_recursive(self, mnem, ops, pc):
        """
        Expande pseudoinstrucciones recursivamente hasta obtener solo instrucciones base.
        """
        work = [(mnem,ops)]  # Cola de trabajo: (mnemónico, operandos)
        out=[]  # Lista de salida: instrucciones base
        while work:  # Mientras haya trabajo pendiente
            m,o = work.pop(0)  # Toma primer elemento
            if self.is_base(m):  # Si es instrucción base
                out.append((m,o))  # Agrega a salida
            else:  # Si es pseudoinstrucción
                # Expande y agrega resultado al inicio de la cola
                work = self.expand_pseudo(m,o,pc) + work
        return out

    def is_base(self, m):
        """
        Verifica si un mnemónico es una instrucción base RV32I.
        """
        return (m in R_FUNCTS or m in I_ALU_FUNCTS or m in I_SHIFT_FUNCTS or
                m in LOAD_FUNCTS or m in STORE_FUNCTS or m in BR_FUNCTS or
                m in U_FUNCTS or m in J_FUNCTS or m in JALR_FUNCTS or
                m in ('ecall','ebreak','la.hi','la.lo'))  # Pseudoinstrucciones internas

    def expand_pseudo(self, m, o, pc):
        """
        Expande una pseudoinstrucción a una o más instrucciones base.
        Retorna lista de tuplas (mnemónico_base, operandos_base).
        """
        # Pseudoinstrucciones simples de un solo registro
        if m=='nop': 
            return [('addi',['x0','x0','0'])]  # addi x0, x0, 0
        if m=='mv':
            if len(o)!=2: 
                raise ValueError("mv espera: rd, rs")
            return [('addi',[o[0],o[1],'0'])]  # addi rd, rs, 0
        if m=='not':
            if len(o)!=2: 
                raise ValueError("not espera: rd, rs")
            return [('xori',[o[0],o[1],'-1'])]  # xori rd, rs, -1
        if m=='neg':
            if len(o)!=2: 
                raise ValueError("neg espera: rd, rs")
            return [('sub',[o[0],'x0',o[1]])]  # sub rd, x0, rs
        if m=='seqz':
            if len(o)!=2: 
                raise ValueError("seqz espera: rd, rs")
            return [('sltiu',[o[0],o[1],'1'])]  # sltiu rd, rs, 1
        if m=='snez':
            if len(o)!=2: 
                raise ValueError("snez espera: rd, rs")
            return [('sltu',[o[0],'x0',o[1]])]  # sltu rd, x0, rs
        if m=='sltz':
            if len(o)!=2: 
                raise ValueError("sltz espera: rd, rs")
            return [('slt',[o[0],o[1],'x0'])]  # slt rd, rs, x0
        if m=='sgtz':
            if len(o)!=2: 
                raise ValueError("sgtz espera: rd, rs")
            return [('slt',[o[0],'x0',o[1]])]  # slt rd, x0, rs

        # Ramas condicionales abreviadas (un registro)
        if m=='beqz':
            if len(o)!=2: 
                raise ValueError("beqz espera: rs, etiq/disp")
            return [('beq',[o[0],'x0',o[1]])]  # beq rs, x0, destino
        if m=='bnez':
            if len(o)!=2: 
                raise ValueError("bnez espera: rs, etiq/disp")
            return [('bne',[o[0],'x0',o[1]])]  # bne rs, x0, destino
        if m=='blez':
            if len(o)!=2: 
                raise ValueError("blez espera: rs, etiq/disp")
            return [('bge',['x0',o[0],o[1]])]  # bge x0, rs, destino
        if m=='bgez':
            if len(o)!=2: 
                raise ValueError("bgez espera: rs, etiq/disp")
            return [('bge',[o[0],'x0',o[1]])]  # bge rs, x0, destino
        if m=='bltz':
            if len(o)!=2: 
                raise ValueError("bltz espera: rs, etiq/disp")
            return [('blt',[o[0],'x0',o[1]])]  # blt rs, x0, destino
        if m=='bgtz':
            if len(o)!=2: 
                raise ValueError("bgtz espera: rs, etiq/disp")
            return [('blt',['x0',o[0],o[1]])]  # blt x0, rs, destino

        # Ramas condicionales con operandos intercambiados
        if m=='bgt':
            if len(o)!=3: 
                raise ValueError("bgt espera: rs, rt, etiq/disp")
            return [('blt',[o[1],o[0],o[2]])]  # blt rt, rs, destino
        if m=='ble':
            if len(o)!=3: 
                raise ValueError("ble espera: rs, rt, etiq/disp")
            return [('bge',[o[1],o[0],o[2]])]  # bge rt, rs, destino
        if m=='bgtu':
            if len(o)!=3: 
                raise ValueError("bgtu espera: rs, rt, etiq/disp")
            return [('bltu',[o[1],o[0],o[2]])]  # bltu rt, rs, destino
        if m=='bleu':
            if len(o)!=3: 
                raise ValueError("bleu espera: rs, rt, etiq/disp")
            return [('bgeu',[o[1],o[0],o[2]])]  # bgeu rt, rs, destino

        # Saltos abreviados
        if m=='j':
            if len(o)!=1: 
                raise ValueError("j espera: etiq/disp")
            return [('jal',['x0',o[0]])]  # jal x0, destino (solo salto)
        if m=='jal' and len(o)==1:  # jal sin registro de retorno explícito
            return [('jal',['x1',o[0]])]  # jal x1, destino (call)
        if m=='jr':
            if len(o)!=1: 
                raise ValueError("jr espera: rs")
            return [('jalr',['x0',o[0],'0'])]  # jalr x0, rs, 0
        if m=='jalr' and len(o)==1:  # jalr sin registro de retorno explícito
            return [('jalr',['x1',o[0],'0'])]  # jalr x1, rs, 0
        if m=='ret': 
            return [('jalr',['x0','x1','0'])]  # jalr x0, x1, 0 (return)
        if m=='call':
            if len(o)!=1: 
                raise ValueError("call espera: offset12")
            return [('jalr',['x1','x1',o[0]])]  # jalr x1, x1, offset
        if m=='tail':
            if len(o)!=1: 
                raise ValueError("tail espera: offset12")
            return [('jalr',['x0','x6',o[0]])]  # jalr x0, x6, offset (tail call)

        # Load immediate (li) - carga de constantes de 32 bits
        if m=='li':
            if len(o)!=2: 
                raise ValueError("li espera: rd, imm32")
            rd=o[0]  # Registro destino
            imm=parse_imm(o[1])  # Valor inmediato
            if -2048<=imm<=2047:  # Si cabe en 12 bits con signo
                return [('addi',[rd,'x0',str(imm)])]  # addi rd, x0, imm
            # Para valores más grandes: LUI + ADDI
            low=imm & 0xFFF  # 12 bits bajos
            if low & 0x800:  # Si el bit 11 es 1 (negativo en 12 bits)
                high=(imm+0x1000)>>12  # Ajusta parte alta por carry
                low=low-0x1000  # Ajusta parte baja
            else:
                high=imm>>12  # Parte alta directa
                low=sign_extend(low,12)  # Parte baja con signo extendido
            return [('lui',[rd,str(high)]),('addi',[rd,rd,str(low)])]  # LUI + ADDI

        # Load address (la) - carga de dirección de símbolo
        if m=='la':
            if len(o)!=2: 
                raise ValueError("la espera: rd, simbolo")
            return [('la.hi',[o[0],o[1]]),('la.lo',[o[0],o[1]])]  # Parte alta + parte baja

        # Cargas/almacenamientos con sintaxis abreviada (sin registro base explícito)
        if m in ('lb','lh','lw','lbu','lhu') and len(o)==2:
            rd,sym=o  # Registro destino y símbolo
            return [('la',[rd,sym]),(m,[rd,f'0({rd})'])]  # la + load con base
        if m in ('sb','sh','sw') and len(o)==2:
            rs,sym=o  # Registro fuente y símbolo
            return [('la',['t3',sym]),(m,[rs,f'0(t3)'])]  # la + store con base

        return [(m,o)]  # Por defecto, no expande (instrucción base)

    # ---------------------- Codificación de instrucciones base ----------------------
    
    def encode_base(self, mnem, ops, pc):
        """
        Codifica una instrucción base RV32I a palabra binaria de 32 bits.
        """
        # Tipo R: instrucciones con registros (add, sub, and, or, etc.)
        if mnem in R_FUNCTS:
            if len(ops)!=3: 
                raise ValueError(f"{mnem} espera 3 operandos")
            rd,rs1,rs2 = map(reg_number,ops)  # Convierte operandos a números de registro
            opcode,f3,f7 = R_FUNCTS[mnem]  # Obtiene códigos de función
            # Construye palabra: f7[5:0]|rs2|rs1|f3|rd|opcode[6:0]
            return (int(f7,2)<<25) | (rs2<<20) | (rs1<<15) | (int(f3,2)<<12) | (rd<<7) | int(opcode,2)

        # Tipo I ALU: instrucciones con inmediato (addi, andi, ori, etc.)
        if mnem in I_ALU_FUNCTS:
            if len(ops)!=3: 
                raise ValueError(f"{mnem} espera 3 operandos")
            rd,rs1,imm = ops  # Extrae operandos
            rd=reg_number(rd)  # Convierte registro destino
            rs1=reg_number(rs1)  # Convierte registro fuente 1
            # Intenta evaluar inmediato (puede ser %hi/%lo)
            try: immv = eval_hilo(imm,self.symbols)  # Primero intenta %hi/%lo
            except: immv = parse_imm(imm)  # Si falla, parsea como número normal
            check_range_signed(immv,12,f"{mnem} imm")  # Verifica rango
            opcode,f3 = I_ALU_FUNCTS[mnem]  # Obtiene códigos de función
            # Construye palabra: imm[11:0]|rs1|f3|rd|opcode[6:0]
            return (sign_extend(immv,12)<<20) | (rs1<<15) | (int(f3,2)<<12) | (rd<<7) | int(opcode,2)

        # Tipo I shift: desplazamientos con inmediato (slli, srli, srai)
        if mnem in I_SHIFT_FUNCTS:
            if len(ops)!=3: 
                raise ValueError(f"{mnem} espera 3 operandos")
            rd,rs1,imm = ops  # Extrae operandos
            rd=reg_number(rd)  # Convierte registro destino
            rs1=reg_number(rs1)  # Convierte registro fuente 1
            immv=parse_imm(imm)  # Parse inmediato de desplazamiento
            check_range_unsigned(immv,5,f"{mnem} shamt")  # Verifica rango (0-31)
            opcode,f3,f7 = I_SHIFT_FUNCTS[mnem]  # Obtiene códigos de función
            # Construye palabra: f7[5:0]|shamt|rs1|f3|rd|opcode[6:0]
            return (int(f7,2)<<25) | (immv<<20) | (rs1<<15) | (int(f3,2)<<12) | (rd<<7) | int(opcode,2)

        # Instrucciones de carga (lb, lh, lw, lbu, lhu)
        if mnem in LOAD_FUNCTS:
            if len(ops)!=2: 
                raise ValueError(f"{mnem} espera 2 operandos: rd, offset(rs1)")
            rd,rest=ops  # Registro destino y resto
            m=re.match(r'(\-?\w+)\((\w+)\)$',rest)  # Patrón: offset(registro)
            if not m: 
                raise ValueError(f"Formato inválido: {rest}, espera offset(rs1)")
            offset,rs1 = m.groups()  # Extrae offset y registro base
            rd=reg_number(rd)  # Convierte registro destino
            rs1=reg_number(rs1)  # Convierte registro base
            # Intenta evaluar offset (puede ser %hi/%lo)
            try: immv = eval_hilo(offset,self.symbols)  # Primero intenta %hi/%lo
            except: immv = parse_imm(offset)  # Si falla, parsea como número normal
            check_range_signed(immv,12,f"{mnem} offset")  # Verifica rango
            opcode,f3 = LOAD_FUNCTS[mnem]  # Obtiene códigos de función
            # Construye palabra: imm[11:0]|rs1|f3|rd|opcode[6:0]
            return (sign_extend(immv,12)<<20) | (rs1<<15) | (int(f3,2)<<12) | (rd<<7) | int(opcode,2)

        # Instrucciones de almacenamiento (sb, sh, sw)
        if mnem in STORE_FUNCTS:
            if len(ops)!=2: 
                raise ValueError(f"{mnem} espera 2 operandos: rs2, offset(rs1)")
            rs2,rest=ops  # Registro fuente y resto
            m=re.match(r'(\-?\w+)\((\w+)\)$',rest)  # Patrón: offset(registro)
            if not m: 
                raise ValueError(f"Formato inválido: {rest}, espera offset(rs1)")
            offset,rs1 = m.groups()  # Extrae offset y registro base
            rs2=reg_number(rs2)  # Convierte registro fuente
            rs1=reg_number(rs1)  # Convierte registro base
            # Intenta evaluar offset (puede ser %hi/%lo)
            try: immv = eval_hilo(offset,self.symbols)  # Primero intenta %hi/%lo
            except: immv = parse_imm(offset)  # Si falla, parsea como número normal
            check_range_signed(immv,12,f"{mnem} offset")  # Verifica rango
            opcode,f3 = STORE_FUNCTS[mnem]  # Obtiene códigos de función
            # Construye palabra: imm[11:5]|rs2|rs1|f3|imm[4:0]|opcode[6:0]
            return ((immv>>5)<<25) | (rs2<<20) | (rs1<<15) | (int(f3,2)<<12) | ((immv&0x1F)<<7) | int(opcode,2)

        # Instrucciones de bifurcación (beq, bne, blt, bge, etc.)
        if mnem in BR_FUNCTS:
            if len(ops)!=3: 
                raise ValueError(f"{mnem} espera 3 operandos")
            rs1,rs2,label=ops  # Registros y etiqueta destino
            rs1=reg_number(rs1)  # Convierte registro 1
            rs2=reg_number(rs2)  # Convierte registro 2
            # Calcula desplazamiento desde PC actual
            if label in self.symbols:
                offset = self.symbols[label] - pc  # Diferencia con dirección destino
            else:
                offset = parse_imm(label)  # Si no es símbolo, parsea como número
            check_range_signed(offset,13,f"{mnem} offset")  # Verifica rango
            if offset & 1: 
                raise ValueError(f"Offset de salto debe ser par: {offset}")
            opcode,f3 = BR_FUNCTS[mnem]  # Obtiene códigos de función
            # Construye palabra: imm[12]|imm[10:5]|rs2|rs1|f3|imm[4:1]|imm[11]|opcode[6:0]
            return ((offset&0x1000)<<19) | ((offset&0x7E0)<<20) | (rs2<<20) | (rs1<<15) | (int(f3,2)<<12) | ((offset&0x1E)<<7) | ((offset&0x800)>>4) | int(opcode,2)

        # Instrucciones tipo U (lui, auipc)
        if mnem in U_FUNCTS:
            if len(ops)!=2: 
                raise ValueError(f"{mnem} espera 2 operandos")
            rd,imm=ops  # Registro destino e inmediato
            rd=reg_number(rd)  # Convierte registro destino
            # Intenta evaluar inmediato (puede ser %hi/%lo)
            try: immv = eval_hilo(imm,self.symbols)  # Primero intenta %hi/%lo
            except: immv = parse_imm(imm)  # Si falla, parsea como número normal
            check_range_unsigned(immv,20,f"{mnem} imm")  # Verifica rango
            opcode = U_FUNCTS[mnem]  # Obtiene código de operación
            # Construye palabra: imm[31:12]|rd|opcode[6:0]
            return ((immv&0xFFFFF)<<12) | (rd<<7) | int(opcode,2)

        # Instrucciones tipo J (jal)
        if mnem in J_FUNCTS:
            if len(ops)!=2: 
                raise ValueError(f"{mnem} espera 2 operandos")
            rd,label=ops  # Registro destino y etiqueta
            rd=reg_number(rd)  # Convierte registro destino
            # Calcula desplazamiento desde PC actual
            if label in self.symbols:
                offset = self.symbols[label] - pc  # Diferencia con dirección destino
            else:
                offset = parse_imm(label)  # Si no es símbolo, parsea como número
            check_range_signed(offset,21,f"{mnem} offset")  # Verifica rango
            if offset & 1: 
                raise ValueError(f"Offset de salto debe ser par: {offset}")
            opcode = J_FUNCTS[mnem]  # Obtiene código de operación
            # Construye palabra: imm[20]|imm[10:1]|imm[11]|imm[19:12]|rd|opcode[6:0]
            return ((offset&0x100000)<<11) | ((offset&0x7FE)<<20) | ((offset&0x800)<<9) | ((offset&0xFF000)>>12) | (rd<<7) | int(opcode,2)

        # Instrucciones JALR
        if mnem in JALR_FUNCTS:
            if len(ops)!=3: 
                raise ValueError(f"{mnem} espera 3 operandos")
            rd,rs1,imm=ops  # Registro destino, base e inmediato
            rd=reg_number(rd)  # Convierte registro destino
            rs1=reg_number(rs1)  # Convierte registro base
            # Intenta evaluar inmediato (puede ser %hi/%lo)
            try: immv = eval_hilo(imm,self.symbols)  # Primero intenta %hi/%lo
            except: immv = parse_imm(imm)  # Si falla, parsea como número normal
            check_range_signed(immv,12,f"{mnem} offset")  # Verifica rango
            opcode,f3 = JALR_FUNCTS[mnem]  # Obtiene códigos de función
            # Construye palabra: imm[11:0]|rs1|f3|rd|opcode[6:0]
            return (sign_extend(immv,12)<<20) | (rs1<<15) | (int(f3,2)<<12) | (rd<<7) | int(opcode,2)

        # Instrucciones del sistema (ecall, ebreak)
        if mnem=='ecall': 
            return 0x00000073  # Código fijo para ecall
        if mnem=='ebreak': 
            return 0x00100073  # Código fijo para ebreak

        # Pseudoinstrucciones internas para load address
        if mnem=='la.hi':
            if len(ops)!=2: 
                raise ValueError("la.hi espera: rd, simbolo")
            rd,sym=ops  # Registro destino y símbolo
            rd=reg_number(rd)  # Convierte registro destino
            if sym not in self.symbols:  # Verifica que símbolo exista
                raise ValueError(f"Etiqueta indefinida: {sym}")
            addr = self.symbols[sym]  # Obtiene dirección del símbolo
            immv = (addr + 0x800) >> 12  # Calcula parte alta con redondeo
            return ((immv&0xFFFFF)<<12) | (rd<<7) | int(U_FUNCTS['lui'],2)  # LUI rd, %hi(sym)

        if mnem=='la.lo':
            if len(ops)!=2: 
                raise ValueError("la.lo espera: rd, simbolo")
            rd,sym=ops  # Registro destino y símbolo
            rd=reg_number(rd)  # Convierte registro destino
            if sym not in self.symbols:  # Verifica que símbolo exista
                raise ValueError(f"Etiqueta indefinida: {sym}")
            addr = self.symbols[sym]  # Obtiene dirección del símbolo
            low = addr & 0xFFF  # Extrae 12 bits bajos
            if low & 0x800:  # Si bit 11 es 1 (negativo)
                low = low - 0x1000  # Ajusta para extensión de signo
            return (sign_extend(low,12)<<20) | (rd<<15) | (0<<12) | (rd<<7) | int(I_ALU_FUNCTS['addi'][0],2)  # ADDI rd, rd, %lo(sym)

        raise ValueError(f"Instrucción no reconocida: {mnem}")  # Error si no reconoce

# ---------------------------------------------------------------------------
# Funciones de salida/formateo
# ---------------------------------------------------------------------------

def format_bin32(w):
    """
    Formatea una palabra de 32 bits como 8 nibbles hexadecimales.
    """
    return f"{w:08X}"  # Convierte a hexadecimal de 8 dígitos

def format_fields(w):
    """
    Descompone una instrucción en sus campos según el formato.
    """
    opcode = w & 0x7F  # Bits 0-6: opcode
    rd     = (w>>7)&0x1F  # Bits 7-11: registro destino
    funct3 = (w>>12)&0x7  # Bits 12-14: función 3
    rs1    = (w>>15)&0x1F  # Bits 15-19: registro fuente 1
    rs2    = (w>>20)&0x1F  # Bits 20-24: registro fuente 2
    funct7 = (w>>25)&0x7F  # Bits 25-31: función 7
    return f"op:{opcode:02x} rd:{rd} f3:{funct3} rs1:{rs1} rs2:{rs2} f7:{funct7:02x}"

def main():
    """
    Función principal: procesa argumentos y ejecuta el ensamblador.
    """
    # Configura parser de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Ensamblador RV32I')
    parser.add_argument('input', help='Archivo de entrada .asm')
    parser.add_argument('hexout', nargs='?', help='Archivo de salida .hex')
    parser.add_argument('binout', nargs='?', help='Archivo de salida .bin')
    parser.add_argument('--text-base', type=lambda x: int(x,0), default=0x00000000, help='Base del segmento .text')
    parser.add_argument('--data-base', type=lambda x: int(x,0), default=0x10000000, help='Base del segmento .data')
    parser.add_argument('--list', help='Genera listado .lst')
    parser.add_argument('--show-list', type=int, default=0, metavar='N', help='Muestra primeras N líneas del listado')
    parser.add_argument('--dump-symbols', action='store_true', help='Muestra tabla de símbolos')
    parser.add_argument('--raw-hex', action='store_true', help='Salida .hex plana (solo palabras)')
    parser.add_argument('--raw-bin', action='store_true', help='Salida .bin plana (solo bytes)')
    args = parser.parse_args()  # Parsea argumentos

    # Lee archivo de entrada
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            src_lines = f.readlines()  # Lee todas las líneas
    except Exception as e:
        print(f"Error leyendo {args.input}: {e}")  # Error si no puede leer
        return 1

    # Crea ensamblador y ejecuta primera pasada
    asm = Assembler(text_base=args.text_base, data_base=args.data_base)
    if not asm.pass1(src_lines):  # Si hay errores en pasada 1
        print("Errores en Pasada 1:")  # Muestra errores
        for err in asm.errors: print(" ", err)
        return 1

    # Muestra tabla de símbolos si se solicitó
    if args.dump_symbols:
        print("Tabla de símbolos:")
        for k,v in sorted(asm.symbols.items()):
            print(f"  {k:20s} : 0x{v:08X}")

    # Ejecuta segunda pasada para generar código máquina
    machine = asm.pass2()  # Genera código máquina
    if asm.errors:  # Si hay errores en pasada 2
        print("Errores en Pasada 2:")  # Muestra errores
        for err in asm.errors: print(" ", err)
        return 1

    # Genera archivo de listado si se solicitó
    if args.list or args.show_list>0:
        lst_lines = []  # Líneas del listado
        for addr,word,mnem,ops,raw in machine:  # Itera por instrucciones
            hexw = format_bin32(word)  # Palabra en hexadecimal
            binw = f"{word:032b}"  # Palabra en binario
            fields = format_fields(word)  # Campos decodificados
            # Formato: dirección | binario | hexadecimal | instrucción fuente
            lst_lines.append(f"{addr:08X} | {binw} | {hexw} | ; {raw} [{fields}]")
        # Muestra primeras líneas si se solicitó
        if args.show_list>0:
            print(f"Listado (primeras {args.show_list} líneas):")
            for i in range(min(args.show_list, len(lst_lines))):
                print(" ", lst_lines[i])
        # Escribe archivo .lst
        if args.list:
            try:
                with open(args.list, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lst_lines)+'\n')  # Escribe todas las líneas
                print(f"Listado guardado en {args.list}")
            except Exception as e:
                print(f"Error escribiendo {args.list}: {e}")

    # Genera salida .hex
    if args.hexout:
        try:
            with open(args.hexout, 'w', encoding='utf-8') as f:
                if args.raw_hex:  # Si es formato raw (solo palabras)
                    for addr,word,mnem,ops,raw in machine:
                        f.write(f"{word:08X}\n")  # Una palabra por línea
                else:  # Formato tabla completo
                    for addr,word,mnem,ops,raw in machine:
                        binw = f"{word:032b}"  # Binario de 32 bits
                        hexw = format_bin32(word)  # Hexadecimal
                        # Formato: PC | binario | hex | ; instrucción fuente
                        f.write(f"{addr:08X} | {binw} | {hexw} | ; {raw}\n")
            print(f"Salida .hex guardada en {args.hexout}")
        except Exception as e:
            print(f"Error escribiendo {args.hexout}: {e}")

    # Genera salida .bin
    if args.binout:
        try:
            with open(args.binout, 'wb') as f:  # Abre en modo binario
                for addr,word,mnem,ops,raw in machine:
                    # Convierte palabra de 32 bits a bytes (little-endian)
                    b = word.to_bytes(4, byteorder='little', signed=False)
                    f.write(b)  # Escribe bytes
            print(f"Salida .bin guardada en {args.binout}")
        except Exception as e:
            print(f"Error escribiendo {args.binout}: {e}")

    return 0  # Éxito

# Punto de entrada del programa
if __name__ == '__main__':
    exit(main())  # Ejecuta función principal y termina con código de salida