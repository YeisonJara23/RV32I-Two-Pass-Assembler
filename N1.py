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

import argparse
import os
import re
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

REG_ALIASES = {
    'zero':0, 'ra':1, 'sp':2, 'gp':3, 'tp':4,
    't0':5, 't1':6, 't2':7,
    's0':8, 'fp':8, 's1':9,
    'a0':10, 'a1':11, 'a2':12, 'a3':13, 'a4':14, 'a5':15, 'a6':16, 'a7':17,
    's2':18, 's3':19, 's4':20, 's5':21, 's6':22, 's7':23, 's8':24, 's9':25, 's10':26, 's11':27,
    't3':28, 't4':29, 't5':30, 't6':31,
}

def reg_number(tok: str) -> int:
    tok = tok.strip()
    if re.fullmatch(r'x([0-9]|[12][0-9]|3[01])', tok):
        return int(tok[1:])
    if tok in REG_ALIASES:
        return REG_ALIASES[tok]
    raise ValueError(f"Registro inválido: {tok}")

def parse_imm(s: str) -> int:
    s = s.strip().lower().replace('_','')
    if s.startswith('-0x'): return -int(s[3:],16)
    if s.startswith('0x'):  return  int(s,16)
    if s.startswith('-0b'): return -int(s[3:],2)
    if s.startswith('0b'):  return  int(s,2)
    return int(s,10)

def sign_extend(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    value &= mask
    if value & (1 << (bits - 1)):
        value -= (1 << bits)
    return value

def check_range_signed(v: int, bits: int, ctx: str):
    lo = -(1 << (bits-1)); hi = (1 << (bits-1)) - 1
    if v < lo or v > hi:
        raise ValueError(f"Inmediato fuera de rango para {ctx}: {v} no cabe en {bits} bits (signed)")

def check_range_unsigned(v: int, bits: int, ctx: str):
    if v < 0 or v > ((1<<bits)-1):
        raise ValueError(f"Valor fuera de rango para {ctx}: {v} no cabe en {bits} bits (unsigned)")

def u32(x:int)->int: return x & 0xFFFFFFFF

# ---------------------------------------------------------------------------
# %hi() / %lo()
# ---------------------------------------------------------------------------

RE_HILO = re.compile(r'%(?P<kind>hi|lo)\((?P<sym>[A-Za-z_]\w*)\)$')

def eval_hilo(tok: str, symbols: dict) -> int:
    """
    %hi(sym) = (addr + 0x800) >> 12   (redondeo para LUI)
    %lo(sym) = addr[11:0] con signo (para ADDI/offsets de 12 bits)
    """
    m = RE_HILO.fullmatch(tok.strip())
    if not m:
        raise ValueError("no hilo")
    sym = m.group('sym')
    if sym not in symbols:
        raise ValueError(f"Etiqueta indefinida: {sym}")
    addr = symbols[sym]
    kind = m.group('kind')
    low = addr & 0xFFF
    if kind == 'hi':
        return (addr + 0x800) >> 12
    else:
        return low - 0x1000 if (low & 0x800) else low

# ---------------------------------------------------------------------------
# Mapas RV32I
# ---------------------------------------------------------------------------

R_FUNCTS = {
    'add':  ('0110011', '000', '0000000'),
    'sub':  ('0110011', '000', '0100000'),
    'sll':  ('0110011', '001', '0000000'),
    'slt':  ('0110011', '010', '0000000'),
    'sltu': ('0110011', '011', '0000000'),
    'xor':  ('0110011', '100', '0000000'),
    'srl':  ('0110011', '101', '0000000'),
    'sra':  ('0110011', '101', '0100000'),
    'or':   ('0110011', '110', '0000000'),
    'and':  ('0110011', '111', '0000000'),
}
I_ALU_FUNCTS = {
    'addi': ('0010011', '000'),
    'slti': ('0010011', '010'),
    'sltiu':('0010011', '011'),
    'xori': ('0010011', '100'),
    'ori':  ('0010011', '110'),
    'andi': ('0010011', '111'),
}
I_SHIFT_FUNCTS = {
    'slli': ('0010011', '001', '0000000'),
    'srli': ('0010011', '101', '0000000'),
    'srai': ('0010011', '101', '0100000'),
}
LOAD_FUNCTS = {
    'lb':  ('0000011','000'),
    'lh':  ('0000011','001'),
    'lw':  ('0000011','010'),
    'lbu': ('0000011','100'),
    'lhu': ('0000011','101'),
}
STORE_FUNCTS = {
    'sb': ('0100011','000'),
    'sh': ('0100011','001'),
    'sw': ('0100011','010'),
}
BR_FUNCTS = {
    'beq':  ('1100011','000'),
    'bne':  ('1100011','001'),
    'blt':  ('1100011','100'),
    'bge':  ('1100011','101'),
    'bltu': ('1100011','110'),
    'bgeu': ('1100011','111'),
}
U_FUNCTS = { 'lui': '0110111', 'auipc': '0010111' }
J_FUNCTS = { 'jal': '1101111' }
JALR_FUNCTS = { 'jalr': ('1100111','000') }

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

COMMENT_RE = re.compile(r'(;|#|//).*')
LABEL_RE   = re.compile(r'^\s*([A-Za-z_]\w*):')

def strip_comment(line:str)->str:
    m = COMMENT_RE.search(line)
    return line if not m else line[:m.start()]

def tokenize_operands(op_str:str):
    """
    Tokeniza operandos, preservando %hi(sym)/%lo(sym):
      1) divide por comas/espacios/paréntesis,
      2) recombina ('%hi','simbolo') -> '%hi(simbolo)' (igual para %lo).
    """
    raw = [t for t in re.split(r'[,\s()]+', op_str.strip()) if t]
    toks = []
    i = 0
    while i < len(raw):
        if raw[i] in ('%hi','%lo') and i+1 < len(raw) and re.fullmatch(r'[A-Za-z_]\w*', raw[i+1]):
            toks.append(f"{raw[i]}({raw[i+1]})")
            i += 2
        else:
            toks.append(raw[i]); i += 1
    return toks

# ---------------------------------------------------------------------------
# IR
# ---------------------------------------------------------------------------

class LineIR:
    def __init__(self, seg, addr, kind, content, raw):
        self.seg = seg          # 'text' o 'data'
        self.addr = addr        # dirección base
        self.kind = kind        # 'instr'|'data'
        self.content = content  # (mnem, ops, ln)
        self.raw = raw          # línea original (sin comentario)

# ---------------------------------------------------------------------------
# Ensamblador
# ---------------------------------------------------------------------------

class Assembler:
    def __init__(self, text_base=0x00000000, data_base=0x10000000):
        self.text_base = text_base
        self.data_base = data_base
        self.reset()

    def reset(self):
        self.symbols: Dict[str,int] = {}
        self.lines: List[LineIR] = []
        self.seg = 'text'
        self.pc_text = self.text_base
        self.pc_data = 0
        self.errors: List[str] = []

    def current_addr(self):
        return self.pc_text if self.seg=='text' else (self.data_base + self.pc_data)

    def bump(self, n):
        if self.seg=='text': self.pc_text += n
        else:                self.pc_data += n

    # ---------------------- Pasada 1 ----------------------
    def pass1(self, src_lines):
        self.reset()
        for ln, line in enumerate(src_lines, start=1):
            raw = strip_comment(line).strip()
            if not raw: continue

            # label:
            while True:
                m = LABEL_RE.match(raw)
                if not m: break
                label = m.group(1)
                if label in self.symbols:
                    self.errors.append(f"Línea {ln}: Etiqueta duplicada: {label}")
                else:
                    self.symbols[label] = self.current_addr()
                raw = raw[m.end():].lstrip()
                if not raw: break
            if not raw: continue

            # directivas
            if raw.startswith('.'):
                d = raw.split()[0].lower()
                if d == '.text': self.seg = 'text'
                elif d == '.data': self.seg = 'data'
                elif d == '.word':
                    args = [v for v in raw[len('.word'):].split(',') if v.strip()]
                    for _ in args: self.bump(4)
                elif d == '.byte':
                    args = [v for v in raw[len('.byte'):].split(',') if v.strip()]
                    for _ in args: self.bump(1)
                elif d in ('.asciz','.string'):
                    rest = raw.split(None,1)[1] if len(raw.split())>1 else ''
                    m = re.match(r'"(.*)"', rest)
                    if not m:
                        self.errors.append(f"Línea {ln}: Cadena inválida en {d}")
                    else:
                        s = m.group(1).encode('utf-8') + b'\x00'
                        self.bump(len(s))
                elif d == '.align':
                    rest = raw.split()
                    if len(rest)<2:
                        self.errors.append(f"Línea {ln}: .align requiere argumento")
                    else:
                        n = int(rest[1],10); align = 1<<n
                        addr = self.current_addr()
                        while addr % align != 0:
                            self.bump(1); addr += 1
                continue

            # instrucción / pseudo
            parts = raw.split(None,1)
            mnem = parts[0].lower()
            ops  = tokenize_operands(parts[1]) if len(parts)>1 else []
            nwords = self.expansion_length(mnem, ops)
            self.lines.append(LineIR(self.seg, self.current_addr(), 'instr', (mnem, ops, ln), raw))
            if self.seg == 'text':
                self.bump(4*nwords)

        return len(self.errors)==0

    def expansion_length(self, mnem, ops) -> int:
        if mnem in ('la',): return 2
        if mnem == 'li':
            try:
                v = parse_imm(ops[1]); return 1 if -2048<=v<=2047 else 2
            except Exception: return 2
        if (mnem in ('lb','lh','lw','lbu','lhu') and len(ops)==2) or \
           (mnem in ('sb','sh','sw')  and len(ops)==2):
            return 2
        if mnem in ('nop','mv','not','neg','seqz','snez','sltz','sgtz',
                    'beqz','bnez','blez','bgez','bltz','bgtz',
                    'bgt','ble','bgtu','bleu','j','jal','jr','jalr','ret','call','tail'):
            return 1
        return 1

    # ---------------------- Pasada 2 ----------------------
    def pass2(self):
        machine: List[Tuple[int,int,str,List[str],str]] = []
        for item in self.lines:
            if item.kind!='instr' or item.seg!='text': continue
            mnem, ops, ln = item.content
            try:
                for base_mnem, base_ops in self.expand_recursive(mnem, ops, item.addr):
                    word = self.encode_base(base_mnem, base_ops, item.addr)
                    # guardamos también la fuente original para usarla en la tabla
                    machine.append((item.addr, word, base_mnem, base_ops, item.raw.strip()))
                    item.addr += 4
            except Exception as e:
                self.errors.append(f"Línea {ln}: {e}")
        return machine

    def expand_recursive(self, mnem, ops, pc):
        work = [(mnem,ops)]; out=[]
        while work:
            m,o = work.pop(0)
            if self.is_base(m): out.append((m,o))
            else: work = self.expand_pseudo(m,o,pc) + work
        return out

    def is_base(self, m):
        return (m in R_FUNCTS or m in I_ALU_FUNCTS or m in I_SHIFT_FUNCTS or
                m in LOAD_FUNCTS or m in STORE_FUNCTS or m in BR_FUNCTS or
                m in U_FUNCTS or m in J_FUNCTS or m in JALR_FUNCTS or
                m in ('ecall','ebreak','la.hi','la.lo'))

    def expand_pseudo(self, m, o, pc):
        # simples
        if m=='nop': return [('addi',['x0','x0','0'])]
        if m=='mv':
            if len(o)!=2: raise ValueError("mv espera: rd, rs")
            return [('addi',[o[0],o[1],'0'])]
        if m=='not':
            if len(o)!=2: raise ValueError("not espera: rd, rs")
            return [('xori',[o[0],o[1],'-1'])]
        if m=='neg':
            if len(o)!=2: raise ValueError("neg espera: rd, rs")
            return [('sub',[o[0],'x0',o[1]])]
        if m=='seqz':
            if len(o)!=2: raise ValueError("seqz espera: rd, rs")
            return [('sltiu',[o[0],o[1],'1'])]
        if m=='snez':
            if len(o)!=2: raise ValueError("snez espera: rd, rs")
            return [('sltu',[o[0],'x0',o[1]])]
        if m=='sltz':
            if len(o)!=2: raise ValueError("sltz espera: rd, rs")
            return [('slt',[o[0],o[1],'x0'])]
        if m=='sgtz':
            if len(o)!=2: raise ValueError("sgtz espera: rd, rs")
            return [('slt',[o[0],'x0',o[1]])]

        # ramas abreviadas
        if m=='beqz':
            if len(o)!=2: raise ValueError("beqz espera: rs, etiq/disp")
            return [('beq',[o[0],'x0',o[1]])]
        if m=='bnez':
            if len(o)!=2: raise ValueError("bnez espera: rs, etiq/disp")
            return [('bne',[o[0],'x0',o[1]])]
        if m=='blez':
            if len(o)!=2: raise ValueError("blez espera: rs, etiq/disp")
            return [('bge',['x0',o[0],o[1]])]
        if m=='bgez':
            if len(o)!=2: raise ValueError("bgez espera: rs, etiq/disp")
            return [('bge',[o[0],'x0',o[1]])]
        if m=='bltz':
            if len(o)!=2: raise ValueError("bltz espera: rs, etiq/disp")
            return [('blt',[o[0],'x0',o[1]])]
        if m=='bgtz':
            if len(o)!=2: raise ValueError("bgtz espera: rs, etiq/disp")
            return [('blt',['x0',o[0],o[1]])]
        if m=='bgt':
            if len(o)!=3: raise ValueError("bgt espera: rs, rt, etiq/disp")
            return [('blt',[o[1],o[0],o[2]])]
        if m=='ble':
            if len(o)!=3: raise ValueError("ble espera: rs, rt, etiq/disp")
            return [('bge',[o[1],o[0],o[2]])]
        if m=='bgtu':
            if len(o)!=3: raise ValueError("bgtu espera: rs, rt, etiq/disp")
            return [('bltu',[o[1],o[0],o[2]])]
        if m=='bleu':
            if len(o)!=3: raise ValueError("bleu espera: rs, rt, etiq/disp")
            return [('bgeu',[o[1],o[0],o[2]])]

        # saltos abreviados
        if m=='j':
            if len(o)!=1: raise ValueError("j espera: etiq/disp")
            return [('jal',['x0',o[0]])]
        if m=='jal' and len(o)==1:
            return [('jal',['x1',o[0]])]
        if m=='jr':
            if len(o)!=1: raise ValueError("jr espera: rs")
            return [('jalr',['x0',o[0],'0'])]
        if m=='jalr' and len(o)==1:
            return [('jalr',['x1',o[0],'0'])]
        if m=='ret': return [('jalr',['x0','x1','0'])]
        if m=='call':
            if len(o)!=1: raise ValueError("call espera: offset12")
            return [('jalr',['x1','x1',o[0]])]
        if m=='tail':
            if len(o)!=1: raise ValueError("tail espera: offset12")
            return [('jalr',['x0','x6',o[0]])]

        # li / la
        if m=='li':
            if len(o)!=2: raise ValueError("li espera: rd, imm32")
            rd=o[0]; imm=parse_imm(o[1])
            if -2048<=imm<=2047: return [('addi',[rd,'x0',str(imm)])]
            low=imm & 0xFFF
            if low & 0x800:
                high=(imm+0x1000)>>12; low=low-0x1000
            else:
                high=imm>>12; low=sign_extend(low,12)
            return [('lui',[rd,str(high)]),('addi',[rd,rd,str(low)])]
        if m=='la':
            if len(o)!=2: raise ValueError("la espera: rd, simbolo")
            return [('la.hi',[o[0],o[1]]),('la.lo',[o[0],o[1]])]

        # globals
        if m in ('lb','lh','lw','lbu','lhu') and len(o)==2:
            rd,sym=o; return [('la',[rd,sym]),(m,[rd,f'0({rd})'])]
        if m in ('sb','sh','sw') and len(o)==2:
            rs,sym=o; return [('la',['t3',sym]),(m,[rs,f'0(t3)'])]

        return [(m,o)]

    # ---------------------- Codificadores base ----------------------
    def encode_base(self, m, o, pc):
        if m in ('la.hi','la.lo'):
            rd = reg_number(o[0]); sym=o[1]
            if sym not in self.symbols: raise ValueError(f"Etiqueta indefinida: {sym}")
            addr=self.symbols[sym]; low=addr & 0xFFF
            if low & 0x800: high=(addr+0x1000)>>12; low=low-0x1000
            else:           high=addr>>12;        low=sign_extend(low,12)
            if m=='la.hi': return self.pack_utype(high & 0xFFFFF, rd, U_FUNCTS['lui'])
            else:          return self.pack_itype(low, rd, '000', rd, I_ALU_FUNCTS['addi'][0])

        if m in R_FUNCTS:
            if len(o)!=3: raise ValueError(f"{m} espera: rd, rs1, rs2")
            rd,rs1,rs2 = map(reg_number,o)
            opcode,funct3,funct7 = R_FUNCTS[m]
            return self.pack_rtype(funct7,rs2,rs1,funct3,rd,opcode)

        if m in I_ALU_FUNCTS:
            if len(o)!=3: raise ValueError(f"{m} espera: rd, rs1, imm12")
            rd,rs1 = reg_number(o[0]), reg_number(o[1])
            immv = self.resolve_symbol_or_imm(o[2])
            check_range_signed(immv,12,m)
            opcode,funct3 = I_ALU_FUNCTS[m]
            return self.pack_itype(immv,rs1,funct3,rd,opcode)

        if m in I_SHIFT_FUNCTS:
            if len(o)!=3: raise ValueError(f"{m} espera: rd, rs1, shamt")
            rd,rs1 = reg_number(o[0]), reg_number(o[1])
            shamt = parse_imm(o[2]); check_range_unsigned(shamt,5,m+' shamt')
            opcode,funct3,funct7 = I_SHIFT_FUNCTS[m]
            immv = (int(funct7,2)<<5) | shamt
            return self.pack_itype(immv,rs1,funct3,rd,opcode)

        if m in LOAD_FUNCTS:
            if len(o) not in (2,3): raise ValueError(f"{m} espera: rd, offset(rs1)")
            rd = reg_number(o[0])
            if len(o)==2:
                maddr = re.match(r'(.+)\((.+)\)', o[1])
                if not maddr: raise ValueError(f"{m}: operando debe ser offset(rs1)")
                immv = self.resolve_symbol_or_imm(maddr.group(1)); rs1 = reg_number(maddr.group(2))
            else:
                immv = self.resolve_symbol_or_imm(o[1]); rs1 = reg_number(o[2])
            check_range_signed(immv,12,m)
            opcode,funct3 = LOAD_FUNCTS[m]
            return self.pack_itype(immv,rs1,funct3,rd,opcode)

        if m in STORE_FUNCTS:
            if len(o) not in (2,3): raise ValueError(f"{m} espera: rs2, offset(rs1)")
            rs2 = reg_number(o[0])
            if len(o)==2:
                maddr = re.match(r'(.+)\((.+)\)', o[1])
                if not maddr: raise ValueError(f"{m}: operando debe ser offset(rs1)")
                immv = self.resolve_symbol_or_imm(maddr.group(1)); rs1 = reg_number(maddr.group(2))
            else:
                immv = self.resolve_symbol_or_imm(o[1]); rs1 = reg_number(o[2])
            check_range_signed(immv,12,m)
            opcode,funct3 = STORE_FUNCTS[m]
            return self.pack_stype(immv,rs2,rs1,funct3,opcode)

        if m in BR_FUNCTS:
            if len(o)!=3: raise ValueError(f"{m} espera: rs1, rs2, etiq/disp")
            rs1,rs2 = reg_number(o[0]), reg_number(o[1])
            target = self.resolve_symbol_or_imm(o[2]); offset = target - pc
            check_range_signed(offset,13,m+' (B-type offset)')
            opcode,funct3 = BR_FUNCTS[m]
            return self.pack_btype(offset,rs2,rs1,funct3,opcode)

        if m in U_FUNCTS:
            if len(o)!=2: raise ValueError(f"{m} espera: rd, imm20")
            rd = reg_number(o[0]); immv = self.resolve_symbol_or_imm(o[1]) & 0xFFFFF
            opcode = U_FUNCTS[m]
            return self.pack_utype(immv,rd,opcode)

        if m in J_FUNCTS:
            if len(o)!=2: raise ValueError("jal espera: rd, etiq/disp")
            rd = reg_number(o[0]); target = self.resolve_symbol_or_imm(o[1]); offset = target - pc
            check_range_signed(offset,21,'jal (J-type offset)')
            opcode = J_FUNCTS[m]
            return self.pack_jtype(offset,rd,opcode)

        if m in JALR_FUNCTS:
            if len(o)!=3: raise ValueError("jalr espera: rd, rs1, imm12")
            rd,rs1 = reg_number(o[0]), reg_number(o[1])
            immv = self.resolve_symbol_or_imm(o[2]); check_range_signed(immv,12,'jalr imm')
            opcode,funct3 = JALR_FUNCTS[m]
            return self.pack_itype(immv,rs1,funct3,rd,opcode)

        if m=='ecall':  return 0x00000073
        if m=='ebreak': return 0x00100073
        raise ValueError(f"Instrucción no válida/soportada: {m}")

    def resolve_symbol_or_imm(self, tok: str) -> int:
        try: return eval_hilo(tok, self.symbols)   # %hi/%lo
        except Exception: pass
        if tok in self.symbols: return self.symbols[tok]
        try: return parse_imm(tok)
        except: raise ValueError(f"Etiqueta indefinida: {tok}")

    # empaquetadores
    def pack_rtype(self, funct7, rs2, rs1, funct3, rd, opcode):
        word = (int(funct7,2)<<25)|((rs2&31)<<20)|((rs1&31)<<15)|(int(funct3,2)<<12)|((rd&31)<<7)|int(opcode,2)
        return u32(word)
    def pack_itype(self, imm, rs1, funct3, rd, opcode):
        imm = sign_extend(imm,12) & 0xFFF
        word = (imm<<20)|((rs1&31)<<15)|(int(funct3,2)<<12)|((rd&31)<<7)|int(opcode,2)
        return u32(word)
    def pack_stype(self, imm, rs2, rs1, funct3, opcode):
        imm = sign_extend(imm,12) & 0xFFF
        i11_5=(imm>>5)&0x7F; i4_0=imm&0x1F
        word=(i11_5<<25)|((rs2&31)<<20)|((rs1&31)<<15)|(int(funct3,2)<<12)|(i4_0<<7)|int(opcode,2)
        return u32(word)
    def pack_btype(self, offset, rs2, rs1, funct3, opcode):
        off = sign_extend(offset,13)
        i12=(off>>12)&1; i10_5=(off>>5)&0x3F; i4_1=(off>>1)&0xF; i11=(off>>11)&1
        word=(i12<<31)|(i10_5<<25)|((rs2&31)<<20)|((rs1&31)<<15)|(int(funct3,2)<<12)|(i4_1<<8)|(i11<<7)|int(opcode,2)
        return u32(word)
    def pack_utype(self, imm20, rd, opcode):
        word=((imm20&0xFFFFF)<<12)|((rd&31)<<7)|int(opcode,2)
        return u32(word)
    def pack_jtype(self, offset, rd, opcode):
        off=sign_extend(offset,21); i20=(off>>20)&1; i10_1=(off>>1)&0x3FF; i11=(off>>11)&1; i19_12=(off>>12)&0xFF
        word=(i20<<31)|(i19_12<<12)|(i11<<20)|(i10_1<<21)|((rd&31)<<7)|int(opcode,2)
        return u32(word)

# ---------------------------------------------------------------------------
# Listado (.lst) y tabla “bonita”
# ---------------------------------------------------------------------------

def bits32(word:int)->str: return f"{word:032b}"

def field_groups(mnem: str, word: int):
    b = bits32(word)
    if (mnem in R_FUNCTS):
        return [("funct7",b[0:7]),("rs2",b[7:12]),("rs1",b[12:17]),("funct3",b[17:20]),("rd",b[20:25]),("opcode",b[25:32])]
    if (mnem in I_ALU_FUNCTS) or (mnem in I_SHIFT_FUNCTS) or (mnem in LOAD_FUNCTS) or (mnem in JALR_FUNCTS) or (mnem in ('ecall','ebreak')) or (mnem=='la.lo'):
        return [("imm11_0",b[0:12]),("rs1",b[12:17]),("funct3",b[17:20]),("rd",b[20:25]),("opcode",b[25:32])]
    if (mnem in STORE_FUNCTS):
        return [("imm11_5",b[0:7]),("rs2",b[7:12]),("rs1",b[12:17]),("funct3",b[17:20]),("imm4_0",b[20:25]),("opcode",b[25:32])]
    if (mnem in BR_FUNCTS):
        return [("imm12",b[0:1]),("imm10_5",b[1:7]),("rs2",b[7:12]),("rs1",b[12:17]),("funct3",b[17:20]),("imm4_1",b[20:24]),("imm11",b[24:25]),("opcode",b[25:32])]
    if (mnem in U_FUNCTS) or (mnem=='la.hi'):
        return [("imm31_12",b[0:20]),("rd",b[20:25]),("opcode",b[25:32])]
    if (mnem in J_FUNCTS):
        return [("imm20",b[0:1]),("imm19_12",b[1:9]),("imm11",b[9:10]),("imm10_1",b[10:20]),("rd",b[20:25]),("opcode",b[25:32])]
    return [("bits",b)]

def make_listing(machine):
    rows=[]
    for rec in machine:
        addr, word, mnem, ops = rec[:4]
        asm = f"{mnem} " + ", ".join(ops)
        hexw = f"{word:08x}"
        groups = field_groups(mnem,word)
        group_bits = " ".join([f"{name}={bits}" for name,bits in groups])
        rows.append(f"pc:{addr:08x}  {asm:<24}  {hexw}  {group_bits}")
    return rows

def bin_group32(word:int)->str:
    bits=f"{word:032b}"
    return " ".join(bits[i:i+4] for i in range(0,32,4))

def write_table(machine, out_path:str):
    """Escribe la tabla en out_path (PC | BIN 32b en nibbles | HEX | ; Fuente)."""
    with open(out_path,'w',encoding='utf-8') as f:
        f.write("PC        :  BINARIO (32b, grupos de 4)                HEX        ; Fuente\n")
        f.write("-"*86 + "\n")
        for addr, word, mnem, ops, src in machine:
            # si no hay fuente, mostramos mnem/ops
            src = src or (f"{mnem} " + ", ".join(ops))
            f.write(f"{addr:08x}:  {bin_group32(word):<47}  0x{word:08x}  ; {src}\n")

def write_raw_hex(machine, out_path:str):
    with open(out_path,'w',encoding='utf-8') as fh:
        for _, word, *_ in machine:
            fh.write(f"{word:08x}\n")

def write_raw_bin(machine, out_path:str):
    with open(out_path,'w',encoding='utf-8') as fb:
        for _, word, *_ in machine:
            fb.write(f"{word:032b}\n")

# ---------------------------------------------------------------------------
# Front-end
# ---------------------------------------------------------------------------

def assemble_only(in_path, text_base, data_base):
    asm = Assembler(text_base=text_base, data_base=data_base)
    with open(in_path,'r',encoding='utf-8') as f:
        src_lines = f.readlines()
    ok = asm.pass1(src_lines)
    machine = asm.pass2()
    if not ok or asm.errors:
        raise SystemExit("Errores:\n" + "\n".join(asm.errors))
    return asm, machine

def fmt_bytes(n: int) -> str:
    for unit in ['B','KB','MB','GB']:
        if n < 1024: return f"{n:.0f} {unit}"
        n/=1024
    return f"{n:.0f} TB"

def main():
    p = argparse.ArgumentParser(description="N1.py — Ensamblador RV32I (dos pasadas)")
    p.add_argument('asm'); p.add_argument('hex'); p.add_argument('bin')
    p.add_argument('--text-base', default='0x0')
    p.add_argument('--data-base', default='0x10000000')
    p.add_argument('--list', dest='list_path', default='')
    p.add_argument('--show-list', type=int, default=0)
    p.add_argument('--dump-symbols', action='store_true')
    # salidas planas (opcional)
    p.add_argument('--raw-hex', action='store_true', help='Genera .hex plano (1 palabra/linea)')
    p.add_argument('--raw-bin', action='store_true', help='Genera .bin plano (1 palabra/linea)')
    args = p.parse_args()

    try:
        text_base=int(args.text_base,0); data_base=int(args.data_base,0)
    except:
        raise SystemExit("--text-base/--data-base deben ser enteros (0x...)")

    asm, machine = assemble_only(args.asm, text_base, data_base)

    # Generar .hex y .bin (por defecto en tabla “bonita”)
    if args.raw_hex: write_raw_hex(machine, args.hex)
    else:            write_table(machine, args.hex)

    if args.raw_bin: write_raw_bin(machine, args.bin)
    else:            write_table(machine, args.bin)

    # Resumen
    hex_size=os.path.getsize(args.hex) if os.path.exists(args.hex) else 0
    bin_size=os.path.getsize(args.bin) if os.path.exists(args.bin) else 0
    print(f"✅ Ensamblado OK: {len(machine)} instrucción(es).")
    print(f"HEX → {os.path.abspath(args.hex)}  ({fmt_bytes(hex_size)})")
    print(f"BIN → {os.path.abspath(args.bin)}  ({fmt_bytes(bin_size)})")

    # Listado .lst
    if args.list_path or args.show_list>0:
        lst_lines = make_listing(machine)
        if args.list_path:
            with open(args.list_path,'w',encoding='utf-8') as f:
                f.write("\n".join(lst_lines)+"\n")
        if args.show_list>0:
            print("\nLISTADO:")
            for i,line in enumerate(lst_lines[:args.show_list],start=1):
                print(f"{i:04} {line}")

    if args.dump_symbols:
        print("\nSímbolos (ordenados por dirección):")
        for name,addr in sorted(asm.symbols.items(), key=lambda kv: kv[1]):
            print(f"  {addr:08x}  {name}")

if __name__ == '__main__':
    main()
