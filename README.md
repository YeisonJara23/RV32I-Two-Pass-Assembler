
# Ensamblador RV32I para RISC-V - Documentación Completa

## Introducción y Propósito

El Ensamblador RV32I es una herramienta educativa desarrollada en Python que permite traducir programas escritos en lenguaje ensamblador RISC-V a código máquina ejecutable. Este proyecto está diseñado específicamente para facilitar el aprendizaje de arquitectura de computadoras y programación a bajo nivel, proporcionando una implementación clara y documentada del proceso de ensamblado.

## Funcionalidades Principales

### Proceso de Dos Pasadas

El ensamblador implementa una arquitectura de dos pasadas que garantiza un análisis completo y robusto del código fuente. En la primera pasada, el sistema recorre el código para identificar todas las etiquetas, construir la tabla de símbolos y calcular las direcciones de memoria. La segunda pasada utiliza esta información para generar el código máquina final, resolviendo todas las referencias a símbolos y expandiendo las pseudoinstrucciones.

### ¿Por qué es Necesario?

-   Los procesadores solo entienden **código binario** (0s y 1s)
    
-   Escribir directamente en binario es **extremadamente difícil** para los humanos
    
-   El lenguaje ensamblador proporciona **mnemónicos** (abreviaturas) para las instrucciones
    
-   El ensamblador **automatiza** la conversión y evita errores

### Soporte para Pseudoinstrucciones

Una de las funcionalidades más valiosas es el soporte integral para pseudoinstrucciones. Estas son abreviaciones que simplifican la escritura de código ensamblador, haciendo que sea más legible y fácil de escribir. Por ejemplo, en lugar de escribir la secuencia completa para cargar un valor inmediato grande, los usuarios pueden emplear la pseudoinstrucción "li" (load immediate), que el ensamblador expande automáticamente en las instrucciones "lui" y "addi" apropiadas.

### Manejo de Símbolos y Etiquetas

El sistema mantiene una tabla de símbolos completa que permite el uso de etiquetas para representar direcciones de memoria. Esto facilita la escritura de código con saltos y referencias a datos, ya que los programadores pueden usar nombres significativos en lugar de direcciones numéricas. El ensamblador se encarga de calcular las direcciones correctas durante el proceso de ensamblado.

### Múltiples Formatos de Salida

Para adaptarse a diferentes necesidades, el ensamblador genera varios formatos de salida. Los usuarios pueden obtener representaciones detalladas que muestran el código máquina junto con el código fuente original, o formatos más simples que contienen solo el código binario o hexadecimal para su uso en simuladores o hardware real.

## Implementación Técnica

### Arquitectura del Sistema

El núcleo del ensamblador está implementado como una clase principal en Python que coordina todo el proceso. Esta clase gestiona el estado del ensamblado, incluyendo los contadores de programa, la tabla de símbolos y la lista de errores. El diseño modular permite una fácil extensión y mantenimiento del código.

### Procesamiento de Código Fuente

El sistema incluye un tokenizador inteligente que procesa el código fuente, eliminando comentarios y dividiendo las instrucciones en sus componentes. Este tokenizador es capaz de manejar construcciones complejas como las macros %hi() y %lo() para el manejo de direcciones de 32 bits, preservándolas como unidades coherentes durante el análisis.

### Expansión de Pseudoinstrucciones

El mecanismo de expansión de pseudoinstrucciones opera de manera recursiva, permitiendo que las pseudoinstrucciones se expandan en otras pseudoinstrucciones hasta que solo queden instrucciones base del conjunto RV32I. Este enfoque proporciona gran flexibilidad y permite crear pseudoinstrucciones complejas que se construyen sobre otras más simples.

### Codificación de Instrucciones

Para cada tipo de instrucción RV32I (R, I, S, B, U, J), el ensamblador implementa funciones de codificación específicas que empaquetan los campos de la instrucción en palabras de 32 bits según el formato establecido por la especificación RISC-V. Esto incluye el manejo correcto de la extensión de signo y la verificación de rangos para los valores inmediatos.

## Cómo Utilizar el Ensamblador

### Instalación y Configuración

El ensamblador requiere Python 3.6 o superior y no tiene dependencias externas, lo que facilita su instalación. Los usuarios simplemente necesitan descargar los archivos del proyecto y pueden comenzar a usarlo inmediatamente.

### Uso Básico

El uso fundamental del ensamblador implica ejecutar el script de Python desde la línea de comandos, especificando el archivo de código fuente y los archivos de salida deseados. El sistema procesa el código y genera los archivos solicitados, proporcionando retroalimentación sobre el éxito o los errores encontrados.

### Ejemplos de Comandos de Uso

**Comando básico para ensamblar un programa:**

**Comando básico para ensamblar un programa:**

    python N1.py programa.asm programa.hex programa.bin

**Generar salida hexadecimal en formato plano (ideal para simuladores):**

    python N1.py selsort.asm selsort.hex selsort.bin --raw-hex

**Crear listado detallado con visualización de símbolos:**

    python N1.py selsort.asm selsort.hex selsort.bin --list selsort.lst --show-list 20 --dump-symbols
**Ensamblar con bases de memoria personalizadas:**

    python N1.py programa.asm salida.hex salida.bin --text-base 0x1000 --data-base 0x2000
   **Generar solo archivo binario en formato plano:**
   

    python N1.py programa.asm programa.hex programa.bin --raw-bin

   **Comando para depuración con salida completa:**
   

    python N1.py programa.asm programa.hex programa.bin --list debug.lst --show-list 50 --dump-symbols
**Ensamblado rápido sin listado:**

    python N1.py programa.asm programa.hex programa.bin
    
**Comando para ver solo los primeros 10 resultados:**

    python N1.py programa.asm programa.hex programa.bin --show-list 10

### Opciones Avanzadas

Para casos de uso más específicos, el ensamblador ofrece varias opciones de configuración. Los usuarios pueden personalizar las direcciones base de los segmentos de código y datos, generar listados detallados del ensamblado, mostrar tablas de símbolos, y elegir entre diferentes formatos de salida.

    
### Ejemplos de Uso

El proyecto incluye varios programas de ejemplo que demuestran diferentes aspectos del lenguaje ensamblador RISC-V. Estos ejemplos van desde programas simples que realizan operaciones aritméticas básicas hasta algoritmos más complejos como la secuencia de Fibonacci, proporcionando puntos de partida útiles para nuevos usuarios.

## Flujo de Trabajo Recomendado

### Desarrollo de Programas

Para desarrollar programas usando este ensamblador, se recomienda comenzar con los ejemplos proporcionados y modificarlos gradualmente. Los usuarios pueden escribir su código en cualquier editor de texto y guardarlo con extensión .asm.

### Proceso de Ensamblado

El proceso típico implica ejecutar el ensamblador sobre el código fuente para generar los archivos de código máquina. Si se encuentran errores, el sistema proporciona mensajes descriptivos que indican la línea y naturaleza del problema, permitiendo correcciones rápidas.

### Prueba y Verificación

Una vez generado el código máquina, los usuarios pueden cargarlo en un simulador RISC-V o hardware compatible para probar su funcionamiento. Los formatos de salida detallados facilitan la depuración al mostrar la correspondencia entre el código fuente y el código máquina generado.

## Casos de Uso Específicos

### Educación en Arquitectura de Computadoras

Este ensamblador es ideal para cursos de arquitectura de computadoras, permitiendo a los estudiantes experimentar con programación a bajo nivel sin la complejidad de herramientas profesionales. La transparencia del proceso de ensamblado ayuda a entender cómo se traduce el código a instrucciones de máquina.

### Prototipado Rápido

Desarrolladores que trabajan con sistemas RISC-V pueden usar este ensamblador para prototipar algoritmos y verificar conceptos antes de implementarlos en entornos más complejos. La simplicidad del flujo de trabajo acelera el ciclo de desarrollo.

### Referencia para Implementación

El código fuente del ensamblador sirve como referencia valiosa para entender cómo implementar un ensamblador básico. La estructura clara y la documentación extensa lo hacen accesible para quienes desean aprender sobre este tipo de herramientas.

## Escenarios Prácticos de Uso

### Para Desarrollo de Algoritmos

Cuando se desarrollan algoritmos como ordenamiento (ejemplo selsort), el comando con --raw-hex es útil para obtener un formato limpio que puede ser cargado directamente en simuladores. La opción --dump-symbols ayuda a verificar que todas las etiquetas del algoritmo se hayan resuelto correctamente.

### Para Depuración de Código

Al encontrar errores en programas complejos, el comando con --list y --show-list permite examinar las primeras líneas del listado generado para identificar problemas sin generar un archivo completo. Esto acelera el ciclo de depuración.

### Para Integración con Otras Herramientas

El formato --raw-hex y --raw-bin genera salidas limpias que pueden ser procesadas fácilmente por otros programas, scripts de automatización, o cargadas en entornos de desarrollo integrados.

### Para Documentación de Proyectos

Al finalizar un proyecto, el comando con --list genera un archivo de listado completo que sirve como documentación técnica, mostrando cómo cada línea de código fuente se tradujo a instrucciones de máquina.

## Consideraciones de Diseño

### Enfoque Educativo

Cada aspecto del ensamblador está diseñado con la educación en mente. Los mensajes de error son descriptivos, el código es legible, y el proceso es transparente. Esto contrasta con ensambladores profesionales que priorizan el rendimiento sobre la claridad.


### Compatibilidad

El ensamblador se adhiere estrechamente a la especificación RV32I base, asegurando compatibilidad con una amplia gama de simuladores y hardware. Las extensiones futuras podrían añadir soporte para otros conjuntos de instrucciones RISC-V.

## Conclusión

El Ensamblador RV32I representa una herramienta valiosa para cualquiera interesado en la programación de bajo nivel y la arquitectura RISC-V. Su combinación de funcionalidades robustas, diseño educativo y facilidad de uso lo hacen adecuado tanto para principiantes como para usuarios experimentados que buscan una herramienta de prototipado rápida y confiable. Los diversos comandos y opciones disponibles permiten adaptar el proceso de ensamblado a diferentes necesidades, desde desarrollo rápido hasta depuración detallada y documentación técnica.



