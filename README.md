
# Simulación Cuántica de Moléculas Usando PennyLane

Este proyecto implementa un flujo de trabajo para simular y optimizar moléculas mediante métodos de computación cuántica, utilizando técnicas como la Variational Quantum Eigensolver (VQE) y otras herramientas de la biblioteca PennyLane. Es ideal para experimentar con optimización molecular y simulación de reacciones químicas.

---

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Dependencias](#dependencias)
4. [Uso](#uso)
5. [Ejemplo de Flujo de Trabajo](#ejemplo-de-flujo-de-trabajo)
6. [Archivos Relevantes](#archivos-relevantes)
7. [Visualización de Resultados](#visualización-de-resultados)
8. [Referencias](#referencias)

---

## Descripción General

El propósito principal de este proyecto es simular y optimizar geometrías moleculares mediante técnicas cuánticas, integrando herramientas clásicas y cuánticas para resolver problemas en química computacional.

El proyecto utiliza el Hamiltoniano molecular generado mediante **PennyLane Quantum Chemistry Module**, junto con optimizadores cuánticos y algoritmos adaptativos para explorar geometrías y calcular energías moleculares exactas.

---

## Estructura del Proyecto

La estructura del proyecto está dividida en directorios y módulos, cada uno con una función específica:

### Directorios

- **`modules/`**: Contiene los módulos principales para simulación y optimización.
  - `hamiltonian_builder.py`: Construcción del Hamiltoniano molecular y estado de referencia Hartree-Fock.
  - `ansatz_preparer.py`: Preparación del circuito cuántico (ansatz) y cálculo de gradientes.
  - `optimizer.py`: Lógica de optimización para circuitos cuánticos y geometrías moleculares.
  - `visualizer.py`: Herramientas de visualización para resultados y evolución de geometrías.
  - `molecule_manager.py`: Gestión e inicialización de moléculas.
- **`config/`**: Configuración y utilidades.
  - `config_functions.py`: Configuración de parámetros, carga de moléculas y opciones del usuario.
- **`results/`**: Carpeta donde se almacenan resultados y visualizaciones generadas.
- **`temp_results_autograd/`**: Carpeta temporal para resultados intermedios.

### Archivos Clave

1. **`main.py`**:
   - Punto de entrada principal para ejecutar simulaciones o optimizaciones moleculares.
   - Invoca funciones de módulos específicos según la configuración.

2. **`molecules.json`**:
   - Contiene moléculas predefinidas con propiedades como símbolos atómicos, coordenadas iniciales, carga, y multiplicidad.

---

## Dependencias

Este proyecto requiere las siguientes bibliotecas:

- **Python 3.8 o superior**
- PennyLane (para simulación cuántica)
- NumPy
- Matplotlib (para visualización)
- Tabulate (para representación de tablas)

Para instalar las dependencias, ejecuta:
```bash
pip install pennylane numpy matplotlib tabulate
```

---

## Uso

### 1. Ejecución Básica

Para optimizar una molécula, utiliza:
```bash
python main.py --molecule H2O --opt
```

Esto optimiza la geometría de la molécula `H2O` según los parámetros y configuraciones definidos.

### 2. Argumentos Principales

| Argumento          | Descripción                                           | Ejemplo                          |
|---------------------|-------------------------------------------------------|----------------------------------|
| `--molecule`        | Nombre de la molécula a simular.                      | `--molecule H2O`                |
| `--opt`             | Ejecuta optimización de geometría molecular.          | `--opt`                         |
| `--basis_set`       | Conjunto de bases para la simulación.                 | `--basis_set sto-3g`            |
| `--stepsize`        | Tamaño de paso para el optimizador.                   | `--stepsize 0.01`               |
| `--add_molecule`    | Permite añadir una nueva molécula a `molecules.json`. | `--add_molecule`                |
| `--plot`            | Genera gráficos de energía y coordenadas.            | `--plot`                        |

---

## Ejemplo de Flujo de Trabajo

1. **Simulación de una molécula existente**:
   ```bash
   python main.py --molecule H2 --opt
   ```

2. **Añadir una nueva molécula**:
   ```bash
   python main.py --add_molecule
   ```
   Sigue las instrucciones interactivas para definir símbolos, coordenadas, carga y multiplicidad.

3. **Optimización con diferentes configuraciones**:
   ```bash
   python main.py --molecule H2O --basis_set cc-pVDZ --stepsize 0.02
   ```

---

## Archivos Relevantes

### `hamiltonian_builder.py`

- Construcción del Hamiltoniano molecular usando:
  - Coordenadas atómicas.
  - Conjuntos de bases definidos (`sto-3g`, `cc-pVDZ`, etc.).
- Generación de estados de referencia Hartree-Fock.

### `ansatz_preparer.py`

- Configuración del circuito cuántico (ansatz) adaptativo.
- Cálculo de gradientes para seleccionar operadores relevantes.

### `optimizer.py`

- Implementación de optimizadores como:
  - **Gradient Descent**
  - **Quantum Natural Gradient**
- Optimización conjunta de parámetros de circuitos y coordenadas moleculares.

---

## Visualización de Resultados

### 1. Energía Durante la Optimización

Se genera un gráfico que muestra cómo evoluciona la energía a lo largo de las iteraciones.

### 2. Geometrías Finales

Las geometrías finales se visualizan en 3D, mostrando la posición de los átomos en el espacio.

Los resultados se guardan automáticamente en `temp_results_autograd/`.

---

## Referencias

- **PennyLane Documentation**: [https://pennylane.ai](https://pennylane.ai)
- **Quantum Chemistry Tutorials**: [https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html](https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html)

## Contacto

Para preguntas o mejoras, contactar a **Albert López Escudero**.
