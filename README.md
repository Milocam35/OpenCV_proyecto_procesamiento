# Proyecto OpenCV con C++

Este es un proyecto base de OpenCV utilizando C++ y CMake.

## Estructura del Proyecto

```
opencv_project/
├── build/          # Archivos de compilación
├── include/        # Archivos de cabecera (.h, .hpp)
├── src/            # Archivos fuente (.cpp)
├── CMakeLists.txt  # Configuración de CMake
└── README.md       # Este archivo
```

## Requisitos

- CMake (versión 3.10 o superior)
- OpenCV (instalado en el sistema)
- Compilador C++ compatible con C++17

## Compilación

1. Navega a la carpeta build:
```bash
cd build
```

2. Genera los archivos de compilación:
```bash
cmake ..
```

3. Compila el proyecto:
```bash
cmake --build .
```

## Ejecución

Después de compilar, ejecuta el programa desde la carpeta build:
```bash
./OpenCV_Project      # En Linux/Mac
OpenCV_Project.exe    # En Windows
```

## Notas

- Asegúrate de tener OpenCV correctamente instalado en tu sistema
- Los archivos de cabecera personalizados van en la carpeta `include/`
- Los archivos fuente van en la carpeta `src/`
- La carpeta `build/` no debe incluirse en el control de versiones