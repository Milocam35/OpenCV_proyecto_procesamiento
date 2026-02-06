cmake_minimum_required(VERSION 3.10)
project(OpenCV_Project)

# Configurar estándar de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Buscar OpenCV
find_package(OpenCV REQUIRED)

# Incluir directorios
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# Archivos fuente
file(GLOB SOURCES "src/*.cpp")

# Crear ejecutable
add_executable(${PROJECT_NAME} ${SOURCES})

# Enlazar librerías de OpenCV
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Mensaje de información
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
