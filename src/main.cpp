#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "OpenCV Project - Version: " << CV_VERSION << std::endl;

    // Ejemplo básico: crear una imagen en blanco
    cv::Mat imagen(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));

    // Agregar texto a la imagen
    cv::putText(imagen, "Hola OpenCV!",
                cv::Point(150, 240),
                cv::FONT_HERSHEY_SIMPLEX,
                1.5,
                cv::Scalar(255, 255, 255),
                2);

    // Mostrar la imagen
    cv::imshow("Ventana de OpenCV", imagen);
    std::cout << "Presiona cualquier tecla para cerrar..." << std::endl;
    cv::waitKey(0);

    return 0;
}
