#include <opencv2/opencv.hpp>
#include <iostream>

// Declaraciones de funciones
cv::Mat conversion_gray(cv::Mat frame);
cv::Mat conversion_yuv(cv::Mat frame);
cv::Mat conversion_hsv(cv::Mat frame);

// Implementacion de funciones
cv::Mat conversion_gray(cv::Mat frame) {
    cv::Mat resultado = frame.clone();

    for(int i = 0; i < frame.rows; i++) {
        for(int j = 0; j < frame.cols; j++) {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);
            uchar gray = (pixel[0] + pixel[1] + pixel[2]) / 3;
            resultado.at<cv::Vec3b>(i, j) = cv::Vec3b(gray, gray, gray);
        }
    }
    return resultado;
}

cv::Mat conversion_yuv(cv::Mat frame) {
    cv::Mat resultado = frame.clone();

    for(int i = 0; i < frame.rows; i++) {
        for(int j = 0; j < frame.cols; j++) {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);

            // OpenCV usa BGR, no RGB
            float B = pixel[0];
            float G = pixel[1];
            float R = pixel[2];

            float Y = R * 0.299 + G * 0.587 + B * 0.114;
            float U = (B - Y) * 0.492;
            float V = (R - Y) * 0.877;

            // Normalizar para visualizacion
            resultado.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(Y);
            resultado.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(U + 128);
            resultado.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(V + 128);
        }
    }
    return resultado;
}

cv::Mat conversion_hsv(cv::Mat frame) {
    cv::Mat resultado = frame.clone();

    for(int i = 0; i < frame.rows; i++) {
        for(int j = 0; j < frame.cols; j++) {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);

            // OpenCV usa BGR
            float B = pixel[0] / 255.0;
            float G = pixel[1] / 255.0;
            float R = pixel[2] / 255.0;

            float max = std::max(R, std::max(G, B));
            float min = std::min(R, std::min(G, B));
            float delta = max - min;

            float H = 0, S = 0, V = max;

            if (delta != 0) {
                S = delta / max;

                if (max == R) {
                    H = 60 * fmod((G - B) / delta, 6);
                } else if (max == G) {
                    H = 60 * ((B - R) / delta + 2);
                } else {
                    H = 60 * ((R - G) / delta + 4);
                }

                if (H < 0) H += 360;
            }

            // Normalizar para OpenCV (H: 0-180, S: 0-255, V: 0-255)
            resultado.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(H / 2);
            resultado.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(S * 255);
            resultado.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(V * 255);
        }
    }
    return resultado;
}

int main() {
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Error: no se pudo abrir la camara" << std::endl;
        return -1;
    }

    std::cout << "Presiona una tecla para capturar la imagen..." << std::endl;
    std::cout << "('q' para salir)" << std::endl;

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: frame vacio" << std::endl;
            break;
        }

        cv::imshow("Camara", frame);

        int key = cv::waitKey(1);
        if (key == 'q') {
            cap.release();
            cv::destroyAllWindows();
            return 0;
        }
        if (key == 'c') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    // Menu de conversion
    std::cout << "\nImagen capturada. Seleccione la conversion:\n";
    std::cout << "1) YUV\n";
    std::cout << "2) HSV\n";
    std::cout << "3) Escala de grises\n";
    std::cout << "Opcion: ";

    int opcion;
    std::cin >> opcion;

    cv::Mat resultado;
    std::string nombre;

    switch (opcion) {
        case 1:
            resultado = conversion_yuv(frame);
            nombre = "YUV";
            break;
        case 2:
            resultado = conversion_hsv(frame);
            nombre = "HSV";
            break;
        case 3:
            resultado = conversion_gray(frame);
            nombre = "Escala de grises";
            break;
        default:
            std::cerr << "Opcion no valida" << std::endl;
            return -1;
    }

    cv::imshow("Original", frame);
    cv::imshow(nombre, resultado);

    std::cout << "Mostrando resultado. Presiona cualquier tecla para salir." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
