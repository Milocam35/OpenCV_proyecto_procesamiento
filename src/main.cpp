#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;

struct Pixel
{
    double r, g, b;                                                 // los 3 canales como double
    Pixel() : r(0), g(0), b(0) {}                                   // constructor por defecto (0,0,0)
    Pixel(double r_, double g_, double b_) : r(r_), g(g_), b(b_) {} // constructor con valores
};

// Declaraciones de funciones
cv::Mat conversion_gray(cv::Mat frame);
cv::Mat conversion_yuv(cv::Mat frame);
cv::Mat conversion_hsv(cv::Mat frame);
cv::Mat conversion_hsv_a_rgb(cv::Mat frame);
cv::Mat modificar_saturacion(cv::Mat frame);
cv::Mat bgr_a_rgb(cv::Mat frame);
double distancia_euclidiana(const Pixel &p1, const Pixel &p2);
cv::Mat kmeans_manual(cv::Mat frame, int K);
cv::Mat gray_world(cv::Mat frame);
cv::Mat correccion_gamma(cv::Mat frame, double gamma);
cv::Mat correccion_vineteo(cv::Mat frame, double k);

cv::Mat bgr_a_rgb(cv::Mat frame)
{
    cv::Mat resultado = frame.clone();

    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            resultado.at<cv::Vec3b>(i, j)[0] = frame.at<cv::Vec3b>(i, j)[2];
            resultado.at<cv::Vec3b>(i, j)[2] = frame.at<cv::Vec3b>(i, j)[0];
        }
    }
    return resultado;
}

double distancia_euclidiana(const Pixel &p1, const Pixel &p2)
{
    double diff_r = p1.r - p2.r;
    double diff_g = p1.g - p2.g;
    double diff_b = p1.b - p2.b;
    return sqrt((diff_r * diff_r) + (diff_g * diff_g) + (diff_b * diff_b));
}

// Implementacion de funciones
cv::Mat conversion_gray(cv::Mat frame)
{
    cv::Mat resultado = frame.clone();

    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);
            uchar gray = (pixel[0] + pixel[1] + pixel[2]) / 3;
            resultado.at<cv::Vec3b>(i, j) = cv::Vec3b(gray, gray, gray);
        }
    }
    return resultado;
}

cv::Mat conversion_yuv(cv::Mat frame)
{
    cv::Mat resultado = frame.clone();

    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
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

cv::Mat conversion_hsv(cv::Mat frame)
{
    cv::Mat resultado = frame.clone();

    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);

            // OpenCV usa BGR
            float B = pixel[0] / 255.0;
            float G = pixel[1] / 255.0;
            float R = pixel[2] / 255.0;

            float max = std::max(R, std::max(G, B));
            float min = std::min(R, std::min(G, B));
            float delta = max - min;

            float H = 0, S = 0, V = max;

            if (delta != 0)
            {
                S = delta / max;

                if (max == R)
                {
                    H = 60 * fmod((G - B) / delta, 6);
                }
                else if (max == G)
                {
                    H = 60 * ((B - R) / delta + 2);
                }
                else
                {
                    H = 60 * ((R - G) / delta + 4);
                }

                if (H < 0)
                    H += 360;
            }

            // Normalizar para OpenCV (H: 0-180, S: 0-255, V: 0-255)
            resultado.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(H / 2);
            resultado.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(S * 255);
            resultado.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(V * 255);
        }
    }
    return resultado;
}

cv::Mat conversion_hsv_a_rgb(cv::Mat frame)
{
    cv::Mat resultado(frame.rows, frame.cols, CV_8UC3);

    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);

            // Desnormalizar de formato OpenCV a valores reales
            float H = pixel[0] * 2.0;
            float S = pixel[1] / 255.0;
            float V = pixel[2] / 255.0;

            float C = V * S;
            float X = C * (1 - fabs(fmod(H / 60.0, 2) - 1));
            float m = V - C;

            float R1, G1, B1;

            if (H < 60)
            {
                R1 = C;
                G1 = X;
                B1 = 0;
            }
            else if (H < 120)
            {
                R1 = X;
                G1 = C;
                B1 = 0;
            }
            else if (H < 180)
            {
                R1 = 0;
                G1 = C;
                B1 = X;
            }
            else if (H < 240)
            {
                R1 = 0;
                G1 = X;
                B1 = C;
            }
            else if (H < 300)
            {
                R1 = X;
                G1 = 0;
                B1 = C;
            }
            else
            {
                R1 = C;
                G1 = 0;
                B1 = X;
            }

            // Convertir a BGR (formato OpenCV) con rango 0-255
            resultado.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>((B1 + m) * 255);
            resultado.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>((G1 + m) * 255);
            resultado.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>((R1 + m) * 255);
        }
    }
    return resultado;
}

cv::Mat modificar_saturacion(cv::Mat frame)
{
    int rows = frame.rows;
    int cols = frame.cols;
    cv::Mat resultado(frame.rows, frame.cols, CV_8UC3);

    cv::Mat frame_hsv = conversion_hsv(frame);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // PASO 1: Obtener valores H, S, V
            cv::Vec3b pixel = frame_hsv.at<cv::Vec3b>(i, j);
            float H = pixel[0];
            float S = pixel[1];
            float V = pixel[2];

            // PASO 2: Multiplicar S por 1.5 (sin exceder 255)
            S = S * 1.5;
            if (S > 255)
            {
                S = 255;
            }

            resultado.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(H);
            resultado.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(S);
            resultado.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(V);
        };
    };

    cv::Mat resultado_rgb = conversion_hsv_a_rgb(resultado);

    return resultado_rgb;
};

cv::Mat gray_world(cv::Mat frame)
{
    int rows = frame.rows;
    int cols = frame.cols;
    int total_pixels = rows * cols;

    // PASO 1 - Calcular suma de cada canal
    double sum_B = 0, sum_G = 0, sum_R = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);
            sum_B += pixel[0];
            sum_G += pixel[1];
            sum_R += pixel[2];
        }
    }

    // PASO 2 - Calcular promedios
    double avg_B = sum_B / total_pixels;
    double avg_G = sum_G / total_pixels;
    double avg_R = sum_R / total_pixels;

    // PASO 3 - Calcular promedio gris
    double gray_avg = (avg_R + avg_G + avg_B) / 3.0;

    // PASO 4 - Calcular factores de escala
    double scale_B = gray_avg / avg_B;
    double scale_G = gray_avg / avg_G;
    double scale_R = gray_avg / avg_R;

    cout << "Promedios BGR: " << avg_B << ", " << avg_G << ", " << avg_R << endl;
    cout << "Promedio gris: " << gray_avg << endl;
    cout << "Factores de escala BGR: " << scale_B << ", " << scale_G << ", " << scale_R << endl;

    // PASO 5 - Crear imagen corregida
    cv::Mat resultado(rows, cols, CV_8UC3);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);
            resultado.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(pixel[0] * scale_B);
            resultado.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(pixel[1] * scale_G);
            resultado.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(pixel[2] * scale_R);
        }
    }

    return resultado;
}

cv::Mat correccion_gamma(cv::Mat frame, double gamma)
{
    int rows = frame.rows;
    int cols = frame.cols;

    // PASO 1 - Crear tabla de lookup
    // Pre-calcular la transformacion para todos los valores 0-255
    uchar tabla[256];
    for (int i = 0; i < 256; i++)
    {
        tabla[i] = cv::saturate_cast<uchar>(255.0 * pow(i / 255.0, gamma));
    }

    // PASO 2 - Aplicar transformacion a cada pixel
    cv::Mat resultado(rows, cols, CV_8UC3);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);
            resultado.at<cv::Vec3b>(i, j)[0] = tabla[pixel[0]];
            resultado.at<cv::Vec3b>(i, j)[1] = tabla[pixel[1]];
            resultado.at<cv::Vec3b>(i, j)[2] = tabla[pixel[2]];
        }
    }

    return resultado;
}

cv::Mat correccion_vineteo(cv::Mat frame, double k)
{
    int rows = frame.rows;
    int cols = frame.cols;

    // PASO 1 - Calcular centro de la imagen
    double cx = cols / 2.0;
    double cy = rows / 2.0;

    // PASO 2 - Calcular distancia maxima (del centro a la esquina)
    double d_max = sqrt(cx * cx + cy * cy);

    // PASO 3 - Aplicar correccion pixel por pixel
    cv::Mat resultado(rows, cols, CV_8UC3);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // Distancia al centro
            double d = sqrt((j - cx) * (j - cx) + (i - cy) * (i - cy));

            // Distancia normalizada
            double d_norm = d / d_max;

            // Factor de correccion
            double f = 1.0 / (1.0 - k * d_norm * d_norm);

            // Aplicar factor a cada canal
            cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);
            resultado.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(pixel[0] * f);
            resultado.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(pixel[1] * f);
            resultado.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(pixel[2] * f);
        }
    }

    return resultado;
}

cv::Mat kmeans_manual(cv::Mat frame, int K)
{
    // Convertir de BRG a RGB para facilitar los calculos
    cv::Mat frame_rgb = bgr_a_rgb(frame);

    // Redimensionar para acelerar
    cv::Mat frame_small;
    cv::resize(frame_rgb, frame_small, cv::Size(160, 120));

    int rows = frame_small.rows;
    int cols = frame_small.cols;
    int total_pixels = rows * cols;
    cout << "Procesando " << total_pixels << " píxeles con K=" << K << endl;
    // TODO: PASO 1 - Crear array de píxeles
    vector<Pixel> pixels;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            Pixel pixel_frame = Pixel(frame_small.at<cv::Vec3b>(i, j)[0], frame_small.at<cv::Vec3b>(i, j)[1], frame_small.at<cv::Vec3b>(i, j)[2]);
            pixels.push_back(pixel_frame);
        }
    }
    // TODO: PASO 2 - Inicializar K centroides aleatorios
    vector<Pixel> centroides(K);
    for (int k = 0; k < K; k++)
    {
        int idx_random = rand() % total_pixels;
        centroides[k] = pixels[idx_random];
    }
    // TODO: PASO 3 - Array para almacenar asignaciones
    // Cada píxel se asigna a un cluster [0, K-1]
    vector<int> asignaciones(total_pixels, -1);

    // TODO: PASO 4 - Iterar K-Means
    int max_iteraciones = 20;
    for (int iter = 0; iter < max_iteraciones; iter++)
    {
        cout << "Iteración " << (iter + 1) << "/" << max_iteraciones << endl;
        // PASO 4a: Asignar cada píxel al centroide más cercano
        for (int i = 0; i < total_pixels; i++)
        {
            // Aquí se asignaría el píxel al cluster más cercano
            double min_dist = 0;
            for (int j = 0; j < K; j++)
            {
                double dist = distancia_euclidiana(pixels[i], centroides[j]);
                if (min_dist > dist || min_dist == 0)
                {
                    min_dist = dist;
                    asignaciones[i] = j;
                }
            }
        }
        // PASO 4b: Recalcular centroides
        for (int i = 0; i < K; i++)
        {
            double r = 0, g = 0, b = 0;
            int total_group = 0;
            for (int j = 0; j < total_pixels; j++)
            {
                if(asignaciones[j] == i)
                {
                    r += pixels[j].r;
                    g += pixels[j].g;
                    b += pixels[j].b;
                    total_group++;
                }
            }
            centroides[i].r = r / total_group;
            centroides[i].g = g / total_group;
            centroides[i].b = b / total_group;
        }
        
    }
    // TODO: PASO 5 - Crear imagen cuantizada
    for(int i = 0; i < total_pixels; i++)
    {
        int cluster_id = asignaciones[i];
        frame_small.at<cv::Vec3b>(i / cols, i % cols)[0] = cv::saturate_cast<uchar>(centroides[cluster_id].r);
        frame_small.at<cv::Vec3b>(i / cols, i % cols)[1] = cv::saturate_cast<uchar>(centroides[cluster_id].g);
        frame_small.at<cv::Vec3b>(i / cols, i % cols)[2] = cv::saturate_cast<uchar>(centroides[cluster_id].b);
    }

    return frame_small;
}

int main()
{
    cv::Mat frame;

    // Seleccionar fuente de imagen
    cout << "Seleccione fuente de imagen:\n";
    cout << "1) Camara\n";
    cout << "2) Archivo\n";
    cout << "Opcion: ";

    int fuente;
    cin >> fuente;

    if (fuente == 2)
    {
        cout << "Ingrese la ruta de la imagen: ";
        string ruta;
        cin >> ruta;
        frame = cv::imread(ruta);
        if (frame.empty())
        {
            cerr << "Error: no se pudo abrir la imagen: " << ruta << endl;
            return -1;
        }
    }
    else
    {
        cv::VideoCapture cap(0);

        if (!cap.isOpened())
        {
            cerr << "Error: no se pudo abrir la camara" << endl;
            return -1;
        }

        cout << "Presiona una tecla para capturar la imagen..." << endl;
        cout << "('q' para salir)" << endl;

        while (true)
        {
            cap >> frame;
            if (frame.empty())
            {
                cerr << "Error: frame vacio" << endl;
                break;
            }

            cv::imshow("Camara", frame);

            int key = cv::waitKey(1);
            if (key == 'q')
            {
                cap.release();
                cv::destroyAllWindows();
                return 0;
            }
            if (key == 'c')
            {
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows();
    }

    // Menu de conversion
    std::cout << "\nImagen capturada. Seleccione la conversion:\n";
    std::cout << "1) YUV\n";
    std::cout << "2) HSV\n";
    std::cout << "3) Escala de grises\n";
    std::cout << "4) Modificar Saturación\n";
    std::cout << "5) K-Means Manual\n";
    std::cout << "6) Gray World\n";
    std::cout << "7) Correccion Gamma\n";
    std::cout << "8) Correccion Vineteo\n";
    std::cout << "Opcion: ";

    int opcion;
    std::cin >> opcion;

    cv::Mat resultado;
    std::string nombre;

    switch (opcion)
    {
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
    case 4:
        resultado = modificar_saturacion(frame);
        nombre = "Saturacion aumentada";
        break;
    case 5:
        resultado = bgr_a_rgb(kmeans_manual(frame, 8));
        nombre = "KMeans Manual";
        break;
    case 6:
        resultado = gray_world(frame);
        nombre = "Gray World";
        break;
    case 7:
        resultado = correccion_gamma(frame, 2);
        nombre = "Gamma 0.5";
        break;
    case 8:
        resultado = correccion_vineteo(frame, 0.4);
        nombre = "Vineteo Corregido";
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
