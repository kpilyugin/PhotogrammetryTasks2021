#include "sift.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <libutils/rasserts.h>

// Ссылки:
// [lowe04] - Distinctive Image Features from Scale-Invariant Keypoints, David G. Lowe, 2004
//
// Примеры реализаций (стоит обращаться только если совсем не понятны какие-то места):
// 1) https://github.com/robwhess/opensift/blob/master/src/sift.c
// 2) https://gist.github.com/lxc-xx/7088609 (адаптация кода с первой ссылки)
// 3) https://github.com/opencv/opencv/blob/1834eed8098aa2c595f4d1099eeaa0992ce8b321/modules/features2d/src/sift.dispatch.cpp (адаптация кода с первой ссылки)
// 4) https://github.com/opencv/opencv/blob/1834eed8098aa2c595f4d1099eeaa0992ce8b321/modules/features2d/src/sift.simd.hpp (адаптация кода с первой ссылки)

#define DEBUG_ENABLE     1
#define DEBUG_PATH       std::string("data/debug/test_sift/debug/")

#define NOCTAVES                    3                    // число октав
#define OCTAVE_NLAYERS              3                    // в [lowe04] это число промежуточных степеней размытия картинки в рамках одной октавы обозначается - s, т.е. s слоев в каждой октаве
#define OCTAVE_GAUSSIAN_IMAGES      (OCTAVE_NLAYERS + 3)
#define OCTAVE_DOG_IMAGES           (OCTAVE_NLAYERS + 2)
#define INITIAL_IMG_SIGMA           0.75                 // предполагаемая степень размытия изначальной картинки

#define SUBPIXEL_FITTING_ENABLE      1
#define SUBPIXEL_FITTING_STEPS       5
#define ELIMINATE_EDGE_RESPONSE_ENABLE 1
#define DETECT_DUPLICATES            1

#define ORIENTATION_NHISTS           36   // число корзин при определении ориентации ключевой точки через гистограммы
#define ORIENTATION_WINDOW_R         3    // минимальный радиус окна в рамках которого будет выбрана ориентиация (в пикселях), R=3 => 5x5 окно
#define ORIENTATION_VOTES_PEAK_RATIO 0.80 // 0.8 => если гистограмма какого-то направления получила >= 80% от максимального чиссла голосов - она тоже победила

#define DESCRIPTOR_SIZE            4 // 4x4 гистограммы декскриптора
#define DESCRIPTOR_NBINS           8 // 8 корзин-направлений в каждой гистограмме дескриптора (4х4 гистограммы, каждая по 8 корзин, итого 4x4x8=128 значений в дескрипторе)
#define DESCRIPTOR_SAMPLES_N       4 // 4x4 замера для каждой гистограммы дескриптора (всего гистограмм 4х4) итого 16х16 замеров
#define DESCRIPTOR_SAMPLE_WINDOW_R 1.0 // минимальный радиус окна в рамках которого строится гистограмма из 8 корзин-направлений (т.е. для каждого из 16 элементов дескриптора), R=1 => 1x1 окно


void phg::SIFT::detectAndCompute(const cv::Mat &originalImg, std::vector<cv::KeyPoint> &kps, cv::Mat &desc) {
    // используйте дебаг в файлы как можно больше, это очень удобно и потраченное время окупается крайне сильно,
    // ведь пролистывать через окошки показывающие картинки долго, и по ним нельзя проматывать назад, а по файлам - можно
    // вы можете запустить алгоритм, сгенерировать десятки картинок со всеми промежуточными визуализациями и после запуска
    // посмотреть на те этапы к которым у вас вопросы или про которые у вас опасения
    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "00_input.png", originalImg);

    cv::Mat img = originalImg.clone();
    // для удобства используем черно-белую картинку и работаем с вещественными числами (это еще и может улучшить точность)
    if (originalImg.type() == CV_8UC1) { // greyscale image
        img.convertTo(img, CV_32FC1, 1.0);
    } else if (originalImg.type() == CV_8UC3) { // BGR image
        img.convertTo(img, CV_32FC3, 1.0);
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    } else {
        rassert(false, 14291409120);
    }
    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "01_grey.png", img);
    cv::GaussianBlur(img, img, cv::Size(0, 0), initial_blur_sigma, initial_blur_sigma);
    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "02_grey_blurred.png", img);

    // Scale-space extrema detection
    std::vector<cv::Mat> gaussianPyramid;
    std::vector<cv::Mat> DoGPyramid;
    buildPyramids(img, gaussianPyramid, DoGPyramid);

    findLocalExtremasAndDescribe(gaussianPyramid, DoGPyramid, kps, desc);
}

void phg::SIFT::buildPyramids(const cv::Mat &imgOrg, std::vector<cv::Mat> &gaussianPyramid, std::vector<cv::Mat> &DoGPyramid) {
    gaussianPyramid.resize(NOCTAVES * OCTAVE_GAUSSIAN_IMAGES);

    const double k = pow(2.0, 1.0 / OCTAVE_NLAYERS); // [lowe04] k = 2^{1/s} а у нас s=OCTAVE_NLAYERS: k ~ 1.25

    // строим пирамиду гауссовых размытий картинки
    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        if (octave == 0) {
            gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES] = imgOrg.clone();
        } else {
            size_t prevOctave = octave - 1;
            // берем картинку с предыдущей октавы и уменьшаем ее в два раза без какого бы то ни было дополнительного размытия (сигмы должны совпадать)
            int lastLayer = OCTAVE_GAUSSIAN_IMAGES - 1;
            int imageWithSameSigma = prevOctave * OCTAVE_GAUSSIAN_IMAGES + lastLayer - 2;
            cv::Mat img = gaussianPyramid[imageWithSameSigma].clone();
            // тут есть очень важный момент, мы должны указать fx=0.5, fy=0.5 иначе при нечетном размере картинка будет не идеально 2 пикселя в один схлопываться - а слегка смещаться
            cv::Size dstSize = cv::Size(0.5 * img.cols, 0.5 * img.rows);
            cv::resize(img, img, dstSize, 0.5, 0.5, cv::INTER_NEAREST);
            gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES] = img;
        }

        #pragma omp parallel for
        for (ptrdiff_t layer = 1; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            size_t prevLayer = layer - 1;

            // если есть два последовательных гауссовых размытия с sigma1 и sigma2, то результат будет с sigma12=sqrt(sigma1^2 + sigma2^2) => sigma2=sqrt(sigma12^2-sigma1^2)
            double sigmaPrev = INITIAL_IMG_SIGMA; // sigma1  - сигма до которой дошла картинка на предыдущем слое: на первом слое это INITIAL_SIGMA
            double sigmaCur  = INITIAL_IMG_SIGMA * pow(k, layer);     // sigma12 - сигма до которой мы хотим дойти на текущем слое
            double sigma = sqrt(sigmaCur * sigmaCur - sigmaPrev * sigmaPrev); // sigma2  - сигма которую надо добавить чтобы довести sigma1 до sigma12
            // посмотрите внимательно на формулу выше и решите как по мнению этой формулы соотносится сигма у первого А-слоя i-ой октавы
            // и сигма у одного из последних слоев Б предыдущей (i-1)-ой октавы из которого этот слой А был получен?
            // а как чисто идейно должны бы соотноситься сигмы размытия у двух картинок если картинка А была получена из картинки Б простым уменьшением в 2 раза?

            // добавочное размытие с варианта "размываем предыдущий слой" на вариант "размываем самый первый слой октавы до степени размытия сигмы нашего текущего слоя"
            // проверьте - картинки отладочного вывода выглядят один-в-один до/после? (посмотрите на них туда-сюда быстро мигая)
            // проверил, разницы нет
            cv::Mat imgLayer = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES].clone();
            cv::Size automaticKernelSize = cv::Size(0, 0);
            cv::GaussianBlur(imgLayer, imgLayer, automaticKernelSize, sigma, sigma);

            gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer] = imgLayer;
        }
    }

    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 0; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "pyramid/o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png", gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer]);
            // какие ожидания от картинок можно придумать? т.е. как дополнительно проверить что работает разумно?
            // картинка с i-го слоя октавы должна визуально совпадать с (i + 3)-й картинкой предыдущей октавы, т.к. у них одинаковое размытие
            // визуально проверить - открыть в редакторе в одинаковом размере окна (но удобнее проверять уже на DoG картинках, там степень размытия сразу видна
        }
    }

    DoGPyramid.resize(NOCTAVES * OCTAVE_DOG_IMAGES);

    // строим пирамиду разниц гауссиан слоев (Difference of Gaussian, DoG),
    // т.к. вычитать надо из слоя слой в рамках одной и той же октавы - то есть приятный параллелизм на уровне октав
    #pragma omp parallel for
    for (ptrdiff_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 1; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            int prevLayer = layer - 1;
            cv::Mat imgPrevGaussian = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + prevLayer];
            cv::Mat imgCurGaussian  = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer];
            cv::Mat imgCurDoG = imgCurGaussian - imgPrevGaussian;
            int dogLayer = layer - 1;
            DoGPyramid[octave * OCTAVE_DOG_IMAGES + dogLayer] = imgCurDoG;
        }
    }

    // нам нужны padding-картинки по краям октавы чтобы извлекать экстремумы, но в статье предлагается не s+2 а s+3:
    // [lowe04] We must produce s + 3 images in the stack of blurred images for each octave, so that final extrema detection covers a complete octave
    // почему OCTAVE_GAUSSIAN_IMAGES=(OCTAVE_NLAYERS + 3) а не например (OCTAVE_NLAYERS + 2)?
    // Нам нужно s+2 картинки в пирамиде DoG, из вычислений в предыдущем цикле видно что для этого нужна одна лишняя картинка в пирамиде гауссиан слоев
    // (т.к. пара (i, i+1) дают i-ю DoG )

    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 0; layer < OCTAVE_DOG_IMAGES; ++layer) {
            double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
            if (DEBUG_ENABLE) {
                const cv::Mat& DoGImage = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer];
                cv::imwrite(DEBUG_PATH + "pyramidDoG/o" + to_string(octave) +
                            "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png",
                            100 * DoGImage); // actual differences are small, multiply by factor 100 to see results
            }
            // картинка с i-го слоя октавы должна визуально совпадать с (i + 3)-й картинкой предыдущей октавы,
            // т.к. они получены из разности картинок с одинаковым размытием
        }
    }
}

namespace {
    float parabolaFitting(float x0, float x1, float x2) {
        rassert((x1 >= x0 && x1 >= x2) || (x1 <= x0 && x1 <= x2), 12541241241241);

        // a*0^2+b*0+c=x0
        // a*1^2+b*1+c=x1
        // a*2^2+b*2+c=x2

        // c=x0
        // a+b+x0=x1     (2)
        // 4*a+2*b+x0=x2 (3)

        // (3)-2*(2): 2*a-y0=y2-2*y1; a=(y2-2*y1+y0)/2
        // (2):       b=y1-y0-a
        float a = (x2-2.0f*x1+x0) / 2.0f;
        float b = x1 - x0 - a;
        // extremum is at -b/(2*a), but our system coordinate start (i) is at 1, so minus 1
        float shift = - b / (2.0f * a) - 1.0f;
        return shift;
    }

    double adjustAngle(double angle) {
        while (angle <  0.0)   angle += 360.0;
        while (angle >= 360.0) angle -= 360.0;
        return angle;
    }

    double extractOrientation(float dy, float dx) {
        double orientation = atan2(dy, dx);
        orientation = orientation * 180.0 / M_PI;
        return adjustAngle(orientation + 90.0);
    }

    // Returns false if keypoint should be discarded
    bool subpixelFitting(const std::vector<cv::Mat> &DoGPyramid,
                         size_t octave, size_t& layer, size_t& x, size_t& y,
                         float& xCorr, float& yCorr, float& layerCorr, float& valueCorr) {
        size_t baseIdx = octave * OCTAVE_DOG_IMAGES + layer;
        auto& cur = DoGPyramid[baseIdx];
        auto img = [&](int i, int j, int k) -> float {
            return DoGPyramid[baseIdx + k].at<float>(y + j, x + i);
        };

        for (int step = 0; step <= SUBPIXEL_FITTING_STEPS; ++step) {
            if (step == SUBPIXEL_FITTING_STEPS + 1) { // discard point if not converged
                return false;
            }

            float center = img(0, 0, 0);
            float dx = (img(1, 0, 0) - img(-1, 0, 0)) * 0.5;
            float dy = (img(0, 1, 0) - img(0, -1, 0)) * 0.5;
            float dz = (img(0, 0, 1) - img(0, 0, -1)) * 0.5;
            float dxx = img(1, 0, 0) + img(-1, 0, 0) - 2 * center;
            float dyy = img(0, 1, 0) + img(0, -1, 0) - 2 * center;
            float dzz = img(0, 0, 1) + img(0, 0, -1) - 2 * center;
            float dxy = (img(1, 1, 0) + img(-1, -1, 0) - img(1, -1, 0) - img(-1, 1, 0)) * 0.25;
            float dxz = (img(1, 0, 1) + img(-1, 0, -1) - img(1, 0, -1) - img(-1, 0, 1)) * 0.25;
            float dyz = (img(0, 1, 1) + img(0, -1, -1) - img(0, 1, -1) - img(0, -1, 1)) * 0.25;

            cv::Vec3f gradient(dx, dy, dz);
            cv::Matx33f H(dxx, dxy, dxz,
                          dxy, dyy, dyz,
                          dxz, dyz, dzz);
            cv::Vec3f X = H.solve(gradient, cv::DECOMP_LU);

            xCorr = -X[0];
            yCorr = -X[1];
            layerCorr = -X[2];
            valueCorr = gradient.dot(cv::Matx31f(xCorr, yCorr, layerCorr));
            if (abs(layerCorr) < 0.5f && abs(yCorr) < 0.5f && abs(xCorr) < 0.5f) {
                return true;
            }

            float maxCorr = 100;
            if (abs(layerCorr) > maxCorr || abs(yCorr) > maxCorr || abs(xCorr) > maxCorr) {
                return false;
            }

            x += round(xCorr);
            y += round(yCorr);
            layer += round(layerCorr);

            if (layer < 1 || layer > OCTAVE_NLAYERS || x < 1 || x >= cur.cols - 1 || y < 1 || y >= cur.rows - 1) {
                return false;
            }
        }
        return true;
    }

    // 4.1
    // For stability: discard keypoints along the edges, they can move along the edge
    // Returns false if keypoint should be discarded
    bool eliminateEdgeResponse(const std::vector<cv::Mat> &DoGPyramid,
                               size_t octave, size_t layer, size_t x, size_t y, float edgeThreshold) {
        size_t baseIdx = octave * OCTAVE_DOG_IMAGES + layer;
        auto img = [&](int i, int j, int k) -> float {
            return DoGPyramid[baseIdx + k].at<float>(y + j, x + i);
        };

        float center = img(0, 0, 0);
        float dxx = img(1, 0, 0) + img(-1, 0, 0) - 2 * center;
        float dyy = img(0, 1, 0) + img(0, -1, 0) - 2 * center;
        float dxy = (img(1, 1, 0) + img(-1, -1, 0) - img(1, -1, 0) - img(-1, 1, 0)) * 0.25;
        float trace = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;
        float r = edgeThreshold;
        if (det <= 0 || trace * trace * r >= (r + 1) * (r + 1) * det) {
            return false;
        }
        return true;
    }
}

void phg::SIFT::findLocalExtremasAndDescribe(const std::vector<cv::Mat> &gaussianPyramid,
                                             const std::vector<cv::Mat> &DoGPyramid,
                                             std::vector<cv::KeyPoint> &keyPoints, cv::Mat &desc) {
    std::vector<std::vector<float>> pointsDesc;

    // 3.1 Local extrema detection
    #pragma omp parallel // запустили каждый вычислительный поток процессора
    {
        // каждый поток будет складировать свои точки в свой личный вектор (чтобы не было гонок и не были нужны точки синхронизации)
        std::vector<cv::KeyPoint> thread_points;
        std::vector<std::vector<float>> thread_descriptors;

        for (size_t octave = 0; octave < NOCTAVES; ++octave) {
            double octave_downscale = pow(2.0, octave);
            for (size_t layer = 1; layer + 1 < OCTAVE_DOG_IMAGES; ++layer) {
                const cv::Mat prev = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer - 1];
                const cv::Mat cur  = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer];
                const cv::Mat next = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer + 1];
                const cv::Mat DoGs[3] = {prev, cur, next};

                // теперь каждый поток обработает свой кусок картинки
                #pragma omp for
                for (size_t j = 1; j < cur.rows - 1; ++j) {
                    for (size_t i = 1; i + 1 < cur.cols; ++i) {
                        bool is_max = true;
                        bool is_min = true;
                        float center = DoGs[1].at<float>(j, i);
                        for (int dz = -1; dz <= 1 && (is_min || is_max); ++dz) {
                            for (int dy = -1; dy <= 1 && (is_min || is_max); ++dy) {
                                for (int dx = -1; dx <= 1 && (is_min || is_max); ++dx) {
                                    if (dx != 0 || dy != 0 || dz != 0) {
                                        float neighbor = DoGs[1 + dz].at<float>(j + dy, i + dx);
                                        if (neighbor >= center) {
                                            is_max = false;
                                        }
                                        if (neighbor <= center) {
                                            is_min = false;
                                        }
                                    }
                                }
                            }
                        }
                        bool is_extremum = (is_min || is_max);
                        if (!is_extremum) {
                            continue; // очередной элемент cascade filtering, если не экстремум - сразу заканчиваем обработку этого пикселя
                        }

                        // 4 Accurate keypoint localization
                        cv::KeyPoint kp;
                        float xCorr = 0.0f;
                        float yCorr = 0.0f;
                        float layerCorr = 0.0f;
                        float valueCorr = 0.0f;

                        size_t localLayer = layer;
                        size_t x = i;
                        size_t y = j;
#if SUBPIXEL_FITTING_ENABLE
                        if (!subpixelFitting(DoGPyramid, octave, localLayer, x, y, xCorr, yCorr, layerCorr, valueCorr)) {
                            continue;
                        }
#endif
                        float contrast = fabs(center + valueCorr);
                        // почему порог контрастности должен уменьшаться при увеличении числа слоев в октаве?
                        // Больше слоев в октаве => меньше разница между размытиями соседних картинок которые мы вычитаем => меньше контраст в целом на картинках в DoG
                        if (contrast < contrast_threshold / OCTAVE_NLAYERS) {
                            continue;
                        }

#if ELIMINATE_EDGE_RESPONSE_ENABLE
                        if (!eliminateEdgeResponse(DoGPyramid, octave, localLayer, x, y, edge_threshold)) {
                            continue;
                        }
#endif
                        kp.pt = cv::Point2f((x + 0.5 + xCorr) * octave_downscale,
                                            (y + 0.5 + yCorr) * octave_downscale
                        );
                        kp.response = contrast;

                        const double k = pow(2.0, 1.0 / OCTAVE_NLAYERS); // [lowe04] k = 2^{1/s} а у нас s=OCTAVE_NLAYERS
                        double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, localLayer);
                        kp.size = 2.0 * sigmaCur * 5.0;

                        // 5 Orientation assignment
                        cv::Mat img = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + localLayer];
                        std::vector<float> votes;
                        float biggestVote;
                        int oriRadius = (int) (ORIENTATION_WINDOW_R * pow(k, localLayer + layerCorr));
                        if (!buildLocalOrientationHists(img, i, j, oriRadius, votes, biggestVote))
                            continue;

                        for (size_t bin = 0; bin < ORIENTATION_NHISTS; ++bin) {
                            float prevValue = votes[(bin + ORIENTATION_NHISTS - 1) % ORIENTATION_NHISTS];
                            float value = votes[bin];
                            float nextValue = votes[(bin + 1) % ORIENTATION_NHISTS];
                            if (value > prevValue && value > nextValue && votes[bin] > biggestVote * ORIENTATION_VOTES_PEAK_RATIO) {
                                float correction = parabolaFitting(prevValue, value, nextValue);
                                kp.angle = (bin + 0.5 + correction) * (360.0 / ORIENTATION_NHISTS);
                                rassert(kp.angle >= 0.0 && kp.angle <= 360.0, 123512412412);

                                std::vector<float> descriptor;
                                double descrSampleRadius = (DESCRIPTOR_SAMPLE_WINDOW_R * pow(k, localLayer + layerCorr));
                                if (!buildDescriptor(img, kp.pt.x, kp.pt.y, descrSampleRadius, kp.angle, descriptor))
                                    continue;

                                thread_points.push_back(kp);
                                thread_descriptors.push_back(descriptor);
                            }
                        }
                    }
                }
            }
        }

        // в критической секции объединяем все массивы детектированных точек
        #pragma omp critical
        {
            keyPoints.insert(keyPoints.end(), thread_points.begin(), thread_points.end());
            pointsDesc.insert(pointsDesc.end(), thread_descriptors.begin(), thread_descriptors.end());
        }
    }

#if DETECT_DUPLICATES
    size_t i = 1;
    int numDuplicates = 0;
    while (i < keyPoints.size()) {
        auto cur = keyPoints[i];
        auto prev = keyPoints[i - 1];
        if (cur.pt.x == prev.pt.x && cur.pt.y == prev.pt.y) {
            keyPoints.erase(keyPoints.begin() + i);
            pointsDesc.erase(pointsDesc.begin() + i);
            numDuplicates++;
        } else {
            i++;
        }
    }
    std::cout << "Keypoint duplicates: " << numDuplicates << "\n";
#endif

    rassert(pointsDesc.size() == keyPoints.size(), 12356351235124);
    desc = cv::Mat(pointsDesc.size(), DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, CV_32FC1);
    for (size_t j = 0; j < pointsDesc.size(); ++j) {
        rassert(pointsDesc[j].size() == DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, 1253351412421);
        for (size_t i = 0; i < pointsDesc[i].size(); ++i) {
            desc.at<float>(j, i) = pointsDesc[j][i];
        }
    }
}

bool phg::SIFT::buildLocalOrientationHists(const cv::Mat &img, size_t i, size_t j, size_t radius,
                                           std::vector<float> &votes, float &biggestVote) {
    // 5 Orientation assignment
    votes.resize(ORIENTATION_NHISTS, 0.0f);
    biggestVote = 0.0;

    if (i - 1 < radius - 1 || i + 1 + radius - 1 >= img.cols ||
        j - 1 < radius - 1 || j + 1 + radius - 1 >= img.rows) {
        return false;
    }

    float sum[ORIENTATION_NHISTS] = {0.0f};

    for (size_t y = j - radius + 1; y < j + radius; ++y) {
        for (size_t x = i - radius + 1; x < i + radius; ++x) {
            double dx = img.at<float>(y, x + 1) - img.at<float>(y, x - 1);
            double dy = img.at<float>(y + 1, x) - img.at<float>(y - 1, x);
            // m(x, y)=(L(x + 1, y) − L(x − 1, y))^2 + (L(x, y + 1) − L(x, y − 1))^2
            double magnitude = sqrt(dx * dx + dy * dy);

            // atan( (L(x, y + 1) − L(x, y − 1)) / (L(x + 1, y) − L(x − 1, y)) )
            double orientation = extractOrientation(dy, dx);
            rassert(orientation >= 0.0 && orientation < 360.0, 5361615612);
            static_assert(360 % ORIENTATION_NHISTS == 0, "Inappropriate bins number!");
            size_t bin = orientation * ORIENTATION_NHISTS / 360.0;
            rassert(bin < ORIENTATION_NHISTS, 361236315613);
            sum[bin] += magnitude;
            // TODO может быть сгладить получившиеся гистограммы улучшит результат?
        }
    }

    for (size_t bin = 0; bin < ORIENTATION_NHISTS; ++bin) {
        votes[bin] = sum[bin];
        biggestVote = std::max(biggestVote, sum[bin]);
    }

    return true;
}

bool phg::SIFT::buildDescriptor(const cv::Mat &img, float px, float py, double descrSampleRadius, float angle,
                                std::vector<float> &descriptor) {
    cv::Mat relativeShiftRotation = cv::getRotationMatrix2D(cv::Point2f(0.0f, 0.0f), -angle, 1.0);

    const double smpW = 2.0 * descrSampleRadius - 1.0;

    descriptor.resize(DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, 0.0f);
    for (int hstj = 0; hstj < DESCRIPTOR_SIZE; ++hstj) { // перебираем строку в решетке гистограмм
        for (int hsti = 0; hsti < DESCRIPTOR_SIZE; ++hsti) { // перебираем колонку в решетке гистограмм

            float sum[DESCRIPTOR_NBINS] = {0.0f};

            for (int smpj = 0; smpj < DESCRIPTOR_SAMPLES_N; ++smpj) { // перебираем строчку замера для текущей гистограммы
                for (int smpi = 0; smpi < DESCRIPTOR_SAMPLES_N; ++smpi) { // перебираем столбик очередного замера для текущей гистограммы
                    double xShift = ((-DESCRIPTOR_SIZE / 2.0 + hsti) * DESCRIPTOR_SAMPLES_N + smpi) * smpW;
                    double yShift = ((-DESCRIPTOR_SIZE / 2.0 + hstj) * DESCRIPTOR_SAMPLES_N + smpj) * smpW;
                    cv::Point2f shift(xShift, yShift);
                    std::vector<cv::Point2f> shiftInVector(1, shift);
                    cv::transform(shiftInVector, shiftInVector, relativeShiftRotation); // преобразуем относительный сдвиг с учетом ориентации ключевой точки
                    shift = shiftInVector[0];

                    int x = (int) (px + shift.x);
                    int y = (int) (py + shift.y);

                    int offset = smpW;
                    if (y - offset < 0 || y + offset >= img.rows || x - offset < 0 || x + offset >= img.cols) {
                        return false;
                    }

                    double dx = img.at<float>(y, x + offset) - img.at<float>(y, x - offset);
                    double dy = img.at<float>(y + offset, x) - img.at<float>(y - offset, x);
                    double magnitude = sqrt(dx * dx + dy * dy);
                    double orientation = extractOrientation(dy, dx);

                    // за счет чего этот вклад будет сравниваться с этим же вкладом даже если эта картинка будет повернута?
                    // что нужно сделать с ориентацией каждого градиента из окрестности этой ключевой точки?
                    // Нужно вычесть из получившейся ориентации градиента угол ключевой точки
                    orientation = adjustAngle(orientation - angle);

                    rassert(orientation >= 0.0 && orientation < 360.0, 3515215125412);
                    static_assert(360 % DESCRIPTOR_NBINS == 0, "Inappropriate bins number!");
                    size_t bin = orientation * DESCRIPTOR_NBINS / 360.0;
                    rassert(bin < DESCRIPTOR_NBINS, 361236315613);
                    sum[bin] += magnitude;
                    // TODO хорошая идея добавить трилинейную интерполяцию как предложено в статье, или хотя бы сэмулировать ее - сгладить получившиеся гистограммы
                }
            }

            float norm = 0;
            for (float bin : sum) {
                norm += bin * bin;
            }
            norm = sqrt(norm);
            for (float& bin : sum) {
                bin *= 1.0 / norm;
            }

            float *votes = &(descriptor[(hstj * DESCRIPTOR_SIZE + hsti) * DESCRIPTOR_NBINS]); // нашли где будут лежать корзины нашей гистограммы
            for (int bin = 0; bin < DESCRIPTOR_NBINS; ++bin) {
                votes[bin] = sum[bin];
            }
        }
    }
    return true;
}
