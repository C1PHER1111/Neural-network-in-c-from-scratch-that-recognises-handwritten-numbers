#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <numeric>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static constexpr double MNIST_MEAN = 0.1307;   // standard MNIST normalization
static constexpr double LEAK       = 0.01;

// ============================================================================
// 1. FLAT MATRIX ENGINE (column-vector fast path)
// ============================================================================
class Matrix {
public:
    int rows;
    int cols;
    std::vector<double> data;

    Matrix() : rows(1), cols(1), data(1, 0.0) {}
    Matrix(int r, int c, double initial_value = 0.0)
        : rows(r), cols(c), data(static_cast<size_t>(r) * c, initial_value) {}

    inline double& at(int i, int j)       { return data[static_cast<size_t>(i) * cols + j]; }
    inline double  at(int i, int j) const { return data[static_cast<size_t>(i) * cols + j]; }

    static Matrix he(int r, int c) {
        Matrix m(r, c);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dis(0.0, std::sqrt(2.0 / c));
        for (auto& v : m.data) v = dis(gen);
        return m;
    }

    // this: (out x in)  x: (in x 1)  ->  out: (out x 1). One contiguous pass, no zero-fill.
    void matvec_into(const Matrix& x, Matrix& out) const {
        const double* xv = x.data.data();
        for (int i = 0; i < rows; ++i) {
            const double* w = data.data() + static_cast<size_t>(i) * cols;
            double s = 0.0;
            for (int k = 0; k < cols; ++k) s += w[k] * xv[k];
            out.data[i] = s;
        }
    }

    void add_scaled_in_place(const Matrix& other, double scale) {
        for (size_t i = 0; i < data.size(); ++i) data[i] += other.data[i] * scale;
    }
};

// ============================================================================
// 2. NEURAL NETWORK (pre-allocated buffers only, momentum, cached transposes)
// ============================================================================
class NeuralNetwork {
private:
    std::vector<int> dims;
    int L;
    double momentum;

    std::vector<Matrix> weights, biases, weights_T;      // params + cached transposes
    std::vector<Matrix> nabla_w, nabla_b;                // gradient accumulators
    std::vector<Matrix> vel_w, vel_b;                    // momentum velocity
    std::vector<Matrix> activations;                     // dims[i] x 1
    std::vector<Matrix> z_values;                        // dims[i+1] x 1
    std::vector<Matrix> errors;                          // dims[i+1] x 1  (one per layer)

    void refresh_transpose(int i) {
        for (int r = 0; r < weights[i].rows; ++r)
            for (int c = 0; c < weights[i].cols; ++c)
                weights_T[i].at(c, r) = weights[i].at(r, c);
    }

public:
    explicit NeuralNetwork(const std::vector<int>& layer_dims, double mom = 0.9)
        : dims(layer_dims), L(static_cast<int>(layer_dims.size()) - 1), momentum(mom) {
        for (int i = 0; i < L; ++i) {
            weights.push_back(Matrix::he(dims[i + 1], dims[i]));
            biases.emplace_back(dims[i + 1], 1, 0.0);
            weights_T.emplace_back(dims[i], dims[i + 1], 0.0);
            refresh_transpose(i);
            nabla_w.emplace_back(dims[i + 1], dims[i], 0.0);
            nabla_b.emplace_back(dims[i + 1], 1, 0.0);
            vel_w.emplace_back(dims[i + 1], dims[i], 0.0);
            vel_b.emplace_back(dims[i + 1], 1, 0.0);
        }
        for (int d : dims) activations.emplace_back(d, 1, 0.0);
        for (int i = 0; i < L; ++i) {
            z_values.emplace_back(dims[i + 1], 1, 0.0);
            errors.emplace_back(dims[i + 1], 1, 0.0);
        }
    }

    const Matrix& forward(const Matrix& input) {
        std::copy(input.data.begin(), input.data.end(), activations[0].data.begin());
        for (int i = 0; i < L; ++i) {
            weights[i].matvec_into(activations[i], z_values[i]);
            z_values[i].add_scaled_in_place(biases[i], 1.0);

            Matrix& a = activations[i + 1];
            const Matrix& z = z_values[i];

            if (i == L - 1) {                       // softmax
                double m = *std::max_element(z.data.begin(), z.data.end());
                double sum = 0.0;
                for (int r = 0; r < z.rows; ++r) {
                    a.data[r] = std::exp(std::clamp(z.data[r] - m, -30.0, 30.0));
                    sum += a.data[r];
                }
                double inv = 1.0 / sum;
                for (int r = 0; r < z.rows; ++r) a.data[r] *= inv;
            } else {                                // leaky ReLU
                for (int r = 0; r < z.rows; ++r)
                    a.data[r] = z.data[r] > 0.0 ? z.data[r] : LEAK * z.data[r];
            }
        }
        return activations.back();
    }

    void backward(const Matrix& target) {
        Matrix* errp = &errors[L - 1];
        const Matrix& out = activations.back();
        for (int r = 0; r < errp->rows; ++r)
            errp->data[r] = out.data[r] - target.data[r];   // softmax + cross-entropy

        for (int i = L - 1; i >= 0; --i) {
            Matrix& err = *errp;
            const Matrix& a_prev = activations[i];
            Matrix& gW = nabla_w[i];
            Matrix& gb = nabla_b[i];
            const int R = err.rows, C = a_prev.rows;

            for (int r = 0; r < R; ++r) {                   // accumulate grads in place
                const double e = err.data[r];
                gb.data[r] += e;
                double* gw = gW.data.data() + static_cast<size_t>(r) * C;
                const double* av = a_prev.data.data();
                for (int c = 0; c < C; ++c) gw[c] += e * av[c];
            }

            if (i > 0) {
                weights_T[i].matvec_into(err, errors[i - 1]);   // propagate
                const Matrix& zp = z_values[i - 1];
                Matrix& ep = errors[i - 1];
                for (int r = 0; r < ep.rows; ++r)
                    ep.data[r] *= (zp.data[r] > 0.0 ? 1.0 : LEAK);
                errp = &errors[i - 1];
            }
        }
    }

    void update(double lr, int batch_n) {
        const double s = lr / batch_n;
        for (int i = 0; i < L; ++i) {
            size_t sz = weights[i].data.size();
            double* w  = weights[i].data.data();
            double* v  = vel_w[i].data.data();
            const double* g = nabla_w[i].data.data();
            for (size_t k = 0; k < sz; ++k) { v[k] = momentum * v[k] - s * g[k]; w[k] += v[k]; }

            size_t bz = biases[i].data.size();
            double* b  = biases[i].data.data();
            double* vb = vel_b[i].data.data();
            const double* gb = nabla_b[i].data.data();
            for (size_t k = 0; k < bz; ++k) { vb[k] = momentum * vb[k] - s * gb[k]; b[k] += vb[k]; }

            refresh_transpose(i);   // weights changed -> refresh cache once per update
        }
    }

    void train_batch(const std::vector<Matrix>& images, const std::vector<Matrix>& targets,
                     const std::vector<size_t>& batch, double lr) {
        for (int i = 0; i < L; ++i) {
            std::fill(nabla_w[i].data.begin(), nabla_w[i].data.end(), 0.0);
            std::fill(nabla_b[i].data.begin(), nabla_b[i].data.end(), 0.0);
        }
        for (size_t t = 0; t < batch.size(); ++t) {
            forward(images[batch[t]]);
            backward(targets[batch[t]]);
        }
        update(lr, static_cast<int>(batch.size()));
    }

    int predict(const Matrix& input) {
        const Matrix& out = forward(input);
        int best = 0;
        for (int d = 1; d < 10; ++d)
            if (out.data[d] > out.data[best]) best = d;
        return best;
    }

    double evaluate(const std::vector<Matrix>& images, const std::vector<int>& labels) {
        int correct = 0;
        for (size_t i = 0; i < images.size(); ++i)
            if (predict(images[i]) == labels[i]) ++correct;
        return 100.0 * correct / static_cast<double>(images.size());
    }
};

// ============================================================================
// 3. IMAGE LOADING, SEGMENTATION & MNIST-FORMAT CONVERSION
// ============================================================================
// A loaded grayscale image, kept in two forms:
//   ink : normalized so that INK IS BRIGHT  (what the network is trained on)
//   view: normalized original             (for saving annotated output)
struct GrayImage {
    int w = 0, h = 0;
    bool ok = false;
    std::vector<double> ink, view;
};

GrayImage load_gray(const std::string& filepath) {
    GrayImage gi;
    int channels = 0;
    unsigned char* img_data = stbi_load(filepath.c_str(), &gi.w, &gi.h, &channels, 1);
    if (!img_data || gi.w <= 0 || gi.h <= 0) {
        if (img_data) stbi_image_free(img_data);
        return gi;
    }

    double corner_avg = (img_data[0] + img_data[gi.w - 1] +
                         img_data[(gi.h - 1) * gi.w] +
                         img_data[gi.h * gi.w - 1]) / 4.0;
    bool invert = (corner_avg > 128.0);   // white page -> dark ink -> invert

    const size_t n = static_cast<size_t>(gi.w) * gi.h;
    gi.ink.resize(n);
    gi.view.resize(n);
    for (size_t p = 0; p < n; ++p) {
        double v = img_data[p] / 255.0;
        gi.view[p] = v;
        gi.ink[p]  = invert ? 1.0 - v : v;
    }
    stbi_image_free(img_data);
    gi.ok = true;
    return gi;
}

double sample_bilinear(const std::vector<double>& g, int W, int H, double x, double y) {
    x = std::clamp(x, 0.0, W - 1.0);
    y = std::clamp(y, 0.0, H - 1.0);
    int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
    double fx = x - x0, fy = y - y0;
    int x1 = std::min(x0 + 1, W - 1), y1 = std::min(y0 + 1, H - 1);
    double a = g[static_cast<size_t>(y0) * W + x0], b = g[static_cast<size_t>(y0) * W + x1];
    double c = g[static_cast<size_t>(y1) * W + x0], d = g[static_cast<size_t>(y1) * W + x1];
    return (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy;
}

// Render one detected digit box into a 784-vector, MNIST-style:
// fit longest side into 20px inside a 28px canvas, center by mass, normalize.
Matrix box_to_mnist(const GrayImage& gi, int x0, int y0, int x1, int y1) {
    int bw = x1 - x0 + 1, bh = y1 - y0 + 1;
    double scale = 20.0 / std::max({1, bw, bh});
    int tw = std::max(1, static_cast<int>(std::lround(bw * scale)));
    int th = std::max(1, static_cast<int>(std::lround(bh * scale)));
    int ox = (28 - tw) / 2, oy = (28 - th) / 2;

    std::vector<double> img(784, 0.0);
    for (int ty = 0; ty < th; ++ty) {
        for (int tx = 0; tx < tw; ++tx) {
            double sx = x0 + (tx + 0.5) * bw / tw - 0.5;
            double sy = y0 + (ty + 0.5) * bh / th - 0.5;
            img[static_cast<size_t>(oy + ty) * 28 + (ox + tx)] =
                sample_bilinear(gi.ink, gi.w, gi.h, sx, sy);
        }
    }

    // shift so the ink's center of mass lands at the canvas center (like MNIST)
    double mx = 0.0, my = 0.0, mass = 0.0;
    for (int y = 0; y < 28; ++y)
        for (int x = 0; x < 28; ++x) {
            double v = img[static_cast<size_t>(y) * 28 + x];
            if (v > 0.0) { mx += v * x; my += v * y; mass += v; }
        }
    if (mass > 0.0) { mx /= mass; my /= mass; }
    int shx = static_cast<int>(std::lround(13.5 - mx));
    int shy = static_cast<int>(std::lround(13.5 - my));

    Matrix out(784, 1, 0.0);
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            int sx = x - shx, sy = y - shy;
            if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28)
                out.data[static_cast<size_t>(y) * 28 + x] =
                    img[static_cast<size_t>(sy) * 28 + sx] - MNIST_MEAN;
        }
    }
    return out;
}

// Whole-image fallback for a SINGLE handwritten digit (old behavior).
Matrix load_and_preprocess_image(const GrayImage& gi) {
    int min_x = gi.w, max_x = -1, min_y = gi.h, max_y = -1;
    const double threshold = 0.12;
    for (int y = 0; y < gi.h; ++y)
        for (int x = 0; x < gi.w; ++x) {
            if (gi.ink[static_cast<size_t>(y) * gi.w + x] > threshold) {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    if (max_x < 0) return Matrix(784, 1, -MNIST_MEAN);
    return box_to_mnist(gi, min_x, min_y, max_x, max_y);
}

// ------------------------- multi-digit segmentation -------------------------
struct DigitBox {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    int digit = -1;
    double confidence = 0.0;
};

static void dilate3(std::vector<unsigned char>& m, int W, int H) {
    std::vector<unsigned char> src = m;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            unsigned char v = 0;
            for (int dy = -1; dy <= 1 && !v; ++dy)
                for (int dx = -1; dx <= 1 && !v; ++dx) {
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < W && ny >= 0 && ny < H)
                        v = src[static_cast<size_t>(ny) * W + nx];
                }
            m[static_cast<size_t>(y) * W + x] = v;
        }
}

// Find candidate digit boxes: threshold ink -> dilate (joins dotted strokes)
// -> connected components -> filter out page border / ruled lines / specks.
std::vector<DigitBox> find_digit_boxes(const GrayImage& gi) {
    const int W = gi.w, H = gi.h;
    const double thr = 0.10;
    const size_t n = static_cast<size_t>(W) * H;

    std::vector<unsigned char> m(n);
    for (size_t i = 0; i < n; ++i) m[i] = gi.ink[i] > thr ? 1 : 0;

    // Erase long horizontal/vertical runs FIRST: ruled lines and the page border
    // would otherwise touch digit strokes and merge everything into one giant
    // component (which is then discarded as "too big").
    auto erase_long_runs = [&](bool horizontal) {
        const int outer = horizontal ? H : W;
        const int inner = horizontal ? W : H;
        const int limit   = static_cast<int>(0.18 * inner) + 1;
        for (int a = 0; a < outer; ++a) {
            int run_start = -1;
            for (int b = 0; b <= inner; ++b) {
                bool on = b < inner &&
                          m[horizontal ? static_cast<size_t>(a) * W + b
                                       : static_cast<size_t>(b) * W + a];
                if (on && run_start < 0) run_start = b;
                if (!on && run_start >= 0) {
                    if (b - run_start > limit)
                        for (int k = run_start; k < b; ++k)
                            m[horizontal ? static_cast<size_t>(a) * W + k
                                         : static_cast<size_t>(k) * W + a] = 0;
                    run_start = -1;
                }
            }
        }
    };
    erase_long_runs(true);
    erase_long_runs(false);

    dilate3(m, W, H);
    dilate3(m, W, H);

    std::vector<int> lab(n, -1);
    std::vector<DigitBox> boxes;
    std::vector<int> stack;
    int cur_label = 0;
    const int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};

    for (int y0 = 0; y0 < H; ++y0) {
        for (int x0 = 0; x0 < W; ++x0) {
            size_t s0 = static_cast<size_t>(y0) * W + x0;
            if (!m[s0] || lab[s0] >= 0) continue;

            stack.clear();
            stack.push_back(static_cast<int>(s0));
            lab[s0] = cur_label;
            DigitBox b; b.x0 = b.x1 = x0; b.y0 = b.y1 = y0;
            long area = 0;

            while (!stack.empty()) {
                int p = stack.back(); stack.pop_back();
                int x = p % W, y = p / W;
                ++area;
                if (x < b.x0) b.x0 = x; if (x > b.x1) b.x1 = x;
                if (y < b.y0) b.y0 = y; if (y > b.y1) b.y1 = y;
                for (int k = 0; k < 4; ++k) {
                    int nx = x + dx[k], ny = y + dy[k];
                    if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                    size_t q = static_cast<size_t>(ny) * W + nx;
                    if (m[q] && lab[q] < 0) { lab[q] = cur_label; stack.push_back(static_cast<int>(q)); }
                }
            }

            int bw = b.x1 - b.x0 + 1, bh = b.y1 - b.y0 + 1;
            double aspect = std::max(bw, bh) / static_cast<double>(std::max(1, std::min(bw, bh)));
            bool too_big  = bw > 0.50 * W || bh > 0.55 * H ||
                            area > 0.25 * static_cast<long>(W) * H;   // page border etc.
            bool too_small = area < std::max(20, static_cast<int>(0.00015 * W * H)) ||
                             std::min(bw, bh) < 4;                     // specks, noise
            bool line_like = aspect > 14.0;                            // ruled lines
            if (!too_big && !too_small && !line_like) boxes.push_back(b);
            ++cur_label;
        }
    }

    // merge fragments of the same digit that dilation failed to join
    bool merged = true;
    while (merged) {
        merged = false;
        for (size_t i = 0; i < boxes.size() && !merged; ++i) {
            for (size_t j = i + 1; j < boxes.size(); ++j) {
                int gapx = std::max(boxes[i].x0, boxes[j].x0) - std::min(boxes[i].x1, boxes[j].x1);
                int gapy = std::max(boxes[i].y0, boxes[j].y0) - std::min(boxes[i].y1, boxes[j].y1);
                if (gapx <= 6 && gapy <= 6) {
                    boxes[i].x0 = std::min(boxes[i].x0, boxes[j].x0);
                    boxes[i].y0 = std::min(boxes[i].y0, boxes[j].y0);
                    boxes[i].x1 = std::max(boxes[i].x1, boxes[j].x1);
                    boxes[i].y1 = std::max(boxes[i].y1, boxes[j].y1);
                    boxes.erase(boxes.begin() + static_cast<long>(j));
                    merged = true;
                    break;
                }
            }
        }
    }
    return boxes;
}

// Group boxes into reading order: rows top-to-bottom, left-to-right within a row.
std::vector<std::vector<const DigitBox*>> cluster_rows(const std::vector<DigitBox>& boxes) {
    std::vector<const DigitBox*> bs;
    for (const auto& b : boxes) bs.push_back(&b);
    std::sort(bs.begin(), bs.end(), [](const DigitBox* a, const DigitBox* b) {
        return (a->y0 + a->y1) < (b->y0 + b->y1);
    });

    std::vector<int> hs;
    for (const DigitBox* p : bs) hs.push_back(p->y1 - p->y0 + 1);
    std::nth_element(hs.begin(), hs.begin() + static_cast<long>(hs.size() / 2), hs.end());
    double med_h = std::max(1, hs[hs.size() / 2]);

    std::vector<std::vector<const DigitBox*>> rows;
    double row_cy = -1e9;
    for (const DigitBox* p : bs) {
        double cy = 0.5 * (p->y0 + p->y1);
        if (rows.empty() || std::fabs(cy - row_cy) > 0.6 * med_h) {
            rows.emplace_back();
            row_cy = cy;
        }
        rows.back().push_back(p);
    }
    for (auto& r : rows)
        std::sort(r.begin(), r.end(), [](const DigitBox* a, const DigitBox* b) {
            return a->x0 < b->x0;
        });
    return rows;
}

// Save a copy of the page with a red rectangle around every detected digit.
void save_annotated_ppm(const std::string& path, const GrayImage& gi,
                        const std::vector<DigitBox>& boxes) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return;
    f << "P6\n" << gi.w << " " << gi.h << "\n255\n";
    std::vector<unsigned char> px(static_cast<size_t>(gi.w) * gi.h * 3);
    for (size_t i = 0; i < static_cast<size_t>(gi.w) * gi.h; ++i) {
        unsigned char v = static_cast<unsigned char>(std::clamp(gi.view[i] * 255.0, 0.0, 255.0));
        px[i * 3] = v; px[i * 3 + 1] = v; px[i * 3 + 2] = v;
    }
    auto set_px = [&](int x, int y) {
        if (x < 0 || x >= gi.w || y < 0 || y >= gi.h) return;
        size_t i = (static_cast<size_t>(y) * gi.w + x) * 3;
        px[i] = 255; px[i + 1] = 0; px[i + 2] = 0;
    };
    for (const DigitBox& b : boxes) {
        for (int x = b.x0; x <= b.x1; ++x) { set_px(x, b.y0); set_px(x, b.y1); }
        for (int y = b.y0; y <= b.y1; ++y) { set_px(b.x0, y); set_px(b.x1, y); }
    }
    f.write(reinterpret_cast<const char*>(px.data()), static_cast<std::streamsize>(px.size()));
}

// ============================================================================
// 4. MNIST PARSER
// ============================================================================
// FIXED byte reversal (was: last byte c1 instead of c4 -> garbage dimensions).
int reverse_int(int i) {
    unsigned char c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return (static_cast<int>(c1) << 24) | (static_cast<int>(c2) << 16) |
           (static_cast<int>(c3) << 8)  |  static_cast<int>(c4);
}

std::vector<Matrix> load_mnist_images(const std::string& filename, int max_samples) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open image file: " + filename);

    int magic = 0, num_images = 0, rows = 0, cols = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    magic = reverse_int(magic);
    num_images = reverse_int(num_images);
    rows = reverse_int(rows);
    cols = reverse_int(cols);

    if (magic != 2051 || rows != 28 || cols != 28)
        throw std::runtime_error("Bad MNIST image file (magic=" + std::to_string(magic) + ")");

    num_images = std::min(num_images, max_samples);
    std::vector<Matrix> images;
    images.reserve(num_images);
    std::vector<char> buf(784);
    for (int i = 0; i < num_images; ++i) {
        file.read(buf.data(), 784);
        Matrix img(784, 1);
        for (int p = 0; p < 784; ++p)
            img.data[p] = static_cast<unsigned char>(buf[p]) / 255.0 - MNIST_MEAN;
        images.push_back(std::move(img));
    }
    return images;
}

std::vector<int> load_mnist_labels(const std::string& filename, int max_samples) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open label file: " + filename);

    int magic = 0, num_labels = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);
    magic = reverse_int(magic);
    num_labels = reverse_int(num_labels);

    if (magic != 2049) throw std::runtime_error("Bad MNIST label file");

    num_labels = std::min(num_labels, max_samples);
    std::vector<int> labels;
    labels.reserve(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels.push_back(static_cast<int>(label));
    }
    return labels;
}

// ============================================================================
// 5. MAIN
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << "====================================================\n";
    std::cout << "  C++ MNIST Recognizer (fixed + multi-digit)        \n";
    std::cout << "====================================================\n\n";

    int sample_count = 30000;
    std::string target_image;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--samples" && i + 1 < argc) sample_count = std::stoi(argv[++i]);
        else if (a == "--image" && i + 1 < argc) target_image = argv[++i];
        else if (a == "--" && i + 1 < argc) { target_image = argv[i + 1]; break; }
        else if (a.rfind("--", 0) != 0) target_image = a;
    }

    std::vector<Matrix> train_images;
    std::vector<int> train_labels;
    try {
        std::cout << "Loading MNIST training binaries...\n";
        train_images = load_mnist_images("train-images-idx3-ubyte", sample_count);
        train_labels = load_mnist_labels("train-labels-idx1-ubyte", sample_count);
        std::cout << "Loaded " << train_images.size() << " samples successfully.\n\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // one-hot targets, built ONCE (was: allocated per sample per epoch)
    std::vector<Matrix> train_targets(train_images.size());
    for (size_t i = 0; i < train_images.size(); ++i) {
        train_targets[i] = Matrix(10, 1, 0.0);
        train_targets[i].data[train_labels[i]] = 1.0;
    }

    NeuralNetwork nn({784, 128, 10}, /*momentum=*/0.9);

    const double target_accuracy = 95.0;
    const int    batch_size      = 32;
    const size_t max_epochs      = 200;
    double lr = 0.1;
    double acc = 0.0, best_acc = -1.0;
    int stale = 0, epoch = 0;

    std::vector<size_t> indices(train_images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());

    std::cout << "Training (target " << target_accuracy << "%)...\n\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    while (acc < target_accuracy && static_cast<size_t>(epoch) < max_epochs) {
        ++epoch;
        std::shuffle(indices.begin(), indices.end(), g);

        for (size_t i = 0; i < train_images.size(); i += batch_size) {
            size_t end = std::min(i + static_cast<size_t>(batch_size), train_images.size());
            std::vector<size_t> batch(indices.begin() + i, indices.begin() + end);
            nn.train_batch(train_images, train_targets, batch, lr);
        }

        acc = nn.evaluate(train_images, train_labels);
        std::cout << "[Epoch " << std::setw(3) << epoch << "] Accuracy: "
                  << std::fixed << std::setprecision(2) << acc << "%  (lr " << lr << ")\n";

        if (acc > best_acc) { best_acc = acc; stale = 0; }
        else if (++stale >= 3 && lr > 0.002) {
            lr *= 0.5; stale = 0;
            std::cout << "  -> plateau, halving learning rate to " << lr << "\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "\n>>> " << (acc >= target_accuracy ? "Target reached" : "Stopped at max epochs")
              << " in " << elapsed.count() << " s (best " << best_acc << "%)\n";

    if (!target_image.empty()) {
        std::cout << "\n----------------------------------------------------\n";
        std::cout << " Testing external image: " << target_image << "\n";
        std::cout << "----------------------------------------------------\n";
        try {
            GrayImage gi = load_gray(target_image);
            if (!gi.ok) throw std::runtime_error("Could not decode image: " + target_image);

            std::vector<DigitBox> boxes = find_digit_boxes(gi);

            if (boxes.empty()) {
                // single-digit fallback (old behavior)
                Matrix vec = load_and_preprocess_image(gi);
                const Matrix& probs = nn.forward(vec);
                int d = 0;
                for (int k = 1; k < 10; ++k)
                    if (probs.data[k] > probs.data[d]) d = k;
                std::cout << "Single-digit mode. Probabilities (0-9):\n";
                for (int k = 0; k < 10; ++k)
                    std::cout << "  Digit " << k << ": " << std::fixed << std::setprecision(2)
                              << probs.data[k] * 100.0 << "%\n";
                std::cout << "\n>>> Recognized: " << d
                          << " (confidence " << probs.data[d] * 100.0 << "%)\n";
            } else {
                // classify every detected digit
                for (auto& b : boxes) {
                    Matrix vec = box_to_mnist(gi, b.x0, b.y0, b.x1, b.y1);
                    const Matrix& probs = nn.forward(vec);
                    b.digit = 0;
                    for (int k = 1; k < 10; ++k)
                        if (probs.data[k] > probs.data[b.digit]) b.digit = k;
                    b.confidence = probs.data[b.digit];
                }

                auto rows = cluster_rows(boxes);
                std::cout << "Found " << boxes.size() << " digit(s) in "
                          << rows.size() << " row(s):\n\n";
                for (size_t r = 0; r < rows.size(); ++r) {
                    std::cout << "Row " << (r + 1) << ":  ";
                    for (const DigitBox* p : rows[r])
                        std::cout << p->digit << " (" << std::fixed << std::setprecision(0)
                                  << p->confidence * 100.0 << "%)   ";
                    std::cout << "\n";
                }
                save_annotated_ppm("annotated.ppm", gi, boxes);
                std::cout << "\nAnnotated image with detection boxes saved to annotated.ppm\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Inference error: " << e.what() << "\n";
        }
    }
    return 0;
}
