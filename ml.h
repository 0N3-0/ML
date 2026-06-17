#ifndef ML_H_
#define ML_H_

#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]
#define MAT_PRINT(m) ml_mat_print(m, #m)
#define ML_VERIFY_BUF_SIZE 512

#ifndef ML_LOG_LEVEL
#define ML_LOG_LEVEL 1
#endif

#if ML_LOG_LEVEL >= 1
#define LOG_ERROR(...) log_msg(ML_LOG_ERROR, __FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define LOG_ERROR(...) ((void)0)
#endif

#if ML_LOG_LEVEL >= 2
#define LOG_WARN(...) log_msg(ML_LOG_WARN, __FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define LOG_WARN(...) ((void)0)
#endif

#if ML_LOG_LEVEL >= 3
#define LOG_INFO(...) log_msg(ML_LOG_INFO, __FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define LOG_INFO(...) ((void)0)
#endif

#if ML_LOG_LEVEL >= 4
#define LOG_DEBUG(...) log_msg(ML_LOG_DEBUG, __FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define LOG_DEBUG(...) ((void)0)
#endif

#include <math.h>
#include <stddef.h>

typedef float (*Activate)(float);

typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float *es;
} Mat;

typedef float (*Lossf)(Mat target, Mat output);
typedef void (*dLossf)(Mat dst, Mat target, Mat output);

typedef struct {
  Lossf lossf;
  dLossf dlossf;
} Loss;

typedef struct {
  Activate f;
  Activate df;
} Act;

typedef struct {
  float rate;
  size_t batch_size;
  Loss loss;
} TrainConfig;

typedef struct {
  Mat w;
  Mat b;
  Mat z;
  Mat a;
  Act act;
} Layer;

typedef struct {
  Mat w;
  Mat b;
  Mat d;
} GradLayer;

typedef struct {
  size_t layer_count;
  Layer *layer;
} Mod;

typedef struct {
  size_t layer_count;
  GradLayer *layer;
} Grad;

typedef enum {
  ML_LOG_INFO,
  ML_LOG_WARN,
  ML_LOG_ERROR,
  ML_LOG_DEBUG,
} LogLevel;

// clang-format off

void log_msg(
  LogLevel level,
  const char *file,
  int line,
  const char *func_name,
  const char *fmt,
  ...
);                                                                             // formatted log output to stderr, called via LOG_* macros

float rand_float(void);                                                        // random float in [0, 1]
float sigmoidf(float x);                                                       // sigmoid activation
float dsigmoidf(float x);                                                      // derivative of sigmoid
float ReLUf(float x);                                                          // ReLU activation
float dReLUf(float x);                                                         // derivative of ReLU
float LReLUf(float x);                                                         // leaky ReLU activation (alpha=0.01)
float dLReLUf(float x);                                                        // derivative of leaky ReLU
float dtanhf(float x);                                                         // derivative of tanh

Mat ml_mat_alloc(size_t rows, size_t cols, size_t stride);                     // allocate matrix, returns {0} on failure
void ml_mat_free(Mat *m);                                                      // free matrix
void ml_mat_print(Mat m, const char *name);                                    // print matrix to stdout
void ml_mat_rand(Mat m, float low, float high);                                // fill matrix with uniform random values in [low, high]
void ml_mat_sum(Mat dst, Mat src);                                             // add src to dst element-wise
void ml_mat_mul(Mat dst, Mat a, Mat b);                                        // matrix multiply dst = a * b
void ml_mat_mul_elem(Mat dst, Mat a, Mat b);                                   // element-wise multiply dst = a * b
void ml_mat_apply(Mat dst, Mat src, float (*f)(float));                        // apply function f to each element of src, store in dst
void ml_mat_tran(Mat dst, Mat b);                                              // transpose src into dst
void ml_mat_shuffle(Mat m);                                                    // Fisher-Yates shuffle on matrix rows
void ml_mat_scale(Mat m, float scale);                                         // multiply all elements by scale
void ml_mat_zero(Mat m);                                                       // set all elements to zero

Layer ml_modellayer_alloc(
  size_t cur_layer_size,
  size_t pre_layer_size,
  size_t batch_size,
  Act act
);                                                                             // allocate a network layer, returns {0} on failure
void ml_modellayer_free(Layer *layer);                                         // free a network layer

GradLayer ml_gradlayer_alloc(Mod m, size_t l);                                 // allocate gradient buffers for layer l, returns {0} on failure
void ml_gradlayer_free(GradLayer *layer);                                      // free gradient layer

Grad ml_grad_alloc(Mod m);                                                     // allocate gradient buffers for model, returns {0} on failure
void ml_grad_free(Grad *g);                                                    // free all gradient buffers
void ml_grad_scale(Grad g, float scale);                                       // multiply all gradients by scale
void ml_grad_zero(Grad g);                                                     // set all gradient elements to zero

Mod ml_model_alloc(
  size_t *layer_sizes,
  size_t layer_count,
  TrainConfig train_config,
  Act *acts
);                                                                             // allocate model, returns {0} on failure
void ml_model_free(Mod *m);                                                    // free model
int ml_model_xavier_rand(Mod m);                                               // Xavier initialize weights, returns non-zero on failure
void ml_model_rand(Mod m, float low, float high);                              // uniform random initialize weights and biases
void ml_model_forward(Mod m, Mat in);                                          // forward pass through model
int ml_model_print(Mod m);                                                     // print model weights, biases and activations to stdout

float ml_model_cost(Mod m, Mat batch, TrainConfig train_config);               // compute batch cost, returns NAN on failure
int ml_model_finite_diff(
  Mod m,
  Grad g,
  Mat batch,
  TrainConfig train_config,
  float eps
);                                                                             // estimate parameter gradients via finite difference, returns non-zero on failure
int ml_model_backprop(Mod m, Grad g, Mat batch, TrainConfig train_config);     // compute parameter gradients via backpropagation, returns non-zero on failure
int ml_model_optimizer(Mod m, Grad g, TrainConfig train_config);               // update model parameters using SGD, returns non-zero on failure
float ml_model_loss_sse(Mat target, Mat output);                               // sum of squared errors
void ml_model_loss_dsse(Mat dst, Mat target, Mat output);                      // derivative of SSE loss
float ml_model_loss_mse(Mat target, Mat output);                               // mean squared error
void ml_model_loss_dmse(Mat dst, Mat target, Mat output);                      // derivative of MSE loss

int ml_model_verify(Mod m, Mat td);                                            // evaluate model accuracy on labeled data

Mod ml_model_load(const char *file_path, TrainConfig train_config, Act *acts); // load model from binary file, returns {0} on failure
int ml_model_save(Mod m, const char *file_path);                               // save model to binary file, returns non-zero on failure

Mat ml_load_td(const char *file_path);                                         // load training data from CSV file, returns {0} on failure
void ml_model_copy_params(Mod dst, Mod src);                                   // copy weights and biases from src model to dst model

// clang-format on

#endif // !ML_H_

#define ML_IMPLEMENTATION
#ifdef ML_IMPLEMENTATION

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#if ML_LOG_LEVEL >= 1
void log_msg(
  LogLevel level,
  const char *file,
  int line,
  const char *func_name,
  const char *fmt,
  ...
) {
  const char *level_name = NULL;
  switch (level) {
    case ML_LOG_INFO:
      level_name = "INFO";
      break;
    case ML_LOG_WARN:
      level_name = "WARN";
      break;
    case ML_LOG_DEBUG:
      level_name = "DEBUG";
      break;
    case ML_LOG_ERROR:
      level_name = "ERROR";
      break;
    default:
      level_name = "INVALIDLEVEL";
  }

  fprintf(stderr, "[%s] %s:%d:%s ", level_name, file, line, func_name);

  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);

  fprintf(stderr, "\n");
}
#endif

Act ML_SIGMOID = {.f = sigmoidf, .df = dsigmoidf};
Act ML_RELU = {.f = ReLUf, .df = dReLUf};
Act ML_LRELU = {.f = LReLUf, .df = dLReLUf};
Act ML_TANH = {.f = tanhf, .df = dtanhf};

float rand_float(void) {
  return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x) {
  return 1.f / (expf(-x) + 1.f);
}

float dsigmoidf(float x) {
  return sigmoidf(x) * (1 - sigmoidf(x));
}

float ReLUf(float x) {
  if (x <= 0)
    return 0.f;
  return x;
}

float dReLUf(float x) {
  if (x <= 0)
    return 0.f;
  return 1.f;
}

float LReLUf(float x) {
  if (x <= 0)
    return x * 0.01f;
  return x;
}

float dLReLUf(float x) {
  if (x <= 0)
    return 0.01f;
  return 1.f;
}

float dtanhf(float x) {
  return 1 - (tanhf(x) * tanhf(x));
}

Mat ml_mat_alloc(size_t rows, size_t cols, size_t stride) {
  if (rows == 0 || cols == 0 || stride == 0 || stride < cols) {
    LOG_ERROR("invalid matrix param: rows=%zu, cols=%zu, stride=%zu", rows, cols, stride);
    return (Mat){0};
  }

  Mat m = {
    .rows = rows,
    .cols = cols,
    .stride = stride,
    .es = (float *)malloc(rows * stride * sizeof(*m.es)),
  };

  if (m.es == NULL) {
    LOG_ERROR("failed to allocate matrix: rows=%zu, cols=%zu, stride=%zu", rows, cols, stride);
    return (Mat){0};
  }
  return m;
}

Mat ml_mat_slice(Mat m, size_t row, size_t col, size_t rows, size_t cols) {
  if (m.es == NULL) {
    LOG_ERROR("data is NULL");
    return (Mat){0};
  }

  if (rows == 0 || cols == 0) {
    LOG_ERROR("invalid slice param: rows=%zu, cols=%zu", rows, cols);
    exit(1);
  }

  if (row >= m.rows || col >= m.cols || rows > m.rows - row || cols > m.cols - col) {
    LOG_ERROR(
      "slice out of bounds: row=%zu, col=%zu, rows=%zu, cols=%zu, "
      "matrix=%zux%zu",
      row,
      col,
      rows,
      cols,
      m.rows,
      m.cols
    );
    exit(1);
  }

  Mat v = {
    .rows = rows,
    .cols = cols,
    .stride = m.stride,
    .es = &MAT_AT(m, row, col),
  };

  return v;
}

void ml_mat_free(Mat *m) {
  if (m == NULL)
    return;

  free(m->es);
  m->es = NULL;
  m->rows = 0;
  m->cols = 0;
  m->stride = 0;
}

void ml_mat_print(Mat m, const char *name) {
  printf("%s = [\n", name);
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      printf(" %10.6f", MAT_AT(m, i, j));
    }
    printf("\n");
  }
  printf("]\n");
}

void ml_mat_rand(Mat m, float low, float high) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = rand_float() * (high - low) + low;
    }
  }
}

void ml_mat_sum(Mat dst, Mat src) {
  for (size_t i = 0; i < src.rows; ++i) {
    for (size_t j = 0; j < src.cols; ++j) {
      MAT_AT(dst, i, j) += MAT_AT(src, i, j);
    }
  }
}

void ml_mat_mul(Mat dst, Mat a, Mat b) {
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = 0.f;
      for (size_t n = 0; n < a.cols; ++n) {
        MAT_AT(dst, i, j) += MAT_AT(a, i, n) * MAT_AT(b, n, j);
      }
    }
  }
}

void ml_mat_tran(Mat dst, Mat b) {
  for (size_t i = 0; i < b.rows; ++i) {
    for (size_t j = 0; j < b.cols; ++j) {
      MAT_AT(dst, j, i) = MAT_AT(b, i, j);
    }
  }
}

void ml_mat_mul_elem(Mat dst, Mat a, Mat b) {
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = MAT_AT(a, i, j) * MAT_AT(b, i, j);
    }
  }
}

void ml_mat_apply(Mat dst, Mat src, float (*f)(float)) {
  for (size_t i = 0; i < src.rows; ++i) {
    for (size_t j = 0; j < src.cols; ++j) {
      MAT_AT(dst, i, j) = f(MAT_AT(src, i, j));
    }
  }
}

void ml_mat_shuffle(Mat m) {
  for (size_t i = 0; i < m.rows; ++i) {
    size_t change = i + rand() % (m.rows - i);

    for (size_t j = 0; j < m.cols; ++j) {
      float tmp = MAT_AT(m, i, j);
      MAT_AT(m, i, j) = MAT_AT(m, change, j);
      MAT_AT(m, change, j) = tmp;
    }
  }
}

void ml_mat_scale(Mat m, float scale) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) *= scale;
    }
  }
}

void ml_mat_zero(Mat m) {
  if (m.stride == m.cols) {
    memset(m.es, 0, sizeof(*m.es) * m.rows * m.cols);
  } else {
    for (size_t i = 0; i < m.rows; ++i) {
      for (size_t j = 0; j < m.cols; ++j) {
        MAT_AT(m, i, j) = 0;
      }
    }
  }
}

Layer ml_modellayer_alloc(
  size_t cur_layer_size,
  size_t pre_layer_size,
  size_t batch_size,
  Act act
) {
  Layer layer = {0};
  if (act.f == NULL || act.df == NULL) {
    LOG_ERROR("invalid activation function");
    return (Layer){0};
  }

  layer.z = ml_mat_alloc(batch_size, cur_layer_size, cur_layer_size);
  if (layer.z.es == NULL) {
    LOG_ERROR("failed to allocate z");
    return (Layer){0};
  }

  layer.w = ml_mat_alloc(pre_layer_size, cur_layer_size, cur_layer_size);
  if (layer.w.es == NULL) {
    LOG_ERROR("failed to allocate w");
    ml_mat_free(&layer.z);
    return (Layer){0};
  }

  layer.a = ml_mat_alloc(batch_size, cur_layer_size, cur_layer_size);
  if (layer.a.es == NULL) {
    LOG_ERROR("failed to allocate a");
    ml_mat_free(&layer.z);
    ml_mat_free(&layer.w);
    return (Layer){0};
  }

  layer.b = ml_mat_alloc(1, cur_layer_size, cur_layer_size);
  if (layer.b.es == NULL) {
    LOG_ERROR("failed to allocate b");
    ml_mat_free(&layer.z);
    ml_mat_free(&layer.w);
    ml_mat_free(&layer.a);
    return (Layer){0};
  }

  layer.act = act;

  return layer;
}

void ml_modellayer_free(Layer *layer) {

  ml_mat_free(&layer->z);
  ml_mat_free(&layer->w);
  ml_mat_free(&layer->a);
  ml_mat_free(&layer->b);

  layer->act.f = NULL;
  layer->act.df = NULL;
}

Mod ml_model_alloc(size_t *layer_sizes, size_t layer_count, TrainConfig train_config, Act *acts) {
  Mod m = {0};

  if (layer_sizes == NULL) {
    LOG_ERROR("invalid model params: layer_sizes is NULL");
    return m;
  }

  if (acts == NULL) {
    LOG_ERROR("invalid model params: acts is NULL");
    return m;
  }

  if (layer_count <= 1) {
    LOG_ERROR("invalid model params: layer_count must be greater than 1");
    return m;
  }

  if (train_config.batch_size == 0) {
    LOG_ERROR("invalid train config: batch_size is zero");
    return m;
  }

  m.layer_count = layer_count - 1;

  m.layer = (Layer *)malloc(m.layer_count * sizeof(*m.layer));
  if (m.layer == NULL) {
    LOG_ERROR("failed to allocate model layers");
    return (Mod){0};
  }

  for (size_t i = 0; i < m.layer_count; ++i) {
    m.layer[i] =
      ml_modellayer_alloc(layer_sizes[i + 1], layer_sizes[i], train_config.batch_size, acts[i]);
    if (m.layer[i].z.es == NULL) {
      LOG_ERROR("failed to allocate layer %zu", i);
      for (size_t j = 0; j < i; ++j) {
        ml_modellayer_free(&m.layer[j]);
      }
      free(m.layer);
      m.layer = NULL;
      m.layer_count = 0;
      return (Mod){0};
    }
  }

  return m;
}

void ml_model_free(Mod *m) {
  if (m == NULL)
    return;

  for (size_t i = 0; i < m->layer_count; ++i) {
    ml_modellayer_free(&m->layer[i]);
  }

  free(m->layer);
  m->layer = NULL;
  m->layer_count = 0;
}

int ml_model_xavier_rand(Mod m) {
  if (m.layer_count == 0) {
    LOG_ERROR("invalid model: layer_count=%zu", m.layer_count);
    return 1;
  }

  if (m.layer == NULL) {
    LOG_ERROR("invalid model: layer is NULL");
    return 1;
  }

  for (size_t i = 0; i < m.layer_count; ++i) {
    if (m.layer[i].w.es == NULL) {
      LOG_ERROR("invalid model layer %zu: weight data is NULL", i);
      return 1;
    }

    size_t rows = m.layer[i].w.rows;
    size_t cols = m.layer[i].w.cols;

    if (rows == 0 || cols == 0) {
      LOG_ERROR("invalid model layer %zu: weight matrix=%zux%zu", i, rows, cols);
      return 1;
    }

    float limit = sqrtf(6.f / (float)(rows + cols));

    ml_mat_rand(m.layer[i].w, -limit, limit);
  }

  return 0;
}

void ml_model_rand(Mod m, float low, float high) {
  for (size_t i = 0; i < m.layer_count; ++i) {
    ml_mat_rand(m.layer[i].w, low, high);
    ml_mat_rand(m.layer[i].b, low, high);
  }
}

int ml_model_print(Mod m) {
  if (m.layer_count == 0) {
    LOG_ERROR("invalid model: layer_count=%zu", m.layer_count);
    return 1;
  }

  for (size_t i = 0; i < m.layer_count; ++i) {
    printf("layer %zu:\n", i);
    ml_mat_print(m.layer[i].w, "w");
    ml_mat_print(m.layer[i].b, "b");
    ml_mat_print(m.layer[i].a, "a");
    printf("\n");
  }
  return 0;
}

void ml_model_forward(Mod m, Mat in) {
  for (size_t i = 0; i < m.layer_count; ++i) {
    if (!i)
      ml_mat_mul(m.layer[i].z, in, m.layer[i].w);
    else
      ml_mat_mul(m.layer[i].z, m.layer[i - 1].a, m.layer[i].w);
    for (size_t j = 0; j < m.layer[i].z.rows; ++j) {
      Mat t = ml_mat_slice(m.layer[i].z, j, 0, 1, m.layer[i].z.cols);
      ml_mat_sum(t, m.layer[i].b);
    }
    ml_mat_apply(m.layer[i].a, m.layer[i].z, m.layer[i].act.f);
  }
}

float ml_model_loss_sse(Mat target, Mat output) {
  if (output.rows == 0 || output.cols == 0) {
    LOG_ERROR("invalid output matrix: output = %zux%zu", output.rows, output.cols);
    return NAN;
  }

  if (target.rows != output.rows || target.cols < output.cols) {
    LOG_ERROR(
      "invalid loss input: target = %zux%zu, output = %zux%zu",
      target.rows,
      target.cols,
      output.rows,
      output.cols
    );
    return NAN;
  }

  size_t output_size = output.cols;
  size_t input_size = target.cols - output_size;

  float res = 0.f;

  for (size_t i = 0; i < output.rows; ++i) {
    for (size_t j = 0; j < output_size; ++j) {
      float d = MAT_AT(target, i, input_size + j) - MAT_AT(output, i, j);
      res += d * d;
    }
  }

  return res;
}

void ml_model_loss_dsse(Mat dst, Mat target, Mat output) {
  size_t output_size = output.cols;
  size_t input_size = target.cols - output_size;

  for (size_t i = 0; i < output.rows; ++i) {
    for (size_t j = 0; j < output_size; ++j) {
      MAT_AT(dst, i, j) = 2.f * (MAT_AT(output, i, j) - MAT_AT(target, i, input_size + j));
    }
  }
}

float ml_model_loss_mse(Mat target, Mat output) {
  if (output.rows == 0 || output.cols == 0) {
    LOG_ERROR("invalid output matrix: output = %zux%zu", output.rows, output.cols);
    return NAN;
  }

  if (target.rows != output.rows || target.cols < output.cols) {
    LOG_ERROR(
      "invalid loss input: target = %zux%zu, output = %zux%zu",
      target.rows,
      target.cols,
      output.rows,
      output.cols
    );
    return NAN;
  }

  size_t output_size = output.cols;
  size_t input_size = target.cols - output_size;

  float res = 0.f;

  for (size_t i = 0; i < output.rows; ++i) {
    for (size_t j = 0; j < output_size; ++j) {
      float d = MAT_AT(target, i, input_size + j) - MAT_AT(output, i, j);
      res += d * d;
    }
  }

  return res / (float)(output.rows * output.cols);
}

void ml_model_loss_dmse(Mat dst, Mat target, Mat output) {
  size_t output_size = output.cols;
  size_t input_size = target.cols - output_size;
  float scale = 1.f / (float)(output.rows * output.cols);

  for (size_t i = 0; i < output.rows; ++i) {
    for (size_t j = 0; j < output_size; ++j) {
      MAT_AT(dst, i, j) = 2.f * (MAT_AT(output, i, j) - MAT_AT(target, i, input_size + j)) * scale;
    }
  }
}

float ml_model_cost(Mod m, Mat batch, TrainConfig train_config) {
  if (m.layer_count == 0 || m.layer == NULL) {
    LOG_ERROR("invalid model: layer_count=%zu", m.layer_count);
    return NAN;
  }

  if (train_config.loss.lossf == NULL) {
    LOG_ERROR("invalid train config: loss function is NULL");
    return NAN;
  }

  size_t input_size = m.layer[0].w.rows;
  size_t output_size = m.layer[m.layer_count - 1].a.cols;

  if (input_size + output_size != batch.cols) {
    LOG_ERROR(
      "invalid batch matrix: batch = %zux%zu, expected cols = %zu",
      batch.rows,
      batch.cols,
      input_size + output_size
    );
    return NAN;
  }

  Mat in = ml_mat_slice(batch, 0, 0, batch.rows, input_size);

  ml_model_forward(m, in);

  Mat out = m.layer[m.layer_count - 1].a;

  return train_config.loss.lossf(batch, out);
}

int ml_model_finite_diff(Mod m, Grad g, Mat batch, TrainConfig train_config, float eps) {
  float c = ml_model_cost(m, batch, train_config);
  if (c == NAN) {
    LOG_ERROR("failed to calculate cost");
    return 1;
  }

  for (size_t i = 0; i < m.layer_count; ++i) {
    for (size_t j = 0; j < m.layer[i].w.rows; ++j) {
      for (size_t k = 0; k < m.layer[i].w.cols; ++k) {
        float save = MAT_AT(m.layer[i].w, j, k);
        MAT_AT(m.layer[i].w, j, k) += eps;
        MAT_AT(g.layer[i].w, j, k) = (ml_model_cost(m, batch, train_config) - c) / eps;
        MAT_AT(m.layer[i].w, j, k) = save;
      }
    }
    for (size_t j = 0; j < m.layer[i].b.rows; ++j) {
      for (size_t k = 0; k < m.layer[i].b.cols; ++k) {
        float save = MAT_AT(m.layer[i].b, j, k);
        MAT_AT(m.layer[i].b, j, k) += eps;
        MAT_AT(g.layer[i].b, j, k) = (ml_model_cost(m, batch, train_config) - c) / eps;
        MAT_AT(m.layer[i].b, j, k) = save;
      }
    }
  }
  return 0;
}

int ml_model_backprop(Mod m, Grad g, Mat batch, TrainConfig train_config) {
  if (m.layer_count == 0) {
    LOG_ERROR("invalid model: layer_count=%zu", m.layer_count);
    return 1;
  }

  size_t input_size = m.layer[0].w.rows;
  size_t output_size = m.layer[m.layer_count - 1].a.cols;

  if (batch.cols != input_size + output_size) {
    LOG_ERROR("input/output size mismatch: input=%zu, output=%zu", input_size, output_size);
    return 1;
  }

  ml_grad_zero(g);

  Mat in = {0};
  in.rows = batch.rows;
  in.cols = input_size;
  in.stride = batch.stride;
  in.es = &MAT_AT(batch, 0, 0);

  Mat target = {0};
  target.rows = batch.rows;
  target.cols = output_size;
  target.stride = batch.stride;
  target.es = &MAT_AT(batch, 0, input_size);

  ml_model_forward(m, in);

  size_t last = m.layer_count - 1;

  train_config.loss.dlossf(g.layer[last].d, target, m.layer[last].a);

  for (size_t i = 0; i < g.layer[last].d.rows; ++i) {
    for (size_t j = 0; j < g.layer[last].d.cols; ++j) {
      MAT_AT(g.layer[last].d, i, j) *= m.layer[last].act.df(MAT_AT(m.layer[last].z, i, j));
    }
  }

  for (size_t l = m.layer_count - 1; l-- > 0;) {
    for (size_t i = 0; i < g.layer[l].d.rows; ++i) {
      for (size_t j = 0; j < g.layer[l].d.cols; ++j) {
        float sum = 0.0f;

        for (size_t k = 0; k < g.layer[l + 1].d.cols; ++k) {
          sum += MAT_AT(g.layer[l + 1].d, i, k) * MAT_AT(m.layer[l + 1].w, j, k);
        }

        MAT_AT(g.layer[l].d, i, j) = sum * m.layer[l].act.df(MAT_AT(m.layer[l].z, i, j));
      }
    }
  }

  for (size_t l = 0; l < m.layer_count; ++l) {
    Mat prev = l == 0 ? in : m.layer[l - 1].a;

    for (size_t i = 0; i < g.layer[l].w.rows; ++i) {
      for (size_t j = 0; j < g.layer[l].w.cols; ++j) {
        float sum = 0.0f;

        for (size_t k = 0; k < batch.rows; ++k) {
          sum += MAT_AT(prev, k, i) * MAT_AT(g.layer[l].d, k, j);
        }

        MAT_AT(g.layer[l].w, i, j) = sum;
      }
    }

    for (size_t j = 0; j < g.layer[l].b.cols; ++j) {
      float sum = 0.0f;

      for (size_t i = 0; i < batch.rows; ++i) {
        sum += MAT_AT(g.layer[l].d, i, j);
      }

      MAT_AT(g.layer[l].b, 0, j) = sum;
    }
  }

  return 0;
}
int ml_model_optimizer(Mod m, Grad g, TrainConfig train_config) {
  if (train_config.rate <= 0 || m.layer_count != g.layer_count) {
    LOG_ERROR("invalid rate or layer count mismatch");
    return 1;
  }
  for (size_t i = 0; i < m.layer_count; ++i) {
    if (m.layer[i].w.rows != g.layer[i].w.rows || m.layer[i].w.cols != g.layer[i].w.cols) {
      LOG_ERROR(
        "weight gradient shape mismatch at layer %zu: "
        "m.layer[%zu].w is %zux%zu, but g.w[%zu] is %zux%zu",
        i,
        i,
        m.layer[i].w.rows,
        m.layer[i].w.cols,
        i,
        g.layer[i].w.rows,
        g.layer[i].w.cols
      );
      return 1;
    }
    for (size_t j = 0; j < m.layer[i].w.rows; ++j) {
      for (size_t k = 0; k < m.layer[i].w.cols; ++k) {
        MAT_AT(m.layer[i].w, j, k) -= MAT_AT(g.layer[i].w, j, k) * train_config.rate;
      }
    }
    if (m.layer[i].b.rows != g.layer[i].b.rows || m.layer[i].b.cols != g.layer[i].b.cols) {
      LOG_ERROR(
        "bias gradient shape mismatch at layer %zu: "
        "m.layer[%zu].b is %zux%zu, but g.b[%zu] is %zux%zu",
        i,
        i,
        m.layer[i].b.rows,
        m.layer[i].b.cols,
        i,
        g.layer[i].b.rows,
        g.layer[i].b.cols
      );
      return 1;
    }
    for (size_t j = 0; j < m.layer[i].b.rows; ++j) {
      for (size_t k = 0; k < m.layer[i].b.cols; ++k) {
        MAT_AT(m.layer[i].b, j, k) -= MAT_AT(g.layer[i].b, j, k) * train_config.rate;
      }
    }
  }
  return 0;
}

static int ml_binary(float x) {
  return x >= 0.5f ? 1 : 0;
}

static void
ml_format_vec_float(char *buf, size_t buf_size, Mat m, size_t row, size_t offset, size_t cols) {
  size_t n = 0;

  n += snprintf(buf + n, buf_size - n, "[");

  for (size_t j = 0; j < cols && n < buf_size; ++j) {
    n += snprintf(buf + n, buf_size - n, "%s%.3f", j ? " " : "", MAT_AT(m, row, offset + j));
  }

  snprintf(buf + n, buf_size - n, "]");
}

static void
ml_format_vec_binary(char *buf, size_t buf_size, Mat m, size_t row, size_t offset, size_t cols) {
  size_t n = 0;

  n += snprintf(buf + n, buf_size - n, "[");

  for (size_t j = 0; j < cols && n < buf_size; ++j) {
    n +=
      snprintf(buf + n, buf_size - n, "%s%d", j ? " " : "", ml_binary(MAT_AT(m, row, offset + j)));
  }

  snprintf(buf + n, buf_size - n, "]");
}

int ml_model_verify(Mod m, Mat td) {
  if (m.layer_count == 0) {
    LOG_ERROR("invalid model: layer_count=%zu", m.layer_count);
    return 1;
  }

  if (td.rows == 0 || td.es == NULL) {
    LOG_ERROR("invalid training data");
    return 1;
  }

  size_t input_size = m.layer[0].w.rows;
  size_t output_size = m.layer[m.layer_count - 1].a.cols;
  size_t batch_size = m.layer[m.layer_count - 1].a.rows;

  if (td.cols != input_size + output_size) {
    LOG_ERROR("invalid td shape: td.cols=%zu, expected=%zu", td.cols, input_size + output_size);
    return 1;
  }

  if (batch_size == 0) {
    LOG_ERROR("batch_size is zero");
    return 1;
  }

  if (td.rows % batch_size != 0) {
    LOG_ERROR(
      "td.rows must be divisible by batch_size: "
      "td.rows=%zu, batch_size=%zu",
      td.rows,
      batch_size
    );
    return 1;
  }

  size_t correct_outputs = 0;
  size_t total_outputs = td.rows * output_size;

  size_t correct_samples = 0;
  size_t wrong_samples = 0;

  for (size_t begin = 0; begin < td.rows; begin += batch_size) {
    Mat batch = ml_mat_slice(td, begin, 0, batch_size, td.cols);
    Mat in = ml_mat_slice(batch, 0, 0, batch.rows, input_size);
    Mat target = ml_mat_slice(batch, 0, input_size, batch.rows, output_size);

    ml_model_forward(m, in);

    Mat out = m.layer[m.layer_count - 1].a;

    for (size_t i = 0; i < batch.rows; ++i) {
      int sample_correct = 1;

      for (size_t j = 0; j < output_size; ++j) {
        int pred = ml_binary(MAT_AT(out, i, j));
        int expect = ml_binary(MAT_AT(target, i, j));

        if (pred == expect) {
          correct_outputs++;
        } else {
          sample_correct = 0;
        }
      }

      if (sample_correct) {
        correct_samples++;
        continue;
      }

      if (wrong_samples == 0) {
        LOG_WARN("wrong predictions:");
        LOG_WARN("%-5s %-37s %-25s %-13s %-13s", "idx", "input", "output", "pred", "expected");
      }

      wrong_samples++;

      char input_buf[ML_VERIFY_BUF_SIZE];
      char output_buf[ML_VERIFY_BUF_SIZE];
      char pred_buf[ML_VERIFY_BUF_SIZE];
      char expected_buf[ML_VERIFY_BUF_SIZE];

      ml_format_vec_float(input_buf, sizeof(input_buf), in, i, 0, input_size);

      ml_format_vec_float(output_buf, sizeof(output_buf), out, i, 0, output_size);

      ml_format_vec_binary(pred_buf, sizeof(pred_buf), out, i, 0, output_size);

      ml_format_vec_binary(expected_buf, sizeof(expected_buf), target, i, 0, output_size);

      LOG_WARN(
        "%-5zu %-37s %-25s %-13s %-13s",
        begin + i,
        input_buf,
        output_buf,
        pred_buf,
        expected_buf
      );
    }
  }

  if (wrong_samples == 0) {
    LOG_INFO("verify: all samples correct.");
  } else {
    LOG_WARN("wrong samples: %zu/%zu", wrong_samples, td.rows);
  }

  LOG_INFO(
    "sample accuracy: %zu/%zu = %.2f%%",
    correct_samples,
    td.rows,
    100.f * (float)correct_samples / (float)td.rows
  );

  LOG_INFO(
    "output accuracy: %zu/%zu = %.2f%%",
    correct_outputs,
    total_outputs,
    100.f * (float)correct_outputs / (float)total_outputs
  );
  (void)correct_samples;
  (void)correct_outputs;
  (void)total_outputs;
  return 0;
}

int ml_model_save(Mod m, const char *file_path) {
  FILE *fd = fopen(file_path, "wb");
  if (fd == NULL) {
    LOG_ERROR("failed to open file: %s", file_path);
    return 1;
  }

  if (m.layer_count == 0) {
    LOG_ERROR("invalid model: layer_count=%zu", m.layer_count);
    fclose(fd);
    return 1;
  }

  fwrite("MLMB", 1, 4, fd);

  uint64_t layer_count = (uint64_t)m.layer_count;
  fwrite(&layer_count, sizeof(layer_count), 1, fd);

  for (size_t i = 0; i < m.layer_count; ++i) {
    uint64_t w_rows = (uint64_t)m.layer[i].w.rows;
    uint64_t w_cols = (uint64_t)m.layer[i].w.cols;

    fwrite("W", 1, 1, fd);
    fwrite(&w_rows, sizeof(w_rows), 1, fd);
    fwrite(&w_cols, sizeof(w_cols), 1, fd);

    for (size_t r = 0; r < m.layer[i].w.rows; ++r) {
      for (size_t c = 0; c < m.layer[i].w.cols; ++c) {
        float x = MAT_AT(m.layer[i].w, r, c);
        fwrite(&x, sizeof(x), 1, fd);
      }
    }

    uint64_t b_rows = (uint64_t)m.layer[i].b.rows;
    uint64_t b_cols = (uint64_t)m.layer[i].b.cols;

    fwrite("B", 1, 1, fd);
    fwrite(&b_rows, sizeof(b_rows), 1, fd);
    fwrite(&b_cols, sizeof(b_cols), 1, fd);

    for (size_t r = 0; r < m.layer[i].b.rows; ++r) {
      for (size_t c = 0; c < m.layer[i].b.cols; ++c) {
        float x = MAT_AT(m.layer[i].b, r, c);
        fwrite(&x, sizeof(x), 1, fd);
      }
    }
  }

  fclose(fd);
  LOG_INFO("model saved to %s", file_path);
  return 0;
}

Mod ml_model_load(const char *file_path, TrainConfig train_config, Act *acts) {
  FILE *fd = fopen(file_path, "rb");
  if (fd == NULL) {
    LOG_ERROR("failed to open file: %s", file_path);
    return (Mod){0};
  }

  if (train_config.batch_size == 0) {
    LOG_ERROR("invalid batch_size = 0");
    fclose(fd);
    return (Mod){0};
  }

  if (acts == NULL) {
    LOG_ERROR("acts is NULL");
    fclose(fd);
    return (Mod){0};
  }

  char magic[4];
  size_t nread = fread(magic, 1, 4, fd);
  if (nread != 4 || memcmp(magic, "MLMB", 4) != 0) {
    LOG_ERROR("invalid model file magic");
    fclose(fd);
    return (Mod){0};
  }

  uint64_t file_layer_count = 0;
  nread = fread(&file_layer_count, sizeof(file_layer_count), 1, fd);
  if (nread != 1 || file_layer_count == 0) {
    LOG_ERROR("invalid layer_count=%zu", (size_t)file_layer_count);
    fclose(fd);
    return (Mod){0};
  }

  size_t layer_count = (size_t)file_layer_count;

  size_t *layer_sizes = (size_t *)malloc((layer_count + 1) * sizeof(*layer_sizes));
  if (layer_sizes == NULL) {
    LOG_ERROR("failed to allocate layer_sizes");
    fclose(fd);
    return (Mod){0};
  }

  for (size_t i = 0; i < layer_count; ++i) {
    char type = 0;
    uint64_t rows = 0;
    uint64_t cols = 0;

    nread = fread(&type, 1, 1, fd);
    if (nread != 1 || type != 'W') {
      LOG_ERROR("expected W at layer %zu", i);
      free(layer_sizes);
      fclose(fd);
      return (Mod){0};
    }

    nread = fread(&rows, sizeof(rows), 1, fd);
    if (nread != 1) {
      LOG_ERROR("failed to read W rows at layer %zu", i);
      free(layer_sizes);
      fclose(fd);
      return (Mod){0};
    }

    nread = fread(&cols, sizeof(cols), 1, fd);
    if (nread != 1) {
      LOG_ERROR("failed to read W cols at layer %zu", i);
      free(layer_sizes);
      fclose(fd);
      return (Mod){0};
    }

    if (rows == 0 || cols == 0) {
      LOG_ERROR("invalid W shape at layer %zu: %zux%zu", i, (size_t)rows, (size_t)cols);
      free(layer_sizes);
      fclose(fd);
      return (Mod){0};
    }

    if (i == 0) {
      layer_sizes[0] = (size_t)rows;
    } else if (layer_sizes[i] != (size_t)rows) {
      LOG_ERROR(
        "W input mismatch at layer %zu: expected %zu, got %zu",
        i,
        layer_sizes[i],
        (size_t)rows
      );
      free(layer_sizes);
      fclose(fd);
      return (Mod){0};
    }

    layer_sizes[i + 1] = (size_t)cols;

    if (fseek(fd, (long)(rows * cols * sizeof(float)), SEEK_CUR) != 0) {
      LOG_ERROR("failed to skip W data at layer %zu", i);
      free(layer_sizes);
      fclose(fd);
      return (Mod){0};
    }

    nread = fread(&type, 1, 1, fd);
    if (nread != 1 || type != 'B') {
      LOG_ERROR("expected B at layer %zu", i);
      free(layer_sizes);
      fclose(fd);
      return (Mod){0};
    }

    nread = fread(&rows, sizeof(rows), 1, fd);
    if (nread != 1) {
      LOG_ERROR("failed to read B rows at layer %zu", i);
      free(layer_sizes);
      fclose(fd);
      return (Mod){0};
    }

    nread = fread(&cols, sizeof(cols), 1, fd);
    if (nread != 1) {
      LOG_ERROR("failed to read B cols at layer %zu", i);
      free(layer_sizes);
      fclose(fd);
      return (Mod){0};
    }

    if (rows != 1 || (size_t)cols != layer_sizes[i + 1]) {
      LOG_ERROR(
        "invalid B shape at layer %zu: "
        "got %zux%zu, expected 1x%zu",
        i,
        (size_t)rows,
        (size_t)cols,
        layer_sizes[i + 1]
      );
      free(layer_sizes);
      fclose(fd);
      return (Mod){0};
    }

    if (fseek(fd, (long)(rows * cols * sizeof(float)), SEEK_CUR) != 0) {
      LOG_ERROR("failed to skip B data at layer %zu", i);
      free(layer_sizes);
      fclose(fd);
      return (Mod){0};
    }
  }

  Mod m = ml_model_alloc(layer_sizes, layer_count + 1, train_config, acts);
  free(layer_sizes);

  if (m.layer == NULL) {
    LOG_ERROR("failed to allocate model");
    fclose(fd);
    return (Mod){0};
  }

  rewind(fd);

  nread = fread(magic, 1, 4, fd);
  if (nread != 4 || memcmp(magic, "MLMB", 4) != 0) {
    LOG_ERROR("invalid magic on second read");
    ml_model_free(&m);
    fclose(fd);
    return (Mod){0};
  }

  file_layer_count = 0;
  nread = fread(&file_layer_count, sizeof(file_layer_count), 1, fd);
  if (nread != 1 || (size_t)file_layer_count != m.layer_count) {
    LOG_ERROR("layer_count mismatch: file=%zu, model=%zu", (size_t)file_layer_count, m.layer_count);
    ml_model_free(&m);
    fclose(fd);
    return (Mod){0};
  }

  /*
    第二遍：真正读取 W/B 数据。
  */
  for (size_t i = 0; i < m.layer_count; ++i) {
    char type = 0;
    uint64_t rows = 0;
    uint64_t cols = 0;

    Layer *ly = &m.layer[i];

    nread = fread(&type, 1, 1, fd);
    if (nread != 1 || type != 'W') {
      LOG_ERROR("expected W at layer %zu", i);
      ml_model_free(&m);
      fclose(fd);
      return (Mod){0};
    }

    nread = fread(&rows, sizeof(rows), 1, fd);
    if (nread != 1) {
      LOG_ERROR("failed to read W rows at layer %zu", i);
      ml_model_free(&m);
      fclose(fd);
      return (Mod){0};
    }

    nread = fread(&cols, sizeof(cols), 1, fd);
    if (nread != 1) {
      LOG_ERROR("failed to read W cols at layer %zu", i);
      ml_model_free(&m);
      fclose(fd);
      return (Mod){0};
    }

    if ((size_t)rows != ly->w.rows || (size_t)cols != ly->w.cols) {
      LOG_ERROR(
        "W shape mismatch at layer %zu: "
        "file=%zux%zu, model=%zux%zu",
        i,
        (size_t)rows,
        (size_t)cols,
        ly->w.rows,
        ly->w.cols
      );
      ml_model_free(&m);
      fclose(fd);
      return (Mod){0};
    }

    for (size_t r = 0; r < ly->w.rows; ++r) {
      for (size_t c = 0; c < ly->w.cols; ++c) {
        float x = 0.0f;

        nread = fread(&x, sizeof(x), 1, fd);
        if (nread != 1) {
          LOG_ERROR(
            "failed to read W data at layer %zu, "
            "row %zu, col %zu",
            i,
            r,
            c
          );
          ml_model_free(&m);
          fclose(fd);
          return (Mod){0};
        }

        MAT_AT(ly->w, r, c) = x;
      }
    }

    nread = fread(&type, 1, 1, fd);
    if (nread != 1 || type != 'B') {
      LOG_ERROR("expected B at layer %zu", i);
      ml_model_free(&m);
      fclose(fd);
      return (Mod){0};
    }

    nread = fread(&rows, sizeof(rows), 1, fd);
    if (nread != 1) {
      LOG_ERROR("failed to read B rows at layer %zu", i);
      ml_model_free(&m);
      fclose(fd);
      return (Mod){0};
    }

    nread = fread(&cols, sizeof(cols), 1, fd);
    if (nread != 1) {
      LOG_ERROR("failed to read B cols at layer %zu", i);
      ml_model_free(&m);
      fclose(fd);
      return (Mod){0};
    }

    if ((size_t)rows != ly->b.rows || (size_t)cols != ly->b.cols) {
      LOG_ERROR(
        "B shape mismatch at layer %zu: "
        "file=%zux%zu, model=%zux%zu",
        i,
        (size_t)rows,
        (size_t)cols,
        ly->b.rows,
        ly->b.cols
      );
      ml_model_free(&m);
      fclose(fd);
      return (Mod){0};
    }

    for (size_t r = 0; r < ly->b.rows; ++r) {
      for (size_t c = 0; c < ly->b.cols; ++c) {
        float x = 0.0f;

        nread = fread(&x, sizeof(x), 1, fd);
        if (nread != 1) {
          LOG_ERROR(
            "failed to read B data at layer %zu, "
            "row %zu, col %zu",
            i,
            r,
            c
          );
          ml_model_free(&m);
          fclose(fd);
          return (Mod){0};
        }

        MAT_AT(ly->b, r, c) = x;
      }
    }
  }

  fclose(fd);
  return m;
}

GradLayer ml_gradlayer_alloc(Mod m, size_t l) {
  GradLayer layer = {0};

  layer.w = ml_mat_alloc(m.layer[l].w.rows, m.layer[l].w.cols, m.layer[l].w.stride);
  if (layer.w.es == NULL) {
    LOG_ERROR("failed to allocate w at layer %zu", l);
    return (GradLayer){0};
  }

  layer.b = ml_mat_alloc(m.layer[l].b.rows, m.layer[l].b.cols, m.layer[l].b.stride);
  if (layer.b.es == NULL) {
    LOG_ERROR("failed to allocate b at layer %zu", l);
    ml_mat_free(&layer.w);
    return (GradLayer){0};
  }

  layer.d = ml_mat_alloc(m.layer[l].a.rows, m.layer[l].a.cols, m.layer[l].a.stride);
  if (layer.d.es == NULL) {
    LOG_ERROR("failed to allocate d at layer %zu", l);
    ml_mat_free(&layer.w);
    ml_mat_free(&layer.b);
    return (GradLayer){0};
  }

  return layer;
}

void ml_gradlayer_free(GradLayer *layer) {

  ml_mat_free(&layer->w);
  ml_mat_free(&layer->b);
  ml_mat_free(&layer->d);
}

Grad ml_grad_alloc(Mod m) {
  if (m.layer_count == 0) {
    LOG_ERROR("invalid model: layer_count=%zu", m.layer_count);
    return (Grad){0};
  }
  if (m.layer == NULL) {
    LOG_ERROR("invalid model: layer is NULL");
    return (Grad){0};
  }

  Grad g = {0};

  g.layer_count = m.layer_count;
  g.layer = (GradLayer *)malloc(g.layer_count * sizeof(*g.layer));

  if (g.layer == NULL) {
    LOG_ERROR("failed to allocate grad layers");
    exit(1);
  }

  for (size_t i = 0; i < g.layer_count; ++i) {
    g.layer[i] = ml_gradlayer_alloc(m, i);
    if (g.layer[i].w.es == NULL) {
      LOG_ERROR("failed to allocate grad layer %zu", i);
      for (size_t j = 0; j < i; ++j) {
        ml_gradlayer_free(&g.layer[j]);
      }
      free(g.layer);
      g.layer = NULL;
      g.layer_count = 0;
      return (Grad){0};
    }
  }

  return g;
}

void ml_grad_free(Grad *g) {
  if (g == NULL)
    return;

  for (size_t i = 0; i < g->layer_count; ++i) {
    ml_gradlayer_free(&g->layer[i]);
  }

  free(g->layer);
  g->layer = NULL;
  g->layer_count = 0;
}

void ml_grad_scale(Grad g, float scale) {
  if (g.layer_count == 0 || g.layer == NULL) {
    LOG_ERROR("invalid grad: layer_count=%zu, layer=%p", g.layer_count, (void *)g.layer);
    exit(1);
  }

  for (size_t i = 0; i < g.layer_count; ++i) {
    ml_mat_scale(g.layer[i].w, scale);
    ml_mat_scale(g.layer[i].b, scale);
    ml_mat_scale(g.layer[i].d, scale);
  }
}

void ml_grad_zero(Grad g) {
  if (g.layer_count == 0 || g.layer == NULL) {
    LOG_ERROR("invalid grad: layer_count=%zu, layer=%p", g.layer_count, (void *)g.layer);
    exit(1);
  }

  for (size_t i = 0; i < g.layer_count; ++i) {
    ml_mat_zero(g.layer[i].w);
    ml_mat_zero(g.layer[i].b);
    ml_mat_zero(g.layer[i].d);
  }
}

char *read_line_from_file(FILE *fd) {
  size_t len = 0;
  size_t cap = 0x20;

  char *line = (char *)malloc(cap);
  if (line == NULL) {
    LOG_ERROR("failed to allocate line buffer");
    return NULL;
  }

  line[0] = '\0';

  while (fgets(line + len, cap - len, fd) != NULL) {
    len += strlen(line + len);

    if (len > 0 && line[len - 1] == '\n')
      return line;

    if (feof(fd))
      return line;

    cap *= 2;

    char *new_line = (char *)realloc(line, cap);
    if (new_line == NULL) {
      LOG_ERROR("failed to reallocate line buffer");
      free(line);
      return NULL;
    }
    line = new_line;
  }
  free(line);
  return NULL;
}

Mat ml_load_td(const char *file_path) {
  FILE *fd = fopen(file_path, "r");
  if (fd == NULL) {
    LOG_ERROR("failed to open file: %s", file_path);
    return (Mat){0};
  }
  Mat td = {0};

  char *line = NULL;

  line = read_line_from_file(fd);
  if (line == NULL) {
    LOG_ERROR("failed to read header line");
    fclose(fd);
    return (Mat){0};
  }

  size_t cols = 1;
  for (char *p = line; *p != '\0'; ++p) {
    if (*p == ',')
      cols++;
  }

  free(line);
  line = NULL;

  size_t rows = 0;

  while ((line = read_line_from_file(fd)) != NULL) {
    if (line[0] != '\n' && line[0] != '\0')
      rows++;

    free(line);
    line = NULL;
  }

  if (rows == 0) {
    LOG_ERROR("no data rows found");
    fclose(fd);
    return (Mat){0};
  }
  if (cols == 0) {
    LOG_ERROR("no data columns found");
    fclose(fd);
    return (Mat){0};
  }

  td = ml_mat_alloc(rows, cols, cols);
  if (td.es == NULL) {
    LOG_ERROR("failed to allocate matrix");
    fclose(fd);
    return (Mat){0};
  }

  rewind(fd);

  line = read_line_from_file(fd);
  if (line == NULL) {
    LOG_ERROR("failed to re-read header line");
    ml_mat_free(&td);
    fclose(fd);
    return (Mat){0};
  }
  free(line);
  line = NULL;

  size_t r = 0;

  while ((line = read_line_from_file(fd)) != NULL) {
    if (line[0] == '\n' || line[0] == '\0') {
      free(line);
      line = NULL;
      continue;
    }

    size_t c = 0;
    char *token = strtok(line, ",\n\r");

    while (token != NULL) {
      if (c >= cols) {
        LOG_ERROR("too many columns at row %zu", r);
        free(line);
        ml_mat_free(&td);
        fclose(fd);
        return (Mat){0};
      }

      char *end = NULL;
      float value = strtof(token, &end);

      if (end == token) {
        LOG_ERROR("failed to parse float at row %zu, col %zu", r, c);
        free(line);
        ml_mat_free(&td);
        fclose(fd);
        return (Mat){0};
      }

      MAT_AT(td, r, c) = value;

      c++;
      token = strtok(NULL, ",\n\r");
    }

    if (c != cols) {
      LOG_ERROR("wrong number of columns at row %zu: expected %zu, got %zu", r, cols, c);
      free(line);
      ml_mat_free(&td);
      fclose(fd);
      return (Mat){0};
    }
    r++;
    free(line);
    line = NULL;
  }

  if (r != rows) {
    LOG_ERROR("wrong number of rows: expected %zu, got %zu", rows, r);
    ml_mat_free(&td);
    fclose(fd);
    return (Mat){0};
  }

  fclose(fd);
  return td;
}

void ml_model_copy_params(Mod dst, Mod src) {
  if (dst.layer_count != src.layer_count) {
    LOG_ERROR("layer_count mismatch: dst=%zu, src=%zu", dst.layer_count, src.layer_count);
    exit(1);
  }

  for (size_t l = 0; l < src.layer_count; ++l) {
    for (size_t i = 0; i < src.layer[l].w.rows; ++i) {
      for (size_t j = 0; j < src.layer[l].w.cols; ++j) {
        MAT_AT(dst.layer[l].w, i, j) = MAT_AT(src.layer[l].w, i, j);
      }
    }

    for (size_t i = 0; i < src.layer[l].b.rows; ++i) {
      for (size_t j = 0; j < src.layer[l].b.cols; ++j) {
        MAT_AT(dst.layer[l].b, i, j) = MAT_AT(src.layer[l].b, i, j);
      }
    }
  }
}
#endif // ML_IMPLEMENTATION
