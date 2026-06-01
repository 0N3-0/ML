#ifndef ML_H_
#define ML_H_

#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]
#define MAT_PRINT(m) ml_mat_print(m, #m)
#define ML_VERIFY_BUF_SIZE 512

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float (*Activate)(float);

typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float *es;
} Mat;

typedef struct {
  Activate f;
  Activate df;
} Act;

typedef struct {
  Mat z;
  Mat a;
  Mat w;
  Mat b;
  Act act;
} ModelLayer;

typedef struct {
  Mat w;
  Mat b;
  Mat d;
} GradLayer;

typedef struct {
  size_t layer_count;
  ModelLayer *layer;
} Mod;

typedef struct {
  size_t layer_count;
  GradLayer *layer;
} Grad;

Mat ml_mat_alloc(size_t rows, size_t cols, size_t stride);
void ml_mat_free(Mat *m);
void ml_mat_print(Mat m, const char *name);
void ml_mat_rand(Mat m, float low, float high);
void ml_mat_sum(Mat dst, Mat src);
void ml_mat_mul(Mat dst, Mat a, Mat b);
void ml_mat_shuffle(Mat m);
void ml_mat_scale(Mat m, float scale);
void ml_mat_zero(Mat m);

ModelLayer ml_modellayer_alloc(size_t cur_layer_size, size_t pre_layer_size,
                     size_t batch_size, Act act);
void ml_modellayer_free(ModelLayer *layer);

Mod ml_model_alloc(size_t *layer_sizes, size_t layer_count, size_t batch_size,
                   Act *acts);
void ml_model_free(Mod *m);
void ml_model_rand(Mod m, float low, float high);
void ml_model_print(Mod m);
void ml_model_forward(Mod m, Mat in);
float ml_model_cost(Mod m, Mat td);
void ml_model_finite_diff(Mod m, Grad g, Mat batch, float eps);
void ml_model_backprop(Mod m, Grad g, Mat batch);
void ml_model_train(Mod m, Grad g, float rate);
void ml_model_verify(Mod m, Mat td);
Mod ml_model_load(const char *file_path, size_t batch_size, Act *acts);
void ml_model_save(Mod m, const char *file_path);

GradLayer ml_gradlayer_alloc(Mod m, size_t l);
void ml_gradlayer_free(GradLayer *layer);

Grad ml_grad_alloc(Mod m);
void ml_grad_free(Grad *g);
void ml_grad_scale(Grad g, float scale);
void ml_grad_zero(Grad g);

Mat ml_load_td(const char *file_path);
void ml_model_copy_params(Mod dst, Mod src);
#endif // !ML_H_

// #define ML_IMPLEMENTATION
#ifdef ML_IMPLEMENTATION

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }
float sigmoidf(float x) { return 1.f / (expf(-x) + 1.f); }
float dsigmoidf(float x) { return sigmoidf(x) * (1 - sigmoidf(x)); }
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
float dtanhf(float x) { return 1 - (tanhf(x) * tanhf(x)); }

Act ML_SIGMOID = {.f = sigmoidf, .df = dsigmoidf};
Act ML_RELU = {.f = ReLUf, .df = dReLUf};
Act ML_LRELU = {.f = LReLUf, .df = dLReLUf};
Act ML_TANH = {.f = tanhf, .df = dtanhf};

void ml_act_func(Mat a, Mat z, float (*activate)(float)) {
  assert(a.rows > 0);
  assert(a.cols > 0);
  assert(a.es != NULL);

  assert(z.rows > 0);
  assert(z.cols > 0);
  assert(z.es != NULL);

  assert(a.rows == z.rows);
  assert(a.cols == z.cols);

  assert(activate != NULL);

  for (size_t i = 0; i < a.rows; ++i) {
    for (size_t j = 0; j < a.cols; ++j) {
      MAT_AT(a, i, j) = activate(MAT_AT(z, i, j));
    }
  }
}

Mat ml_mat_alloc(size_t rows, size_t cols, size_t stride) {
  if (rows == 0 || cols == 0 || stride == 0 || stride < cols) {
    fprintf(
        stderr,
        "ml_mat_alloc: invalid matrix param: rows=%zu, cols=%zu, stride=%zu\n",
        rows, cols, stride);
    exit(1);
  }

  Mat m = {
      .rows = rows,
      .cols = cols,
      .stride = stride,
      .es = (float *)malloc(rows * stride * sizeof(*m.es)),
  };

  assert(m.es != NULL);
  return m;
}

Mat ml_mat_slice(Mat m, size_t row, size_t col, size_t rows, size_t cols) {
  assert(m.es != NULL);
  assert(row + rows <= m.rows);
  assert(col + cols <= m.cols);

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
  assert(dst.rows == src.rows && dst.cols == src.cols);
  for (size_t i = 0; i < src.rows; ++i) {
    for (size_t j = 0; j < src.cols; ++j) {
      MAT_AT(dst, i, j) = MAT_AT(dst, i, j) + MAT_AT(src, i, j);
    }
  }
}

void ml_mat_mul(Mat dst, Mat a, Mat b) {
  assert(dst.es != a.es);
  assert(dst.es != b.es);
  assert(a.cols == b.rows);
  assert(dst.rows == a.rows && dst.cols == b.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = 0.f;
      for (size_t n = 0; n < a.cols; ++n) {
        MAT_AT(dst, i, j) += MAT_AT(a, i, n) * MAT_AT(b, n, j);
      }
    }
  }
}

void ml_mat_shuffle(Mat m) {
  if (m.rows == 0 || m.cols == 0 || m.es == NULL) {
    fprintf(stderr, "ml_mat_shuffle: Matrix is not valid\n");
    exit(1);
  }

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
  if (m.rows == 0 || m.cols == 0 || m.es == NULL) {
    fprintf(stderr, "ml_mat_scale: Matrix is not valid\n");
    exit(1);
  }
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) *= scale;
    }
  }
}

void ml_mat_zero(Mat m) {
  if (m.rows == 0 || m.cols == 0 || m.es == NULL) {
    fprintf(stderr, "ml_mat_zero: Matrix is not valid\n");
    exit(1);
  }
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

ModelLayer ml_modellayer_alloc(size_t cur_layer_size, size_t pre_layer_size,
                     size_t batch_size, Act act) {
  ModelLayer layer = {0};
  if (act.f == NULL || act.df == NULL) {
    fprintf(stderr, "ml_modellayer_alloc : Invalid activate function");
    exit(1);
  }

  layer.z = ml_mat_alloc(batch_size, cur_layer_size, cur_layer_size);
  layer.w = ml_mat_alloc(pre_layer_size, cur_layer_size, cur_layer_size);
  layer.a = ml_mat_alloc(batch_size, cur_layer_size, cur_layer_size);
  layer.b = ml_mat_alloc(1, cur_layer_size, cur_layer_size);
  layer.act = act;

  return layer;
}

void ml_modellayer_free(ModelLayer *layer) {

  ml_mat_free(&layer->z);
  ml_mat_free(&layer->w);
  ml_mat_free(&layer->a);
  ml_mat_free(&layer->b);

  layer->act.f = NULL;
  layer->act.df = NULL;
}

Mod ml_model_alloc(size_t *layer_sizes, size_t layer_count, size_t batch_size,
                   Act *act) {
  assert(layer_count > 1);
  Mod m = {0};

  m.layer_count = layer_count - 1;

  m.layer = (ModelLayer *)malloc(m.layer_count * sizeof(*m.layer));

  for (size_t i = 0; i < m.layer_count; ++i) {
    m.layer[i] =
        ml_modellayer_alloc(layer_sizes[i + 1], layer_sizes[i], batch_size, act[i]);
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

void ml_model_rand(Mod m, float low, float high) {
  for (size_t i = 0; i < m.layer_count; ++i) {
    ml_mat_rand(m.layer[i].w, low, high);
    ml_mat_rand(m.layer[i].b, low, high);
  }
}

void ml_model_print(Mod m) {
  assert(m.layer_count != 0);
  for (size_t i = 0; i < m.layer_count; ++i) {
    printf("layer %zu:\n", i);
    ml_mat_print(m.layer[i].w, "w");
    ml_mat_print(m.layer[i].b, "b");
    ml_mat_print(m.layer[i].a, "a");
    printf("\n");
  }
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
    ml_act_func(m.layer[i].a, m.layer[i].z, m.layer[i].act.f);
  }
}

float ml_model_cost(Mod m, Mat batch) {
  float res = 0.f;

  size_t input_size = m.layer[0].w.rows;
  size_t output_size = m.layer[m.layer_count - 1].a.cols;

  assert(input_size + output_size == batch.cols);

  Mat in = ml_mat_slice(batch, 0, 0, batch.rows, input_size);
  ml_model_forward(m, in);
  Mat out = m.layer[m.layer_count - 1].a;

  for (size_t i = 0; i < out.rows; ++i) {
    for (size_t j = 0; j < output_size; ++j) {
      float d = MAT_AT(batch, i, input_size + j) - MAT_AT(out, i, j);
      res += d * d;
    }
  }

  return res / batch.rows / output_size;
}

void ml_model_finite_diff(Mod m, Grad g, Mat batch, float eps) {
  assert(eps > 0);
  float c = ml_model_cost(m, batch);

  for (size_t i = 0; i < m.layer_count; ++i) {
    assert(m.layer[i].w.rows == g.layer[i].w.rows);
    assert(m.layer[i].w.cols == g.layer[i].w.cols);
    for (size_t j = 0; j < m.layer[i].w.rows; ++j) {
      for (size_t k = 0; k < m.layer[i].w.cols; ++k) {
        float save = MAT_AT(m.layer[i].w, j, k);
        MAT_AT(m.layer[i].w, j, k) += eps;
        MAT_AT(g.layer[i].w, j, k) = (ml_model_cost(m, batch) - c) / eps;
        MAT_AT(m.layer[i].w, j, k) = save;
      }
    }
    assert(m.layer[i].b.rows == g.layer[i].b.rows);
    assert(m.layer[i].b.cols == g.layer[i].b.cols);
    for (size_t j = 0; j < m.layer[i].b.rows; ++j) {
      for (size_t k = 0; k < m.layer[i].b.cols; ++k) {
        float save = MAT_AT(m.layer[i].b, j, k);
        MAT_AT(m.layer[i].b, j, k) += eps;
        MAT_AT(g.layer[i].b, j, k) = (ml_model_cost(m, batch) - c) / eps;
        MAT_AT(m.layer[i].b, j, k) = save;
      }
    }
  }
}

void ml_model_backprop(Mod m, Grad g, Mat batch) {
  if (m.layer_count == 0) {
    fprintf(stderr,
            "ml_model_backprop : Invalid model : "
            "layer_count  = %zu",
            m.layer_count);
    exit(1);
  }
  ml_grad_zero(g);

  size_t input_size = m.layer[0].w.rows;
  size_t output_size = m.layer[m.layer_count - 1].a.cols;
  if (batch.cols != input_size + output_size) {
    fprintf(stderr,
            "ml_model_backprop : Invalid input/output layer size of model : "
            "input = %zu , output = %zu",
            input_size, output_size);
    exit(1);
  }

  Mat in = ml_mat_slice(batch, 0, 0, batch.rows, input_size);
  Mat target = ml_mat_slice(batch, 0, input_size, batch.rows, output_size);
  ml_model_forward(m, in);

  for (size_t i = 0; i < batch.rows; ++i) {
    for (size_t j = 0; j < output_size; ++j) {
      MAT_AT(g.layer[m.layer_count - 1].d, i, j) =
          2 *
          (MAT_AT(m.layer[m.layer_count - 1].a, i, j) - MAT_AT(target, i, j)) *
          m.layer[m.layer_count - 1].act.df(
              MAT_AT(m.layer[m.layer_count - 1].z, i, j));
    }
  }

  for (size_t l = m.layer_count - 1; l-- > 0;) {
    for (size_t i = 0; i < batch.rows; ++i) {
      for (size_t j = 0; j < m.layer[l].w.cols; ++j) {
        float sum = 0.f;
        for (size_t k = 0; k < m.layer[l + 1].w.cols; ++k) {
          sum +=
              MAT_AT(g.layer[l + 1].d, i, k) * MAT_AT(m.layer[l + 1].w, j, k);
        }
        MAT_AT(g.layer[l].d, i, j) =
            sum * m.layer[l].act.df(MAT_AT(m.layer[l].z, i, j));
      }
    }
  }

  for (size_t l = m.layer_count; l-- > 0;) {
    Mat prev = l == 0 ? in : m.layer[l - 1].a;
    for (size_t i = 0; i < batch.rows; ++i) {
      for (size_t j = 0; j < m.layer[l].w.cols; ++j) {
        for (size_t k = 0; k < m.layer[l].w.rows; ++k) {
          MAT_AT(g.layer[l].w, k, j) +=
              MAT_AT(prev, i, k) * MAT_AT(g.layer[l].d, i, j);
        }
        MAT_AT(g.layer[l].b, 0, j) += MAT_AT(g.layer[l].d, i, j);
      }
    }
  }

  ml_grad_scale(g, 1.f / (batch.rows * output_size));
}

void ml_model_train(Mod m, Grad g, float rate) {
  assert(rate > 0);
  assert(m.layer_count == g.layer_count);
  for (size_t i = 0; i < m.layer_count; ++i) {
    if (m.layer[i].w.rows != g.layer[i].w.rows ||
        m.layer[i].w.cols != g.layer[i].w.cols) {
      fprintf(stderr,
              "ml_model_train: weight gradient shape mismatch at layer %zu: "
              "m.layer[%zu].w is %zux%zu, but g.w[%zu] is %zux%zu\n",
              i, i, m.layer[i].w.rows, m.layer[i].w.cols, i, g.layer[i].w.rows,
              g.layer[i].w.cols);
      exit(1);
    }
    for (size_t j = 0; j < m.layer[i].w.rows; ++j) {
      for (size_t k = 0; k < m.layer[i].w.cols; ++k) {
        MAT_AT(m.layer[i].w, j, k) -= MAT_AT(g.layer[i].w, j, k) * rate;
      }
    }
    if (m.layer[i].b.rows != g.layer[i].b.rows ||
        m.layer[i].b.cols != g.layer[i].b.cols) {
      fprintf(stderr,
              "ml_model_train: bias gradient shape mismatch at layer %zu: "
              "m.layer[%zu].b is %zux%zu, but g.b[%zu] is %zux%zu\n",
              i, i, m.layer[i].b.rows, m.layer[i].b.cols, i, g.layer[i].b.rows,
              g.layer[i].b.cols);
      exit(1);
    }
    for (size_t j = 0; j < m.layer[i].b.rows; ++j) {
      for (size_t k = 0; k < m.layer[i].b.cols; ++k) {
        MAT_AT(m.layer[i].b, j, k) -= MAT_AT(g.layer[i].b, j, k) * rate;
      }
    }
  }
}

static int ml_binary(float x) { return x >= 0.5f ? 1 : 0; }

static void ml_format_vec_float(char *buf, size_t buf_size, Mat m, size_t row,
                                size_t offset, size_t cols) {
  size_t n = 0;

  n += snprintf(buf + n, buf_size - n, "[");

  for (size_t j = 0; j < cols && n < buf_size; ++j) {
    n += snprintf(buf + n, buf_size - n, "%s%.3f", j ? " " : "",
                  MAT_AT(m, row, offset + j));
  }

  snprintf(buf + n, buf_size - n, "]");
}

static void ml_format_vec_binary(char *buf, size_t buf_size, Mat m, size_t row,
                                 size_t offset, size_t cols) {
  size_t n = 0;

  n += snprintf(buf + n, buf_size - n, "[");

  for (size_t j = 0; j < cols && n < buf_size; ++j) {
    n += snprintf(buf + n, buf_size - n, "%s%d", j ? " " : "",
                  ml_binary(MAT_AT(m, row, offset + j)));
  }

  snprintf(buf + n, buf_size - n, "]");
}

void ml_model_verify(Mod m, Mat td) {
  if (m.layer_count == 0) {
    fprintf(stderr, "ml_model_verify: invalid model: layer_count=%zu\n",
            m.layer_count);
    exit(1);
  }

  if (td.rows == 0 || td.es == NULL) {
    fprintf(stderr, "ml_model_verify: invalid training data\n");
    exit(1);
  }

  size_t input_size = m.layer[0].w.rows;
  size_t output_size = m.layer[m.layer_count - 1].a.cols;
  size_t batch_size = m.layer[m.layer_count - 1].a.rows;

  if (td.cols != input_size + output_size) {
    fprintf(stderr,
            "ml_model_verify: invalid td shape: td.cols=%zu, expected=%zu\n",
            td.cols, input_size + output_size);
    exit(1);
  }

  if (batch_size == 0) {
    fprintf(stderr, "ml_model_verify: invalid batch_size=0\n");
    exit(1);
  }

  if (td.rows % batch_size != 0) {
    fprintf(stderr,
            "ml_model_verify: td.rows must be divisible by batch_size: "
            "td.rows=%zu, batch_size=%zu\n",
            td.rows, batch_size);
    exit(1);
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
        printf("wrong predictions:\n");
        printf("%-5s %-37s %-25s %-13s %-13s\n", "idx", "input", "output",
               "pred", "expected");
      }

      wrong_samples++;

      char input_buf[ML_VERIFY_BUF_SIZE];
      char output_buf[ML_VERIFY_BUF_SIZE];
      char pred_buf[ML_VERIFY_BUF_SIZE];
      char expected_buf[ML_VERIFY_BUF_SIZE];

      ml_format_vec_float(input_buf, sizeof(input_buf), in, i, 0, input_size);

      ml_format_vec_float(output_buf, sizeof(output_buf), out, i, 0,
                          output_size);

      ml_format_vec_binary(pred_buf, sizeof(pred_buf), out, i, 0, output_size);

      ml_format_vec_binary(expected_buf, sizeof(expected_buf), target, i, 0,
                           output_size);

      printf("%-5zu %-37s %-25s %-13s %-13s\n", begin + i, input_buf,
             output_buf, pred_buf, expected_buf);
    }
  }

  if (wrong_samples == 0) {
    printf("verify: all samples correct.\n");
  } else {
    printf("wrong samples: %zu/%zu\n", wrong_samples, td.rows);
  }

  printf("sample accuracy: %zu/%zu = %.2f%%\n", correct_samples, td.rows,
         100.f * (float)correct_samples / (float)td.rows);

  printf("output accuracy: %zu/%zu = %.2f%%\n", correct_outputs, total_outputs,
         100.f * (float)correct_outputs / (float)total_outputs);
}

void ml_model_save(Mod m, const char *file_path) {
  FILE *fd = fopen(file_path, "wb");
  if (fd == NULL) {
    fprintf(stderr, "ml_model_save: failed to open file: %s\n", file_path);
    exit(1);
  }

  if (m.layer_count == 0) {
    fprintf(stderr, "ml_model_save: invalid model: layer_count = 0\n");
    fclose(fd);
    exit(1);
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

    uint64_t b_rows = (uint64_t)m.layer[i].w.rows;
    uint64_t b_cols = (uint64_t)m.layer[i].w.cols;

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
}

Mod ml_model_load(const char *file_path, size_t batch_size, Act *acts) {
  FILE *fd = fopen(file_path, "rb");
  if (fd == NULL) {
    fprintf(stderr, "ml_model_load: failed to open file: %s\n", file_path);
    exit(1);
  }

  if (batch_size == 0) {
    fprintf(stderr, "ml_model_load: invalid batch_size = 0\n");
    fclose(fd);
    exit(1);
  }

  if (acts == NULL) {
    fprintf(stderr, "ml_model_load: activates is NULL\n");
    fclose(fd);
    exit(1);
  }

  char magic[4];
  size_t nread = fread(magic, 1, 4, fd);
  if (nread != 4 || memcmp(magic, "MLMB", 4) != 0) {
    fprintf(stderr, "ml_model_load: invalid model file magic\n");
    fclose(fd);
    exit(1);
  }

  uint64_t file_layer_count = 0;
  nread = fread(&file_layer_count, sizeof(file_layer_count), 1, fd);
  if (nread != 1 || file_layer_count == 0) {
    fprintf(stderr, "ml_model_load: invalid layer_count\n");
    fclose(fd);
    exit(1);
  }

  size_t layer_count = (size_t)file_layer_count;

  size_t *layer_sizes =
      (size_t *)malloc((layer_count + 1) * sizeof(*layer_sizes));
  if (layer_sizes == NULL) {
    fprintf(stderr, "ml_model_load: failed to allocate layer_sizes\n");
    fclose(fd);
    exit(1);
  }

  /*
    第一遍：只读取形状，推导 layer_sizes。
  */
  for (size_t i = 0; i < layer_count; ++i) {
    char type = 0;
    uint64_t rows = 0;
    uint64_t cols = 0;

    nread = fread(&type, 1, 1, fd);
    if (nread != 1 || type != 'W') {
      fprintf(stderr, "ml_model_load: expected W at layer %zu\n", i);
      free(layer_sizes);
      fclose(fd);
      exit(1);
    }

    nread = fread(&rows, sizeof(rows), 1, fd);
    if (nread != 1) {
      fprintf(stderr, "ml_model_load: failed to read W rows at layer %zu\n", i);
      free(layer_sizes);
      fclose(fd);
      exit(1);
    }

    nread = fread(&cols, sizeof(cols), 1, fd);
    if (nread != 1) {
      fprintf(stderr, "ml_model_load: failed to read W cols at layer %zu\n", i);
      free(layer_sizes);
      fclose(fd);
      exit(1);
    }

    if (rows == 0 || cols == 0) {
      fprintf(stderr, "ml_model_load: invalid W shape at layer %zu: %zux%zu\n",
              i, (size_t)rows, (size_t)cols);
      free(layer_sizes);
      fclose(fd);
      exit(1);
    }

    if (i == 0) {
      layer_sizes[0] = (size_t)rows;
    } else if (layer_sizes[i] != (size_t)rows) {
      fprintf(stderr,
              "ml_model_load: W input mismatch at layer %zu: "
              "expected %zu, got %zu\n",
              i, layer_sizes[i], (size_t)rows);
      free(layer_sizes);
      fclose(fd);
      exit(1);
    }

    layer_sizes[i + 1] = (size_t)cols;

    if (fseek(fd, (long)(rows * cols * sizeof(float)), SEEK_CUR) != 0) {
      fprintf(stderr, "ml_model_load: failed to skip W data at layer %zu\n", i);
      free(layer_sizes);
      fclose(fd);
      exit(1);
    }

    nread = fread(&type, 1, 1, fd);
    if (nread != 1 || type != 'B') {
      fprintf(stderr, "ml_model_load: expected B at layer %zu\n", i);
      free(layer_sizes);
      fclose(fd);
      exit(1);
    }

    nread = fread(&rows, sizeof(rows), 1, fd);
    if (nread != 1) {
      fprintf(stderr, "ml_model_load: failed to read B rows at layer %zu\n", i);
      free(layer_sizes);
      fclose(fd);
      exit(1);
    }

    nread = fread(&cols, sizeof(cols), 1, fd);
    if (nread != 1) {
      fprintf(stderr, "ml_model_load: failed to read B cols at layer %zu\n", i);
      free(layer_sizes);
      fclose(fd);
      exit(1);
    }

    if (rows != 1 || (size_t)cols != layer_sizes[i + 1]) {
      fprintf(stderr,
              "ml_model_load: invalid B shape at layer %zu: "
              "got %zux%zu, expected 1x%zu\n",
              i, (size_t)rows, (size_t)cols, layer_sizes[i + 1]);
      free(layer_sizes);
      fclose(fd);
      exit(1);
    }

    if (fseek(fd, (long)(rows * cols * sizeof(float)), SEEK_CUR) != 0) {
      fprintf(stderr, "ml_model_load: failed to skip B data at layer %zu\n", i);
      free(layer_sizes);
      fclose(fd);
      exit(1);
    }
  }

  Mod m = ml_model_alloc(layer_sizes, layer_count + 1, batch_size, acts);
  free(layer_sizes);

  rewind(fd);

  nread = fread(magic, 1, 4, fd);
  if (nread != 4 || memcmp(magic, "MLMB", 4) != 0) {
    fprintf(stderr, "ml_model_load: invalid magic on second read\n");
    ml_model_free(&m);
    fclose(fd);
    exit(1);
  }

  file_layer_count = 0;
  nread = fread(&file_layer_count, sizeof(file_layer_count), 1, fd);
  if (nread != 1 || (size_t)file_layer_count != m.layer_count) {
    fprintf(stderr,
            "ml_model_load: layer_count mismatch: file=%zu, model=%zu\n",
            (size_t)file_layer_count, m.layer_count);
    ml_model_free(&m);
    fclose(fd);
    exit(1);
  }

  /*
    第二遍：真正读取 W/B 数据。
  */
  for (size_t i = 0; i < m.layer_count; ++i) {
    char type = 0;
    uint64_t rows = 0;
    uint64_t cols = 0;

    ModelLayer *ly = &m.layer[i];

    nread = fread(&type, 1, 1, fd);
    if (nread != 1 || type != 'W') {
      fprintf(stderr, "ml_model_load: expected W at layer %zu\n", i);
      ml_model_free(&m);
      fclose(fd);
      exit(1);
    }

    nread = fread(&rows, sizeof(rows), 1, fd);
    if (nread != 1) {
      fprintf(stderr, "ml_model_load: failed to read W rows at layer %zu\n", i);
      ml_model_free(&m);
      fclose(fd);
      exit(1);
    }

    nread = fread(&cols, sizeof(cols), 1, fd);
    if (nread != 1) {
      fprintf(stderr, "ml_model_load: failed to read W cols at layer %zu\n", i);
      ml_model_free(&m);
      fclose(fd);
      exit(1);
    }

    if ((size_t)rows != ly->w.rows || (size_t)cols != ly->w.cols) {
      fprintf(stderr,
              "ml_model_load: W shape mismatch at layer %zu: "
              "file=%zux%zu, model=%zux%zu\n",
              i, (size_t)rows, (size_t)cols, ly->w.rows, ly->w.cols);
      ml_model_free(&m);
      fclose(fd);
      exit(1);
    }

    for (size_t r = 0; r < ly->w.rows; ++r) {
      for (size_t c = 0; c < ly->w.cols; ++c) {
        float x = 0.0f;

        nread = fread(&x, sizeof(x), 1, fd);
        if (nread != 1) {
          fprintf(stderr,
                  "ml_model_load: failed to read W data at layer %zu, "
                  "row %zu, col %zu\n",
                  i, r, c);
          ml_model_free(&m);
          fclose(fd);
          exit(1);
        }

        MAT_AT(ly->w, r, c) = x;
      }
    }

    nread = fread(&type, 1, 1, fd);
    if (nread != 1 || type != 'B') {
      fprintf(stderr, "ml_model_load: expected B at layer %zu\n", i);
      ml_model_free(&m);
      fclose(fd);
      exit(1);
    }

    nread = fread(&rows, sizeof(rows), 1, fd);
    if (nread != 1) {
      fprintf(stderr, "ml_model_load: failed to read B rows at layer %zu\n", i);
      ml_model_free(&m);
      fclose(fd);
      exit(1);
    }

    nread = fread(&cols, sizeof(cols), 1, fd);
    if (nread != 1) {
      fprintf(stderr, "ml_model_load: failed to read B cols at layer %zu\n", i);
      ml_model_free(&m);
      fclose(fd);
      exit(1);
    }

    if ((size_t)rows != ly->b.rows || (size_t)cols != ly->b.cols) {
      fprintf(stderr,
              "ml_model_load: B shape mismatch at layer %zu: "
              "file=%zux%zu, model=%zux%zu\n",
              i, (size_t)rows, (size_t)cols, ly->b.rows, ly->b.cols);
      ml_model_free(&m);
      fclose(fd);
      exit(1);
    }

    for (size_t r = 0; r < ly->b.rows; ++r) {
      for (size_t c = 0; c < ly->b.cols; ++c) {
        float x = 0.0f;

        nread = fread(&x, sizeof(x), 1, fd);
        if (nread != 1) {
          fprintf(stderr,
                  "ml_model_load: failed to read B data at layer %zu, "
                  "row %zu, col %zu\n",
                  i, r, c);
          ml_model_free(&m);
          fclose(fd);
          exit(1);
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

  layer.w =
      ml_mat_alloc(m.layer[l].w.rows, m.layer[l].w.cols, m.layer[l].w.stride);
  layer.b =
      ml_mat_alloc(m.layer[l].b.rows, m.layer[l].b.cols, m.layer[l].b.stride);
  layer.d =
      ml_mat_alloc(m.layer[l].a.rows, m.layer[l].a.cols, m.layer[l].a.stride);

  return layer;
}

void ml_gradlayer_free(GradLayer *layer) {

  ml_mat_free(&layer->w);
  ml_mat_free(&layer->b);
  ml_mat_free(&layer->d);
}

Grad ml_grad_alloc(Mod m) {
  assert(m.layer_count > 0);
  assert(m.layer != NULL);

  Grad g = {0};

  g.layer_count = m.layer_count;
  g.layer = (GradLayer *)malloc(g.layer_count * sizeof(*g.layer));

  if (g.layer == NULL) {
    fprintf(stderr, "ml_grad_alloc: failed to allocate grad layers\n");
    exit(1);
  }

  for (size_t i = 0; i < g.layer_count; ++i) {
    g.layer[i] = ml_gradlayer_alloc(m, i);
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
    fprintf(stderr, "ml_grad_scale: invalid grad: layer_count=%zu, layer=%p\n",
            g.layer_count, (void *)g.layer);
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
    fprintf(stderr, "ml_grad_zero: invalid grad: layer_count=%zu, layer=%p\n",
            g.layer_count, (void *)g.layer);
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
  assert(line != NULL);

  line[0] = '\0';

  while (fgets(line + len, cap - len, fd) != NULL) {
    len += strlen(line + len);

    if (len > 0 && line[len - 1] == '\n')
      return line;

    if (feof(fd))
      return line;

    cap *= 2;

    char *new_line = (char *)realloc(line, cap);
    assert(new_line != NULL);
    line = new_line;
  }
  free(line);
  return NULL;
}

Mat ml_load_td(const char *file_path) {
  FILE *fd = fopen(file_path, "r");
  assert(fd != NULL);
  Mat td = {0};

  char *line = NULL;

  line = read_line_from_file(fd);
  assert(line != NULL);

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

  assert(rows > 0);
  assert(cols > 0);

  td = ml_mat_alloc(rows, cols, cols);

  rewind(fd);

  line = read_line_from_file(fd);
  assert(line != NULL);
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
      assert(c < cols);

      char *end = NULL;
      float value = strtof(token, &end);

      assert(end != token);

      MAT_AT(td, r, c) = value;

      c++;
      token = strtok(NULL, ",\n\r");
    }

    assert(c == cols);
    r++;
    free(line);
    line = NULL;
  }

  assert(r == rows);

  fclose(fd);
  return td;
}

void ml_model_copy_params(Mod dst, Mod src) {
  if (dst.layer_count != src.layer_count) {
    fprintf(stderr, "ml_model_copy_params: layer_count mismatch\n");
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
