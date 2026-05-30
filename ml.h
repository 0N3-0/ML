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

typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float *es;
} Mat;

typedef struct {
  size_t layer_count;
  Mat *z;
  Mat *a;
  Mat *w;
  Mat *b;
} Mod;

typedef struct {
  size_t layer_count;
  Mat *w;
  Mat *b;
  Mat *d;
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

Mod ml_model_alloc(size_t *layer_sizes, size_t layer_count, size_t batch_size);
void ml_model_free(Mod *m);
void ml_model_rand(Mod m, float low, float high);
void ml_model_print(Mod m);
void ml_model_forward(Mod m, Mat in, float (*activate)(float));
float ml_model_cost(Mod m, Mat td, float (*activate)(float));
void ml_model_finite_diff(Mod m, Grad g, Mat batch, float eps,
                          float (*activate)(float));
void ml_model_backprop(Mod m, Grad g, Mat batch, float (*dactivate)(float),
                       float (*activate)(float));
void ml_model_train(Mod m, Grad g, float rate);
void ml_model_verify(Mod m, Mat td, float (*activate)(float));
Mod ml_model_load(const char *file_path, size_t batch_size);
void ml_model_save(Mod m, const char *file_path);

Grad ml_grad_alloc(Mod m);
void ml_grad_free(Grad *g);
void ml_grad_scale(Grad g, float scale);
void ml_grad_zero(Grad g);

Mat ml_load_td(const char *file_path);
#endif // !ML_H_

// #define ML_IMPLEMENTATION
#ifdef ML_IMPLEMENTATION

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }
float sigmoidf(float x) { return 1.f / (expf(-x) + 1.f); }
float dsigmoidf(float x) { return sigmoidf(x) * (1 - sigmoidf(x)); }

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

Mod ml_model_alloc(size_t *layer_sizes, size_t layer_count, size_t batch_size) {
  assert(layer_count > 1);
  Mod m = {0};

  m.layer_count = layer_count - 1;

  m.z = (Mat *)malloc(m.layer_count * sizeof(*m.z));
  m.w = (Mat *)malloc(m.layer_count * sizeof(*m.w));
  m.a = (Mat *)malloc(m.layer_count * sizeof(*m.a));
  m.b = (Mat *)malloc(m.layer_count * sizeof(*m.b));

  assert(m.z != NULL);
  assert(m.w != NULL);
  assert(m.a != NULL);
  assert(m.b != NULL);

  for (size_t i = 0; i < m.layer_count; ++i) {
    m.z[i] = ml_mat_alloc(batch_size, layer_sizes[i + 1], layer_sizes[i + 1]);
    m.w[i] =
        ml_mat_alloc(layer_sizes[i], layer_sizes[i + 1], layer_sizes[i + 1]);
    m.a[i] = ml_mat_alloc(batch_size, layer_sizes[i + 1], layer_sizes[i + 1]);
    m.b[i] = ml_mat_alloc(1, layer_sizes[i + 1], layer_sizes[i + 1]);
  }

  return m;
}

void ml_model_free(Mod *m) {
  if (m == NULL)
    return;

  for (size_t i = 0; i < m->layer_count; ++i) {
    ml_mat_free(&m->z[i]);
    ml_mat_free(&m->w[i]);
    ml_mat_free(&m->a[i]);
    ml_mat_free(&m->b[i]);
  }

  free(m->z);
  free(m->w);
  free(m->a);
  free(m->b);

  m->z = NULL;
  m->w = NULL;
  m->a = NULL;
  m->b = NULL;
  m->layer_count = 0;
}

void ml_model_rand(Mod m, float low, float high) {
  for (size_t i = 0; i < m.layer_count; ++i) {
    ml_mat_rand(m.w[i], low, high);
    ml_mat_rand(m.b[i], low, high);
  }
}

void ml_model_print(Mod m) {
  assert(m.layer_count != 0);
  for (size_t i = 0; i < m.layer_count; ++i) {
    printf("layer %zu:\n", i);
    ml_mat_print(m.w[i], "w");
    ml_mat_print(m.b[i], "b");
    ml_mat_print(m.a[i], "a");
    printf("\n");
  }
}

void ml_model_forward(Mod m, Mat in, float (*activate)(float)) {
  for (size_t i = 0; i < m.layer_count; ++i) {
    if (!i)
      ml_mat_mul(m.z[i], in, m.w[i]);
    else
      ml_mat_mul(m.z[i], m.a[i - 1], m.w[i]);
    for (size_t j = 0; j < m.z[i].rows; ++j) {
      Mat t = ml_mat_slice(m.z[i], j, 0, 1, m.z[i].cols);
      ml_mat_sum(t, m.b[i]);
    }
    ml_act_func(m.a[i], m.z[i], activate);
  }
}

float ml_model_cost(Mod m, Mat batch, float (*activate)(float)) {
  float res = 0.f;

  size_t input_size = m.w[0].rows;
  size_t output_size = m.a[m.layer_count - 1].cols;

  assert(input_size + output_size == batch.cols);

  Mat in = ml_mat_slice(batch, 0, 0, batch.rows, input_size);
  ml_model_forward(m, in, activate);
  Mat out = m.a[m.layer_count - 1];

  for (size_t i = 0; i < out.rows; ++i) {
    for (size_t j = 0; j < output_size; ++j) {
      float d = MAT_AT(batch, i, input_size + j) - MAT_AT(out, i, j);
      res += d * d;
    }
  }

  return res / batch.rows / output_size;
}

void ml_model_finite_diff(Mod m, Grad g, Mat batch, float eps,
                          float (*activate)(float)) {
  assert(eps > 0);
  float c = ml_model_cost(m, batch, activate);

  for (size_t i = 0; i < m.layer_count; ++i) {
    assert(m.w[i].rows == g.w[i].rows);
    assert(m.w[i].cols == g.w[i].cols);
    for (size_t j = 0; j < m.w[i].rows; ++j) {
      for (size_t k = 0; k < m.w[i].cols; ++k) {
        float save = MAT_AT(m.w[i], j, k);
        MAT_AT(m.w[i], j, k) += eps;
        MAT_AT(g.w[i], j, k) = (ml_model_cost(m, batch, activate) - c) / eps;
        MAT_AT(m.w[i], j, k) = save;
      }
    }
    assert(m.b[i].rows == g.b[i].rows);
    assert(m.b[i].cols == g.b[i].cols);
    for (size_t j = 0; j < m.b[i].rows; ++j) {
      for (size_t k = 0; k < m.b[i].cols; ++k) {
        float save = MAT_AT(m.b[i], j, k);
        MAT_AT(m.b[i], j, k) += eps;
        MAT_AT(g.b[i], j, k) = (ml_model_cost(m, batch, activate) - c) / eps;
        MAT_AT(m.b[i], j, k) = save;
      }
    }
  }
}

void ml_model_backprop(Mod m, Grad g, Mat batch, float (*dactivate)(float),
                       float (*activate)(float)) {
  if (m.layer_count == 0) {
    fprintf(stderr,
            "ml_model_backprop : Invalid model : "
            "layer_count  = %zu",
            m.layer_count);
    exit(1);
  }
  ml_grad_zero(g);

  size_t input_size = m.w[0].rows;
  size_t output_size = m.a[m.layer_count - 1].cols;
  if (batch.cols != input_size + output_size) {
    fprintf(stderr,
            "ml_model_backprop : Invalid input/output layer size of model : "
            "input = %zu , output = %zu",
            input_size, output_size);
    exit(1);
  }

  Mat in = ml_mat_slice(batch, 0, 0, batch.rows, input_size);
  Mat target = ml_mat_slice(batch, 0, input_size, batch.rows, output_size);
  ml_model_forward(m, in, activate);

  for (size_t i = 0; i < batch.rows; ++i) {
    for (size_t j = 0; j < output_size; ++j) {
      MAT_AT(g.d[m.layer_count - 1], i, j) =
          2 * (MAT_AT(m.a[m.layer_count - 1], i, j) - MAT_AT(target, i, j)) *
          dactivate(MAT_AT(m.z[m.layer_count - 1], i, j));
    }
  }

  for (size_t l = m.layer_count - 1; l-- > 0;) {
    for (size_t i = 0; i < batch.rows; ++i) {
      for (size_t j = 0; j < m.w[l].cols; ++j) {
        float sum = 0.f;
        for (size_t k = 0; k < m.w[l + 1].cols; ++k) {
          sum += MAT_AT(g.d[l + 1], i, k) * MAT_AT(m.w[l + 1], j, k);
        }
        MAT_AT(g.d[l], i, j) = sum * dactivate(MAT_AT(m.z[l], i, j));
      }
    }
  }

  for (size_t l = m.layer_count; l-- > 0;) {
    Mat prev = l == 0 ? in : m.a[l - 1];
    for (size_t i = 0; i < batch.rows; ++i) {
      for (size_t j = 0; j < m.w[l].cols; ++j) {
        for (size_t k = 0; k < m.w[l].rows; ++k) {
          MAT_AT(g.w[l], k, j) += MAT_AT(prev, i, k) * MAT_AT(g.d[l], i, j);
        }
        MAT_AT(g.b[l], 0, j) += MAT_AT(g.d[l], i, j);
      }
    }
  }

  ml_grad_scale(g, 1.f / (batch.rows * output_size));
}

void ml_model_train(Mod m, Grad g, float rate) {
  assert(rate > 0);
  assert(m.layer_count == g.layer_count);
  for (size_t i = 0; i < m.layer_count; ++i) {
    assert(m.w[i].rows == g.w[i].rows);
    assert(m.w[i].cols == g.w[i].cols);
    for (size_t j = 0; j < m.w[i].rows; ++j) {
      for (size_t k = 0; k < m.w[i].cols; ++k) {
        MAT_AT(m.w[i], j, k) -= MAT_AT(g.w[i], j, k) * rate;
      }
    }
    assert(m.b[i].rows == g.b[i].rows);
    assert(m.b[i].cols == g.b[i].cols);
    for (size_t j = 0; j < m.b[i].rows; ++j) {
      for (size_t k = 0; k < m.b[i].cols; ++k) {
        MAT_AT(m.b[i], j, k) -= MAT_AT(g.b[i], j, k) * rate;
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

void ml_model_verify(Mod m, Mat td, float (*activate)(float)) {
  if (m.layer_count == 0) {
    fprintf(stderr, "ml_model_verify: invalid model: layer_count=%zu\n",
            m.layer_count);
    exit(1);
  }

  if (td.rows == 0 || td.es == NULL) {
    fprintf(stderr, "ml_model_verify: invalid training data\n");
    exit(1);
  }

  if (activate == NULL) {
    fprintf(stderr, "ml_model_verify: activate function is NULL\n");
    exit(1);
  }

  size_t input_size = m.w[0].rows;
  size_t output_size = m.a[m.layer_count - 1].cols;
  size_t batch_size = m.a[m.layer_count - 1].rows;

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

    ml_model_forward(m, in, activate);

    Mat out = m.a[m.layer_count - 1];

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
    uint64_t w_rows = (uint64_t)m.w[i].rows;
    uint64_t w_cols = (uint64_t)m.w[i].cols;

    fwrite("W", 1, 1, fd);
    fwrite(&w_rows, sizeof(w_rows), 1, fd);
    fwrite(&w_cols, sizeof(w_cols), 1, fd);

    for (size_t r = 0; r < m.w[i].rows; ++r) {
      for (size_t c = 0; c < m.w[i].cols; ++c) {
        float x = MAT_AT(m.w[i], r, c);
        fwrite(&x, sizeof(x), 1, fd);
      }
    }

    uint64_t b_rows = (uint64_t)m.b[i].rows;
    uint64_t b_cols = (uint64_t)m.b[i].cols;

    fwrite("B", 1, 1, fd);
    fwrite(&b_rows, sizeof(b_rows), 1, fd);
    fwrite(&b_cols, sizeof(b_cols), 1, fd);

    for (size_t r = 0; r < m.b[i].rows; ++r) {
      for (size_t c = 0; c < m.b[i].cols; ++c) {
        float x = MAT_AT(m.b[i], r, c);
        fwrite(&x, sizeof(x), 1, fd);
      }
    }
  }

  fclose(fd);
}

Mod ml_model_load(const char *file_path, size_t batch_size) {
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

  size_t *layer = malloc((layer_count + 1) * sizeof(*layer));
  if (layer == NULL) {
    fprintf(stderr, "ml_model_load: failed to allocate layer array\n");
    fclose(fd);
    exit(1);
  }

  /*
    第一遍：只读取形状，恢复 layer[]。
  */
  for (size_t i = 0; i < layer_count; ++i) {
    char type = 0;
    uint64_t rows = 0;
    uint64_t cols = 0;

    nread = fread(&type, 1, 1, fd);
    if (nread != 1 || type != 'W') {
      fprintf(stderr, "ml_model_load: expected W at layer %zu\n", i);
      free(layer);
      fclose(fd);
      exit(1);
    }

    nread = fread(&rows, sizeof(rows), 1, fd);
    if (nread != 1) {
      fprintf(stderr, "ml_model_load: failed to read W rows at layer %zu\n", i);
      free(layer);
      fclose(fd);
      exit(1);
    }

    nread = fread(&cols, sizeof(cols), 1, fd);
    if (nread != 1) {
      fprintf(stderr, "ml_model_load: failed to read W cols at layer %zu\n", i);
      free(layer);
      fclose(fd);
      exit(1);
    }

    if (rows == 0 || cols == 0) {
      fprintf(stderr, "ml_model_load: invalid W shape at layer %zu: %zux%zu\n",
              i, (size_t)rows, (size_t)cols);
      free(layer);
      fclose(fd);
      exit(1);
    }

    if (i == 0) {
      layer[0] = (size_t)rows;
    } else if (layer[i] != (size_t)rows) {
      fprintf(stderr,
              "ml_model_load: W input mismatch at layer %zu: "
              "expected %zu, got %zu\n",
              i, layer[i], (size_t)rows);
      free(layer);
      fclose(fd);
      exit(1);
    }

    layer[i + 1] = (size_t)cols;

    if (fseek(fd, (long)(rows * cols * sizeof(float)), SEEK_CUR) != 0) {
      fprintf(stderr, "ml_model_load: failed to skip W data at layer %zu\n", i);
      free(layer);
      fclose(fd);
      exit(1);
    }

    nread = fread(&type, 1, 1, fd);
    if (nread != 1 || type != 'B') {
      fprintf(stderr, "ml_model_load: expected B at layer %zu\n", i);
      free(layer);
      fclose(fd);
      exit(1);
    }

    nread = fread(&rows, sizeof(rows), 1, fd);
    if (nread != 1) {
      fprintf(stderr, "ml_model_load: failed to read B rows at layer %zu\n", i);
      free(layer);
      fclose(fd);
      exit(1);
    }

    nread = fread(&cols, sizeof(cols), 1, fd);
    if (nread != 1) {
      fprintf(stderr, "ml_model_load: failed to read B cols at layer %zu\n", i);
      free(layer);
      fclose(fd);
      exit(1);
    }

    if (rows != 1 || (size_t)cols != layer[i + 1]) {
      fprintf(stderr,
              "ml_model_load: invalid B shape at layer %zu: "
              "got %zux%zu, expected 1x%zu\n",
              i, (size_t)rows, (size_t)cols, layer[i + 1]);
      free(layer);
      fclose(fd);
      exit(1);
    }

    if (fseek(fd, (long)(rows * cols * sizeof(float)), SEEK_CUR) != 0) {
      fprintf(stderr, "ml_model_load: failed to skip B data at layer %zu\n", i);
      free(layer);
      fclose(fd);
      exit(1);
    }
  }

  Mod m = ml_model_alloc(layer, layer_count + 1, batch_size);
  free(layer);

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

    if ((size_t)rows != m.w[i].rows || (size_t)cols != m.w[i].cols) {
      fprintf(stderr,
              "ml_model_load: W shape mismatch at layer %zu: "
              "file=%zux%zu, model=%zux%zu\n",
              i, (size_t)rows, (size_t)cols, m.w[i].rows, m.w[i].cols);
      ml_model_free(&m);
      fclose(fd);
      exit(1);
    }

    for (size_t r = 0; r < m.w[i].rows; ++r) {
      for (size_t c = 0; c < m.w[i].cols; ++c) {
        float x = 0.f;
        nread = fread(&x, sizeof(x), 1, fd);
        if (nread != 1) {
          fprintf(stderr, "ml_model_load: failed to read W data at layer %zu\n",
                  i);
          ml_model_free(&m);
          fclose(fd);
          exit(1);
        }

        MAT_AT(m.w[i], r, c) = x;
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

    if ((size_t)rows != m.b[i].rows || (size_t)cols != m.b[i].cols) {
      fprintf(stderr,
              "ml_model_load: B shape mismatch at layer %zu: "
              "file=%zux%zu, model=%zux%zu\n",
              i, (size_t)rows, (size_t)cols, m.b[i].rows, m.b[i].cols);
      ml_model_free(&m);
      fclose(fd);
      exit(1);
    }

    for (size_t r = 0; r < m.b[i].rows; ++r) {
      for (size_t c = 0; c < m.b[i].cols; ++c) {
        float x = 0.f;
        nread = fread(&x, sizeof(x), 1, fd);
        if (nread != 1) {
          fprintf(stderr, "ml_model_load: failed to read B data at layer %zu\n",
                  i);
          ml_model_free(&m);
          fclose(fd);
          exit(1);
        }

        MAT_AT(m.b[i], r, c) = x;
      }
    }
  }

  fclose(fd);
  return m;
}

Grad ml_grad_alloc(Mod m) {
  Grad g = {0};
  g.layer_count = m.layer_count;

  g.w = (Mat *)malloc(g.layer_count * sizeof(*g.w));
  g.b = (Mat *)malloc(g.layer_count * sizeof(*g.b));
  g.d = (Mat *)malloc(g.layer_count * sizeof(*g.d));

  assert(g.w != NULL);
  assert(g.b != NULL);
  assert(g.d != NULL);

  for (size_t i = 0; i < g.layer_count; ++i) {
    g.w[i] = ml_mat_alloc(m.w[i].rows, m.w[i].cols, m.w[i].stride);
    g.b[i] = ml_mat_alloc(m.b[i].rows, m.b[i].cols, m.b[i].stride);
    g.d[i] = ml_mat_alloc(m.a[i].rows, m.a[i].cols, m.a[i].stride);
  }

  return g;
}

void ml_grad_free(Grad *g) {
  if (g == NULL)
    return;

  for (size_t i = 0; i < g->layer_count; ++i) {
    ml_mat_free(&g->w[i]);
    ml_mat_free(&g->b[i]);
    ml_mat_free(&g->d[i]);
  }

  free(g->w);
  free(g->b);
  free(g->d);

  g->w = NULL;
  g->b = NULL;
  g->d = NULL;
  g->layer_count = 0;
}

void ml_grad_scale(Grad g, float scale) {
  assert(g.layer_count > 0);
  assert(g.w != NULL);
  assert(g.b != NULL);
  assert(g.d != NULL);

  for (size_t i = 0; i < g.layer_count; ++i) {
    ml_mat_scale(g.w[i], scale);
    ml_mat_scale(g.b[i], scale);
    ml_mat_scale(g.d[i], scale);
  }
}

void ml_grad_zero(Grad g) {
  assert(g.layer_count > 0);
  assert(g.w != NULL);
  assert(g.b != NULL);
  assert(g.d != NULL);

  for (size_t i = 0; i < g.layer_count; ++i) {
    ml_mat_zero(g.w[i]);
    ml_mat_zero(g.b[i]);
    ml_mat_zero(g.d[i]);
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
#endif // ML_IMPLEMENTATION
