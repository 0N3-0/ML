#ifndef ML_H_
#define ML_H_

#define MAT_AT(m, i, j) (m).es[(i) * (m).cols + (j)]
#define MAT_PRINT(m) ml_mat_print(m, #m)

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  size_t rows;
  size_t cols;
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

Mat ml_mat_alloc(size_t rows, size_t cols);
void ml_mat_free(Mat *m);
void ml_mat_print(Mat m, const char *name);
void ml_mat_rand(Mat m, float low, float high);
void ml_mat_sum(Mat dst, Mat src);
void ml_mat_mul(Mat dst, Mat a, Mat b);
void ml_mat_scale(Mat m, float scale);
void ml_mat_zero(Mat m);

Mod ml_model_alloc(size_t *layer_sizes, size_t layer_count);
void ml_model_free(Mod *m);
void ml_model_rand(Mod m, float low, float high);
void ml_model_print(Mod m);
void ml_model_forward(Mod m, Mat in, float (*activate)(float));
float ml_model_cost(Mod m, Mat td, float (*activate)(float));
void ml_model_finite_diff(Mod m, Grad g, Mat td, float eps);
void ml_model_backprop(Mod m, Grad g, Mat td, float (*dactivate)(float),
                              float (*activate)(float));
void ml_model_train(Mod m, Grad g, float rate);
void ml_model_verify(Mod m, Mat td, float (*activate)(float));

Grad ml_grad_alloc(Mod m);
void ml_grad_free(Grad *g);
void ml_grad_scale(Grad g, float scale);
void ml_grad_zero(Grad g);

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

Mat ml_mat_alloc(size_t rows, size_t cols) {
  assert(rows > 0);
  assert(cols > 0);

  Mat m = {
      .rows = rows,
      .cols = cols,
      .es = (float *)malloc(rows * cols * sizeof(*m.es)),
  };

  assert(m.es != NULL);
  return m;
}

void ml_mat_free(Mat *m) {
  if (m == NULL)
    return;

  free(m->es);
  m->es = NULL;
  m->rows = 0;
  m->cols = 0;
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

void ml_mat_scale(Mat m, float scale) {
  assert(m.rows > 0);
  assert(m.cols > 0);
  assert(m.es != NULL);

  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) *= scale;
    }
  }
}

void ml_mat_zero(Mat m) {
  assert(m.rows > 0);
  assert(m.cols > 0);
  assert(m.es != NULL);

  memset(m.es, 0, sizeof(*m.es) * m.rows * m.cols);
}

Mod ml_model_alloc(size_t *layer_sizes, size_t layer_count) {
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
    m.z[i] = ml_mat_alloc(1, layer_sizes[i + 1]);
    m.w[i] = ml_mat_alloc(layer_sizes[i], layer_sizes[i + 1]);
    m.a[i] = ml_mat_alloc(1, layer_sizes[i + 1]);
    m.b[i] = ml_mat_alloc(1, layer_sizes[i + 1]);
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
    ml_mat_sum(m.z[i], m.b[i]);
    ml_act_func(m.a[i], m.z[i], activate);
  }
}

float ml_model_cost(Mod m, Mat td, float (*activate)(float)) {
  float res = 0.f;

  size_t input_size = m.w[0].rows;
  size_t output_size = m.a[m.layer_count - 1].cols;

  assert(input_size + output_size == td.cols);

  Mat in = {.rows = 1, .cols = input_size, .es = NULL};

  for (size_t i = 0; i < td.rows; ++i) {
    in.es = &MAT_AT(td, i, 0);
    ml_model_forward(m, in, activate);
    Mat out = m.a[m.layer_count - 1];

    for (size_t j = 0; j < output_size; ++j) {
      float d = 0.f;
      d += MAT_AT(td, i, j + input_size) - MAT_AT(out, 0, j);
      res += d * d;
    }
  }
  return res / td.rows / output_size;
}

void ml_model_finite_diff(Mod m, Grad g, Mat td, float eps) {
  assert(eps > 0);
  float c = ml_model_cost(m, td , sigmoidf);

  for (size_t i = 0; i < m.layer_count; ++i) {
    assert(m.w[i].rows == g.w[i].rows);
    assert(m.w[i].cols == g.w[i].cols);
    for (size_t j = 0; j < m.w[i].rows; ++j) {
      for (size_t k = 0; k < m.w[i].cols; ++k) {
        float save = MAT_AT(m.w[i], j, k);
        MAT_AT(m.w[i], j, k) += eps;
        MAT_AT(g.w[i], j, k) = (ml_model_cost(m, td ,sigmoidf) - c) / eps;
        MAT_AT(m.w[i], j, k) = save;
      }
    }
    assert(m.b[i].rows == g.b[i].rows);
    assert(m.b[i].cols == g.b[i].cols);
    for (size_t j = 0; j < m.b[i].rows; ++j) {
      for (size_t k = 0; k < m.b[i].cols; ++k) {
        float save = MAT_AT(m.b[i], j, k);
        MAT_AT(m.b[i], j, k) += eps;
        MAT_AT(g.b[i], j, k) = (ml_model_cost(m, td,sigmoidf) - c) / eps;
        MAT_AT(m.b[i], j, k) = save;
      }
    }
  }
}

void ml_model_backprop(Mod m, Grad g, Mat td, float (*dactivate)(float),
                              float (*activate)(float)) {
  assert(m.layer_count > 0);
  ml_grad_zero(g);

  size_t input_size = m.w[0].rows;
  size_t output_size = m.a[m.layer_count - 1].cols;
  assert(td.cols == input_size + output_size);

  Mat in = {
      .rows = 1,
      .cols = input_size,
      .es = NULL,
  };

  for (size_t i = 0; i < td.rows; ++i) {
    in.es = &MAT_AT(td, i, 0);
    ml_model_forward(m, in, activate);

    for (size_t j = 0; j < m.a[m.layer_count - 1].cols; ++j) {
      MAT_AT(g.d[m.layer_count - 1], 0, j) =
          2 *
          (MAT_AT(m.a[m.layer_count - 1], 0, j) -
           MAT_AT(td, i, input_size + j)) *
          dactivate(MAT_AT(m.z[m.layer_count - 1], 0, j));
    }

    for (size_t l = m.layer_count - 1; l-- > 0;) {

      for (size_t j = 0; j < m.w[l].cols; ++j) {
        float sum = 0.f;
        for (size_t k = 0; k < m.w[l + 1].cols; ++k) {
          sum += MAT_AT(g.d[l + 1], 0, k) * MAT_AT(m.w[l + 1], j, k);
        }
        MAT_AT(g.d[l], 0, j) = sum * dactivate(MAT_AT(m.z[l], 0, j));
      }
    }

    for (size_t l = m.layer_count; l-- > 0;) {
      Mat prev = l == 0 ? in : m.a[l - 1];
      for (size_t j = 0; j < m.w[l].cols; ++j) {
        for (size_t k = 0; k < m.w[l].rows; ++k) {
          MAT_AT(g.w[l], k, j) += MAT_AT(prev, 0, k) * MAT_AT(g.d[l], 0, j);
        }
        MAT_AT(g.b[l], 0, j) += MAT_AT(g.d[l], 0, j);
      }
    }
  }

  ml_grad_scale(g, 1.f / (td.rows * output_size));
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

void ml_model_verify(Mod m, Mat td, float (*activate)(float)) {
  assert(m.layer_count > 0);

  size_t input_size = m.w[0].rows;
  size_t output_size = m.a[m.layer_count - 1].cols;

  assert(td.cols == input_size + output_size);

  Mat in = {
      .rows = 1,
      .cols = input_size,
      .es = NULL,
  };

  for (size_t i = 0; i < td.rows; ++i) {
    in.es = &MAT_AT(td, i, 0);

    ml_model_forward(m, in, activate);

    Mat out = m.a[m.layer_count - 1];

    printf("input: [");
    for (size_t j = 0; j < input_size; ++j) {
      printf(" %f", MAT_AT(td, i, j));
    }
    printf(" ] ");

    printf("output: [");
    for (size_t j = 0; j < output_size; ++j) {
      printf(" %f", MAT_AT(out, 0, j));
    }
    printf(" ] ");

    printf("expected: [");
    for (size_t j = 0; j < output_size; ++j) {
      printf(" %f", MAT_AT(td, i, input_size + j));
    }
    printf(" ]\n");
  }
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
    g.w[i] = ml_mat_alloc(m.w[i].rows, m.w[i].cols);
    g.b[i] = ml_mat_alloc(m.b[i].rows, m.b[i].cols);
    g.d[i] = ml_mat_alloc(m.a[i].rows, m.a[i].cols);
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
#endif // ML_IMPLEMENTATION
