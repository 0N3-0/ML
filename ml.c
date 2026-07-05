#define ML_IMPLEMENTATION
#include "ml.h"
#include <time.h>

int main(void) {

  srand(time(0));

  Mat td = ml_data_load_bin("train_1k.bin");
  if (td.es == NULL) {
    LOG_ERROR("failed to load training data");
    return 1;
  }

  size_t layer[] = {784, 128, 64, 10};
  Act acts[] = {ML_LRELU, ML_LRELU, ML_LINEAR};
  size_t batch_count = 20;
  TrainConfig train_config = {
    .rate = 0.01f,
    .batch_size = td.rows / batch_count,
    .loss = {
      .lossf = ml_model_loss_softmax_cce,
      .dlossf = ml_model_loss_dsoftmax_cce,
    }
  };

  size_t epoch = 1000;

  Mod m = ml_model_alloc(layer, sizeof(layer) / sizeof(layer[0]), train_config, acts);
  if (m.layer == NULL) {
    LOG_ERROR("failed to allocate model");
    ml_mat_free(&td);
    return 1;
  }

  Grad g = ml_grad_alloc(m);
  if (g.layer == NULL) {
    LOG_ERROR("failed to allocate gradients");
    ml_model_free(&m);
    ml_mat_free(&td);
    return 1;
  }

  if (ml_model_xavier_rand(m) != 0) {
    LOG_ERROR("failed to initialize model weights");
    ml_grad_free(&g);
    ml_model_free(&m);
    ml_mat_free(&td);
    return 1;
  }

  printf("\033[?25l");

  for (size_t i = 0; i < epoch; ++i) {
    ml_mat_shuffle(td);
    float c = 0.f;
    for (size_t j = 0; j < batch_count; ++j) {
      Mat batch = ml_mat_slice(
        td,
        train_config.batch_size * (j % batch_count),
        0,
        train_config.batch_size,
        td.cols
      );
      if (ml_model_backprop(m, g, batch, train_config) != 0) {
        LOG_ERROR("backprop failed at epoch %zu, batch %zu", i, j);
        goto cleanup;
      }
      if (ml_model_optimizer(m, g, train_config) != 0) {
        LOG_ERROR("optimizer failed at epoch %zu, batch %zu", i, j);
        goto cleanup;
      }
      c += ml_model_cost(m, batch, train_config);
    }
    if (i % 10 == 0 || i == epoch - 1) {
      int bar_width = 50;
      float progress = (float)(i + 1) / (float)epoch;
      int pos = (int)(bar_width * progress);

      printf("\r[");

      for (int j = 0; j < bar_width; ++j) {
        if (j < pos) {
          putchar('=');
        } else if (j == pos && pos < bar_width) {
          putchar('>');
        } else {
          putchar(' ');
        }
      }

      printf("] %3d%%  cost = %f", (int)(progress * 100.f), c / batch_count);

      fflush(stdout);
    }
  }

  printf("\033[?25h");
  printf("\n");

  LOG_INFO("training complete: %zu epochs", epoch);

  if (ml_model_save(m, "model.bin") != 0) {
    LOG_ERROR("failed to save model");
  }

cleanup:
  ml_grad_free(&g);
  ml_model_free(&m);
  ml_mat_free(&td);

  return 0;
}
