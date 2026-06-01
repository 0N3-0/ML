#define ML_IMPLEMENTATION
#include "ml.h"
#include <time.h>

int main(void) {

  srand(time(0));

  Mat td = ml_load_td("train_data.csv");
  ml_mat_shuffle(td);
  size_t layer[] = {2, 28, 28, 1};
  Act acts[] = {ML_SIGMOID, ML_SIGMOID, ML_SIGMOID};
  size_t batch_count = 28;
  size_t batch_size = td.rows / batch_count;

  float rate = 0.5f;
  size_t epoch = 10 * 1000;

  Mod m =
      ml_model_alloc(layer, sizeof(layer) / sizeof(layer[0]), batch_size, acts);
  Grad g = ml_grad_alloc(m);

  ml_model_rand(m, -1.f, 1.f);

  printf("\033[?25l");

  for (size_t i = 0; i < epoch; ++i) {
    float c = 0.f;
    for (size_t j = 0; j < batch_count; ++j) {
      Mat batch = ml_mat_slice(td, batch_size * (j % batch_count), 0,
                               batch_size, td.cols);
      ml_model_backprop(m, g, batch);
      ml_model_train(m, g, rate);
      c += ml_model_cost(m, batch);
    }
    if (i % 100 == 0 || i == epoch - 1) {
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

      printf("] %3d%%  cost = %f", (int)(progress * 100.f), c / batch_size);

      fflush(stdout);
    }
  }

  printf("\033[?25h");
  printf("\n");

  ml_model_save(m, "model.bin");

  ml_grad_free(&g);
  ml_model_free(&m);

  return 0;
}
