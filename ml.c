#define ML_IMPLEMENTATION
#include "ml.h"
#include <time.h>

float train_data[] = {
    1, 1, 1, 0, 
    0, 1, 1, 1, 
    1, 0, 0, 0, 
    0, 0, 0, 1,
};

int main(void) {
  srand(time(0));

  Mat td = {
      .rows = 4,
      .cols = 4,
      .es = train_data,
  };

  size_t layer[] = {2, 4, 2, 6, 2};

  float rate = 1e-2f;
  float eps = 1e-3;
  size_t epoch = 1000 * 1000;

  Mod m = ml_model_alloc(layer, sizeof(layer) / sizeof(layer[0]));
  Grad g = ml_grad_alloc(m);

  ml_model_rand(m, -1.f, 1.f);

  printf("\033[?25l");

  for (size_t i = 0; i < epoch; ++i) {
    ml_model_finite_diff(m, g, td, eps);
    ml_model_train(m, g, rate);

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

      printf("] %3d%%  cost = %f", (int)(progress * 100.f),
             ml_model_cost(m, td, sigmoidf));

      fflush(stdout);
    }
  }

  printf("\033[?25h");
  printf("\n");

  ml_model_verify(m, td, sigmoidf);

  ml_grad_free(&g);
  ml_model_free(&m);

  return 0;
}
