#define ML_IMPLEMENTATION
#include "ml.h"
#include <raylib.h>
#include <time.h>

#define WIDTH 1600
#define HEIGHT 1000
#define RENDER_SIZE 280

Color pixels[WIDTH * HEIGHT];


void render_model(Mod m, Mat in, Color *pixels, int width, int height,
                  int render_size) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      pixels[y * width + x] = BLACK;
    }
  }

  int start_x = width / 2 - render_size / 2;
  int start_y = height / 2 - render_size / 2;

  for (int j = 0; j < render_size; ++j) {
    for (int i = 0; i < render_size; ++i) {
      float xn = (float)i / (float)(render_size - 1);
      float yn = (float)j / (float)(render_size - 1);

      MAT_AT(in, 0, 0) = xn;
      MAT_AT(in, 0, 1) = yn;

      ml_model_forward(m, in);

      float p = MAT_AT(m.layer[m.layer_count - 1].a, 0, 0);

      if (p < 0.f)
        p = 0.f;
      if (p > 1.f)
        p = 1.f;

      unsigned char c = (unsigned char)(p * 255.f);

      int x = start_x + i;
      int y = start_y + j;

      pixels[y * width + x] = (Color){c, c, c, 255};
    }
  }
}

int main(void) {
  srand(time(0));

  Mat td = ml_load_td("train_data.csv");

  size_t layer[] = {2, 14, 14, 14, 1};
  Act acts[] = {ML_RELU, ML_SIGMOID, ML_RELU, ML_SIGMOID};

  size_t batch_count = 28;
  size_t batch_size = td.rows / batch_count;

  float rate = 1.f;

  Mod train_m =
      ml_model_alloc(layer, sizeof(layer) / sizeof(layer[0]), batch_size, acts);
  Grad g = ml_grad_alloc(train_m);

  Mod show_m = ml_model_alloc(layer, sizeof(layer) / sizeof(layer[0]), 1 ,acts);
  Mat render_in = ml_mat_alloc(1, 2, 2);
  ml_model_rand(train_m, -1.f, 1.f);
  ml_model_copy_params(show_m, train_m);

  InitWindow(WIDTH, HEIGHT, "ML Training Visualizer");
  SetTargetFPS(60);

  Image image = GenImageColor(WIDTH, HEIGHT, BLACK);
  Texture texture = LoadTextureFromImage(image);
  UnloadImage(image);

  size_t epoch = 0;
  size_t steps_per_frame = 4;

  while (!WindowShouldClose()) {
    float c = 0.f;
    for (size_t s = 0; s < steps_per_frame; ++s) {
      ml_mat_shuffle(td);

      for (size_t j = 0; j < batch_count; ++j) {
        Mat batch = ml_mat_slice(td, j * batch_size, 0, batch_size, td.cols);

        ml_model_backprop(train_m, g, batch);
        ml_model_train(train_m, g, rate);
        c += ml_model_cost(train_m, batch);
      }

      epoch++;
    }

    ml_model_copy_params(show_m, train_m);
    render_model(show_m, render_in, pixels, WIDTH, HEIGHT, RENDER_SIZE);
    UpdateTexture(texture, pixels);

    BeginDrawing();
    ClearBackground(BLACK);

    float scale = 3.0f;
    Vector2 pos = {
        .x = (WIDTH - texture.width * scale) / 2.0f,
        .y = (HEIGHT - texture.height * scale) / 2.0f,
    };

    DrawTextureEx(texture, pos, 0.0f, scale, WHITE);
    float avg_cost = c / (float)(steps_per_frame * batch_count);
    DrawText(TextFormat("epoch: %zu", epoch), 20, 20, 24, WHITE);
    DrawText(TextFormat("cost: %.6f", avg_cost), 20, 50, 24, WHITE);

    EndDrawing();
  }

  UnloadTexture(texture);

  ml_model_save(train_m, "model.bin");

  ml_grad_free(&g);
  ml_model_free(&train_m);
  ml_model_free(&show_m);
  ml_mat_free(&td);

  CloseWindow();

  return 0;
}
