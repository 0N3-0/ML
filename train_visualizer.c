#define ML_IMPLEMENTATION
#include "ml.h"
#include <raylib.h>
#include <time.h>

#define WIDTH 1600
#define HEIGHT 1000
#define RENDER_SIZE 280

#define COST_CAP 10000

#define PANEL_Y 180
#define PANEL_H 560

#define GRAPH_X 60
#define GRAPH_Y PANEL_Y
#define GRAPH_W 700
#define GRAPH_H PANEL_H

#define MODEL_SCALE 2
#define MODEL_VIEW_SIZE (RENDER_SIZE * MODEL_SCALE)

#define MODEL_X 920
#define MODEL_Y PANEL_Y

Color pixels[RENDER_SIZE * RENDER_SIZE];

float cost_history[COST_CAP];
size_t cost_count = 0;

void cost_history_push(float cost) {
  if (cost_count < COST_CAP) {
    cost_history[cost_count++] = cost;
    return;
  }

  for (size_t i = 1; i < COST_CAP; ++i) {
    cost_history[i - 1] = cost_history[i];
  }

  cost_history[COST_CAP - 1] = cost;
}

void draw_cost_graph(float *history, size_t count, int x, int y, int w, int h) {
  DrawText("cost curve", x, y - 42, 28, WHITE);
  DrawRectangleLines(x, y, w, h, DARKGRAY);

  if (count < 2) {
    return;
  }

  float first_cost = history[0];

  float range = first_cost;
  if (range < 1e-6f) {
    range = 1.f;
  }

  int px = x;
  int py = y;
  int pw = w;
  int ph = h;

  bool need_compress = count > (size_t)pw;

  for (size_t i = 1; i < count; ++i) {
    float x0f;
    float x1f;

    if (need_compress) {
      x0f = (float)(i - 1) / (float)(count - 1) * (float)pw;
      x1f = (float)i / (float)(count - 1) * (float)pw;
    } else {
      x0f = (float)(i - 1);
      x1f = (float)i;
    }

    float v0 = (first_cost - history[i - 1]) / range;
    float v1 = (first_cost - history[i]) / range;

    if (v0 < 0.f)
      v0 = 0.f;
    if (v0 > 1.f)
      v0 = 1.f;

    if (v1 < 0.f)
      v1 = 0.f;
    if (v1 > 1.f)
      v1 = 1.f;

    int x0 = px + (int)x0f;
    int x1 = px + (int)x1f;

    int y0 = py + (int)(v0 * (float)ph);
    int y1 = py + (int)(v1 * (float)ph);

    DrawLine(x0, y0, x1, y1, GREEN);
  }
}

void render_model(Mod m, Mat in, Color *pixels, int render_size) {
  for (int y = 0; y < render_size; ++y) {
    for (int x = 0; x < render_size; ++x) {
      pixels[y * render_size + x] = BLACK;
    }
  }

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

      pixels[j * render_size + i] = (Color){c, c, c, 255};
    }
  }
}

int main(void) {
  srand(time(0));

  Mat td = ml_load_td("train_data.csv");
  if (td.es == NULL) {
    LOG_ERROR("failed to load training data");
    return 1;
  }

  size_t layer[] = {2, 28, 28, 1};
  Act acts[] = {ML_GELU, ML_SWISH, ML_SIGMOID};
  size_t batch_count = 28;
  TrainConfig train_config = {
    .rate = 1.f,
    .batch_size = td.rows / batch_count,
    .loss = {
      .lossf = ml_model_loss_mse,
      .dlossf = ml_model_loss_dmse,
    }
  };

  TrainConfig train_config_show = {
    .rate = 1.f,
    .batch_size = 1,
    .loss = {
      .lossf = ml_model_loss_mse,
      .dlossf = ml_model_loss_dmse,
    }
  };

  Mod train_m =
    ml_model_alloc(layer, sizeof(layer) / sizeof(layer[0]), train_config, acts);
  if (train_m.layer == NULL) {
    LOG_ERROR("failed to allocate train model");
    ml_mat_free(&td);
    return 1;
  }

  Grad g = ml_grad_alloc(train_m);
  if (g.layer == NULL) {
    LOG_ERROR("failed to allocate gradients");
    ml_model_free(&train_m);
    ml_mat_free(&td);
    return 1;
  }

  Mod show_m = ml_model_alloc(
    layer,
    sizeof(layer) / sizeof(layer[0]),
    train_config_show,
    acts
  );
  if (show_m.layer == NULL) {
    LOG_ERROR("failed to allocate show model");
    ml_grad_free(&g);
    ml_model_free(&train_m);
    ml_mat_free(&td);
    return 1;
  }

  Mat render_in = ml_mat_alloc(1, 2, 2);
  if (render_in.es == NULL) {
    LOG_ERROR("failed to allocate render input matrix");
    ml_model_free(&show_m);
    ml_grad_free(&g);
    ml_model_free(&train_m);
    ml_mat_free(&td);
    return 1;
  }

  if (ml_model_xavier_rand(train_m) != 0) {
    LOG_ERROR("failed to initialize model weights");
    ml_mat_free(&render_in);
    ml_model_free(&show_m);
    ml_grad_free(&g);
    ml_model_free(&train_m);
    ml_mat_free(&td);
    return 1;
  }

  ml_model_copy_params(show_m, train_m);

  InitWindow(WIDTH, HEIGHT, "ML Training Visualizer");
  SetTargetFPS(60);

  Image image = GenImageColor(RENDER_SIZE, RENDER_SIZE, BLACK);
  Texture texture = LoadTextureFromImage(image);
  UnloadImage(image);

  size_t epoch = 0;
  size_t steps_per_frame = 4;

  bool window_open = true;

  while (!WindowShouldClose()) {
    ml_mat_shuffle(td);
    float c = 0.f;

    for (size_t s = 0; s < steps_per_frame; ++s) {

      for (size_t j = 0; j < batch_count; ++j) {
        Mat batch = ml_mat_slice(
          td,
          j * train_config.batch_size,
          0,
          train_config.batch_size,
          td.cols
        );

        if (ml_model_backprop(train_m, g, batch, train_config) != 0) {
          LOG_ERROR("backprop failed at epoch %zu", epoch);
          goto cleanup;
        }
        if (ml_model_optimizer(train_m, g, train_config) != 0) {
          LOG_ERROR("optimizer failed at epoch %zu", epoch);
          goto cleanup;
        }

        c += ml_model_cost(train_m, batch, train_config);
      }

      epoch++;
    }

    float avg_cost = c / (float)(steps_per_frame * batch_count);
    cost_history_push(avg_cost);

    if (epoch % 8 == 0) {
      ml_model_copy_params(show_m, train_m);
      render_model(show_m, render_in, pixels, RENDER_SIZE);
      UpdateTexture(texture, pixels);
    }

    BeginDrawing();
    ClearBackground(BLACK);

    DrawText(TextFormat("epoch: %zu", epoch), GRAPH_X, 50, 26, WHITE);
    DrawText(TextFormat("cost: %.6f", avg_cost), GRAPH_X, 86, 26, WHITE);

    draw_cost_graph(
      cost_history,
      cost_count,
      GRAPH_X,
      GRAPH_Y,
      GRAPH_W,
      GRAPH_H
    );

    DrawText(
      "model output",
      MODEL_X + MODEL_VIEW_SIZE / 2 - 90,
      MODEL_Y - 42,
      28,
      WHITE
    );

    DrawTextureEx(
      texture,
      (Vector2){MODEL_X, MODEL_Y},
      0.f,
      (float)MODEL_SCALE,
      WHITE
    );

    DrawRectangleLines(
      MODEL_X,
      MODEL_Y,
      MODEL_VIEW_SIZE,
      MODEL_VIEW_SIZE,
      DARKGRAY
    );

    EndDrawing();
  }

  window_open = false;
  UnloadTexture(texture);
  CloseWindow();

  if (ml_model_save(train_m, "model.bin") != 0) {
    LOG_ERROR("failed to save model");
  }

cleanup:
  if (window_open) {
    UnloadTexture(texture);
    CloseWindow();
  }
  ml_grad_free(&g);
  ml_model_free(&train_m);
  ml_model_free(&show_m);
  ml_mat_free(&render_in);
  ml_mat_free(&td);

  return 0;
}
