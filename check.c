#define ML_IMPLEMENTATION
#include "ml.h"

int main(){
  TrainConfig train_config = {
    .batch_size = 1,
    .loss = {
      .lossf = ml_model_loss_softmax_cce,
      .dlossf = ml_model_loss_dsoftmax_cce,
    },
  };
  Act acts[] = {ML_LRELU,ML_LRELU,ML_LINEAR};
  Mod mod = ml_model_load("model.bin", train_config, acts);
  if (!mod.layer) {
    LOG_ERROR("failed to load model");
    return 1;
  }

  Mat data = ml_data_load_bin("check.bin");
  if (!data.es) {
    LOG_ERROR("failed to load data");
    return 1;
  }

  Mat in = ml_mat_slice(data, 0, 0, data.rows, mod.layer[0].w.rows);
  ml_model_forward(mod, in);

  printf("logits:\n");
  MAT_PRINT(mod.layer[mod.layer_count - 1].a);
  printf("predicted class: %zu\n", ml_label_argmax(&MAT_AT(mod.layer[mod.layer_count - 1].a, 0, 0), mod.layer[mod.layer_count - 1].a.cols));

  ml_mat_free(&data);
  ml_model_free(&mod);
}
