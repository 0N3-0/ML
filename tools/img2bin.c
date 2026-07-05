#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_png(const char *name) {
  const char *dot = strrchr(name, '.');
  return dot && strcmp(dot, ".png") == 0;
}

static size_t count_images(const char *img_dir, int num_classes, size_t *out_per_class,
                           int *out_w, int *out_h, int max_per_class) {
  size_t total = 0;
  int w = 0, h = 0;

  for (int c = 0; c < num_classes; c++) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%d", img_dir, c);

    DIR *d = opendir(path);
    if (!d) {
      fprintf(stderr, "ERROR: cannot open directory '%s'\n", path);
      exit(1);
    }

    struct dirent *ent;
    int n = 0;
    while ((ent = readdir(d)) != NULL) {
      if (!is_png(ent->d_name)) continue;
      if (max_per_class > 0 && n >= max_per_class) break;

      n++;
      out_per_class[c]++;
      total++;

      if (w == 0) {
        char fpath[2048];
        snprintf(fpath, sizeof(fpath), "%s/%d/%s", img_dir, c, ent->d_name);
        int ch;
        unsigned char *px = stbi_load(fpath, &w, &h, &ch, 1);
        if (!px) {
          fprintf(stderr, "ERROR: cannot load '%s': %s\n", fpath, stbi_failure_reason());
          closedir(d);
          exit(1);
        }
        stbi_image_free(px);
      }
    }
    closedir(d);
  }

  *out_w = w;
  *out_h = h;
  return total;
}

int main(int argc, char **argv) {
  int max_per_class = 0;

  if (argc >= 2 && strcmp(argv[1], "-f") == 0) {
    if (argc != 6) {
      fprintf(stderr, "Usage: %s -f <image.png> <class> <output|- > <num_classes>\n\n", argv[0]);
      fprintf(stderr, "Convert a single PNG image to MLDB binary format.\n\n");
      fprintf(stderr, "  image.png    path to input image (grayscale or RGB)\n");
      fprintf(stderr, "  class        integer class label (0..num_classes-1)\n");
      fprintf(stderr, "  output       output file path, or '-' to write to stdout\n");
      fprintf(stderr, "  num_classes  total number of output classes (for one-hot encoding)\n\n");
      fprintf(stderr, "Example:\n");
      fprintf(stderr, "  %s -f digit.png 4 - 10     # class 4, 10 classes, to stdout\n", argv[0]);
      fprintf(stderr, "  %s -f digit.png 4 out.bin 10  # write to file\n", argv[0]);
      return 1;
    }

    const char *img_path = argv[2];
    int single_class = atoi(argv[3]);
    const char *output_path = argv[4];
    int num_classes = atoi(argv[5]);

    if (num_classes < 2 || num_classes > 256) {
      fprintf(stderr, "ERROR: num_classes must be 2..256, got %d\n", num_classes);
      return 1;
    }
    if (single_class >= num_classes) {
      fprintf(stderr, "ERROR: class %d >= num_classes %d\n", single_class, num_classes);
      return 1;
    }

    int w, h, ch;
    unsigned char *px = stbi_load(img_path, &w, &h, &ch, 1);
    if (!px) {
      fprintf(stderr, "ERROR: cannot load '%s': %s\n", img_path, stbi_failure_reason());
      return 1;
    }

    size_t input_size = (size_t)(w * h);
    size_t cols = input_size + (size_t)num_classes;
    int use_stdout = strcmp(output_path, "-") == 0;
    FILE *out = use_stdout ? stdout : fopen(output_path, "wb");
    if (!out) {
      fprintf(stderr, "ERROR: cannot create '%s'\n", output_path);
      stbi_image_free(px);
      return 1;
    }

    fwrite("MLDB", 1, 4, out);
    uint64_t rows64 = 1;
    uint64_t cols64 = (uint64_t)cols;
    fwrite(&rows64, sizeof(rows64), 1, out);
    fwrite(&cols64, sizeof(cols64), 1, out);

    float *row = (float *)malloc(cols * sizeof(float));
    if (!row) {
      fprintf(stderr, "ERROR: out of memory\n");
      stbi_image_free(px);
      if (!use_stdout) fclose(out);
      return 1;
    }

    for (int i = 0; i < w * h; i++)
      row[i] = (float)px[i] / 255.0f;
    for (int i = 0; i < num_classes; i++)
      row[input_size + i] = 0.0f;
    row[input_size + single_class] = 1.0f;

    fwrite(row, sizeof(float), cols, out);
    free(row);
    stbi_image_free(px);

    if (!use_stdout) {
      fclose(out);
      printf("Wrote %dx%d image, class=%d, to %s\n", w, h, single_class, output_path);
    }
    return 0;
  }

  if (argc > 1 && strcmp(argv[1], "--max") == 0) {
    if (argc < 3) {
      fprintf(stderr, "ERROR: --max requires an argument\n");
      return 1;
    }
    max_per_class = atoi(argv[2]);
    argc -= 2;
    argv += 2;
  }

  if (argc != 4) {
    fprintf(stderr, "Usage: %s [--max N] <img_dir> <output|- > <num_classes>\n", argv[0]);
    fprintf(stderr, "       %s -f <image.png> <class> <output|- > <num_classes>\n\n", argv[0]);
    fprintf(stderr, "Convert labeled images in class subdirectories to MLDB binary format.\n");
    fprintf(stderr, "Expects img_dir/0/*.png, img_dir/1/*.png, ... up to num_classes-1.\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --max N      load at most N images per class (default: unlimited)\n");
    fprintf(stderr, "  -f           convert a single image instead of a directory\n\n");
    fprintf(stderr, "Arguments:\n");
    fprintf(stderr, "  img_dir      root directory with class subdirectories (0/, 1/, ...)\n");
    fprintf(stderr, "  output       output file path, or '-' to write to stdout\n");
    fprintf(stderr, "  num_classes  total number of output classes (for one-hot encoding)\n\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s --max 100 train train_1k.bin 10   # 100 images/class, 10 classes\n", argv[0]);
    fprintf(stderr, "  %s test - 10                          # all images to stdout\n", argv[0]);
    return 1;
  }

  const char *img_dir = argv[1];
  const char *output_path = argv[2];
  int num_classes = atoi(argv[3]);

  if (num_classes < 2 || num_classes > 256) {
    fprintf(stderr, "ERROR: num_classes must be 2..256, got %d\n", num_classes);
    return 1;
  }

  int use_stdout = strcmp(output_path, "-") == 0;

  size_t per_class[256] = {0};
  int img_w = 0, img_h = 0;

  printf("Scanning %s ...\n", img_dir);
  size_t total = count_images(img_dir, num_classes, per_class, &img_w, &img_h, max_per_class);

  printf("Image size: %dx%d\n", img_w, img_h);
  for (int i = 0; i < num_classes; i++) {
    printf("  class %d: %zu\n", i, per_class[i]);
  }
  printf("Total: %zu images\n", total);

  if (total == 0) {
    fprintf(stderr, "ERROR: no images found\n");
    return 1;
  }

  size_t input_size = (size_t)(img_w * img_h);
  size_t cols = input_size + (size_t)num_classes;

  FILE *out = use_stdout ? stdout : fopen(output_path, "wb");
  if (!out) {
    fprintf(stderr, "ERROR: cannot create '%s'\n", output_path);
    return 1;
  }

  fwrite("MLDB", 1, 4, out);
  uint64_t rows64 = (uint64_t)total;
  uint64_t cols64 = (uint64_t)cols;
  fwrite(&rows64, sizeof(rows64), 1, out);
  fwrite(&cols64, sizeof(cols64), 1, out);

  printf("Writing %zu rows x %zu cols (%zu input + %d output) ...\n", total, cols, input_size,
         num_classes);

  float *row = (float *)malloc(cols * sizeof(float));
  if (!row) {
    fprintf(stderr, "ERROR: out of memory\n");
    if (!use_stdout) fclose(out);
    return 1;
  }

  size_t count = 0;

  for (int c = 0; c < num_classes; c++) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%d", img_dir, c);

    DIR *d = opendir(path);
    if (!d) {
      fprintf(stderr, "ERROR: cannot open '%s'\n", path);
      free(row);
      if (!use_stdout) fclose(out);
      return 1;
    }

    struct dirent *ent;
    int loaded = 0;
    while ((ent = readdir(d)) != NULL) {
      if (!is_png(ent->d_name)) continue;
      if (max_per_class > 0 && loaded >= max_per_class) break;

      char fpath[2048];
      snprintf(fpath, sizeof(fpath), "%s/%d/%s", img_dir, c, ent->d_name);

      int w, h, ch;
      unsigned char *px = stbi_load(fpath, &w, &h, &ch, 1);
      if (!px) {
        fprintf(stderr, "ERROR: %s: %s\n", fpath, stbi_failure_reason());
        closedir(d);
        free(row);
        if (!use_stdout) fclose(out);
        return 1;
      }

      if (w != img_w || h != img_h) {
        fprintf(stderr, "ERROR: %s is %dx%d, expected %dx%d\n", fpath, w, h, img_w, img_h);
        stbi_image_free(px);
        closedir(d);
        free(row);
        if (!use_stdout) fclose(out);
        return 1;
      }

      for (int i = 0; i < w * h; i++) {
        row[i] = (float)px[i] / 255.0f;
      }

      for (int i = 0; i < num_classes; i++) {
        row[input_size + i] = 0.0f;
      }
      row[input_size + c] = 1.0f;

      if (fwrite(row, sizeof(float), cols, out) != cols) {
        fprintf(stderr, "ERROR: write failed\n");
        stbi_image_free(px);
        closedir(d);
        free(row);
        if (!use_stdout) fclose(out);
        return 1;
      }

      stbi_image_free(px);
      count++;
      loaded++;

      if (count % 1000 == 0 || count == total) {
        fprintf(stderr, "\r  %zu/%zu (%.1f%%)", count, total, 100.0f * (float)count / (float)total);
        fflush(stderr);
      }
    }
    closedir(d);
  }

  fprintf(stderr, "\n");
  free(row);
  if (!use_stdout) fclose(out);

  if (use_stdout)
    fprintf(stderr, "Done: %zu rows to stdout\n", total);
  else
    printf("Done: %s (%.1f MB)\n", output_path,
           (float)(4 + 8 + 8 + total * cols * sizeof(float)) / (1024.0f * 1024.0f));

  return 0;
}
