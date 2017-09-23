#include <assert.h>
#include <math.h>
#include <stdio.h>

float dot(long n, float a[n], float b[n]) {
  float s = 0.0;
#pragma omp simd reduction(+:s)
  for (long i = 0; i < n; i++) {
    s += a[i] * b[i];
  }
  return s;
}

int max_length(long bn, long M, int begins[bn], int t[M]) {
  int ml = 0;
  for (long i = 0; i < M; i++) {
    int it = t[i];
    assert(it >= 0);
    assert(it + 1 < bn);
    int end_begin = begins[it + 1] - begins[it];
    ml = (ml < end_begin ? end_begin : ml);
  }
  return ml;
}

float forward(long M, long N, long P, long bn, long pn, long ml, 
                float x[M][N], int t[M], float W[P][N],
                int begins[bn], int paths[pn], float codes[pn], 
                float wxy[M][ml]) {
  float loss = 0.0;
  for (long i = 0; i < M; i++) {
    float l = 0.0;
    int it = t[i];
    assert(it >= 0);
    assert(it + 1 < bn);
    int begin = begins[it];
    int end = begins[it + 1];
    assert(begin <= end);
    for (long o = 0, k = begin; k < end; o++, k++) {
      assert(k >= 0);
      assert(k < pn);
      int p = paths[k];
      assert(p >= 0);
      assert(p < P);
      double wx = 0.0;
      asm volatile("### loop begins forward ");
      for (long j = 0; j < N; j++) {
	wx += W[p][j] * x[i][j];
      }
      asm volatile("### loop ends forward");
      // wxy[i][o] = wx * codes[k];
      // l += logf(1.0 + expf(-wx * codes[k]));
      wxy[i][o] = 1.0 + expf(-wx * codes[k]);
      l += logf(wxy[i][o]);
    }
    loss += l;
  }
  return loss;
}

float forward_(long M, long N, long P, long bn, long pn, long ml, 
               float x[M][N], int t[M], float W[P][N],
               int begins[bn], int paths[pn], float codes[pn], 
               float wxy[M][ml]) {
  float loss = 0.0;
  for (long io = 0; io < M * ml; io++) {
    long i = io / ml;
    long o = io % ml;
    int it = t[i];
    assert(it >= 0);
    assert(it + 1 < bn);
    int begin = begins[it];
    int end = begins[it + 1];
    int k = begin + o;
    if (k < end) {
      assert(k >= 0);
      assert(k < pn);
      int p = paths[k];
      assert(p >= 0);
      assert(p < P);
      double wx = 0.0;
      for (long j = 0; j < N; j++) {
	wx += W[p][j] * x[i][j];
      }
      wxy[i][o] = wx * codes[k];
      loss += logf(1.0 + expf(-wxy[i][o]));
    }
  }
  return loss;
}

void check_scal_error(float x, float c) {
  float eps = 1e-4;
  float tol = 1e-4;
  float o = (c >= 0 ? eps : -eps);
  float err = fabs((x - c) / (c + o));
  (void)err;
  (void)tol;
  assert(err < tol);
}


int x_backward(long M, long N, long P, long bn, long pn, long ml, float gl,
               float x[restrict M][N], int t[restrict M], float W[restrict P][N],
               float gx[restrict M][N], float gW[restrict P][N], 
               int begins[restrict bn], int paths[restrict pn],
               float codes[restrict pn], float __wxy__[restrict M][ml]) {
  for (long i = 0; i < M; i++) {
    int it = t[i];
    int begin = begins[it];
    int end = begins[it + 1];
    for (int k = begin, o = 0; k < end; k++, o++) {
      int p = paths[k];
      float wx = 0.0;
#pragma omp simd reduction(+:wx)
      for (int j = 0; j < N; j++) {
        wx += W[p][j] * x[i][j];
      }
      float wxy_ = wx * codes[k];
      float wxy = __wxy__[i][o];
      check_scal_error(wxy, wxy_);
      float g = -gl * codes[k] / (1.0 + expf(wxy));
      for (int j = 0; j < N; j++) {
        gW[p][j] += g * x[i][j]; /* !! */
        gx[i][j] += g * W[p][j];
      }
    }
  }
  return 0;
}

int backward(long M, long N, long P, long bn, long pn, long ml, float gl,
             float x[restrict M][N], int t[restrict M], float W[restrict P][N],
             float gx[restrict M][N], float gW[restrict P][N], 
             int begins[restrict bn], int paths[restrict pn],
             float codes[restrict pn], float wxy[restrict M][ml]) {
  for (long i = 0; i < M; i++) {
    int it = t[i];
    int begin = begins[it];
    int end = begins[it + 1];
    for (int k = begin, o = 0; k < end; k++, o++) {
      int p = paths[k];
      float g = -gl * codes[k] / wxy[i][o]; // (1.0 + expf(wxy[i][o]));
      for (int j = 0; j < N; j++) {
        gW[p][j] += g * x[i][j];
        gx[i][j] += g * W[p][j];
      }
    }
  }
  return 0;
}

