#include <assert.h>
#include <math.h>

int update(long M, long N, float beta1, float beta2, float lr, float eps,
           float grad[restrict M][N], float m[restrict M][N], float v[restrict M][N], float data[restrict M][N]) {
  for (long i = 0; i < M; i++) {
    for (long j = 0; j < N; j++) {
      m[i][j] += (1 - beta1) * (grad[i][j] - m[i][j]);
      v[i][j] += (1 - beta2) * (grad[i][j] * grad[i][j] - v[i][j]);
      data[i][j] -= lr * m[i][j] / (sqrtf(v[i][j]) + eps);
    }
  }
  return 0;
}

int update_(long M, long N, float beta1, float beta2, float lr, float eps,
           float grad[restrict M * N], float m[restrict M * N], float v[restrict M * N], float data[restrict M * N]) {
  for (long i = 0; i < M * N; i++) {
    m[i] += (1 - beta1) * (grad[i] - m[i]);
    v[i] += (1 - beta2) * (grad[i] * grad[i] - v[i]);
    data[i] -= lr * m[i] / (sqrtf(v[i]) + eps);
  }
  return 0;
}

