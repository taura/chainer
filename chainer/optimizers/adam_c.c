#include <assert.h>
#include <math.h>

int update_inner(long M, long N, float beta1, float beta2, float lr, float eps,
		 long i, float grad[restrict M][N], float m[restrict M][N], float v[restrict M][N], float data[restrict M][N]) {
  asm volatile("### update_inner loop begin");
#pragma omp simd
  for (long j = 0; j < N; j++) {
    m[i][j] += (1 - beta1) * (grad[i][j] - m[i][j]);
    v[i][j] += (1 - beta2) * (grad[i][j] * grad[i][j] - v[i][j]);
    data[i][j] -= lr * m[i][j] / (sqrtf(v[i][j]) + eps);
  }
  asm volatile("### update_inner loop end");
  return 0;
}

int update(long M, long N, float beta1, float beta2, float lr, float eps,
	   float grad[restrict M][N], float m[restrict M][N], float v[restrict M][N], float data[restrict M][N]) {
#pragma omp parallel for schedule(runtime) 
  for (long i = 0; i < M; i++) {
    update_inner(M, N, beta1, beta2, lr, eps, i, grad, m, v, data);
  }
  return 0;
}

#if 0
int update(long M, long N, float beta1, float beta2, float lr, float eps,
           float grad[restrict M][N], float m[restrict M][N], float v[restrict M][N], float data[restrict M][N]) {
#pragma omp parallel for 
  for (long i = 0; i < M; i++) {
    asm volatile("### update loop begin");
#pragma omp simd
    for (long j = 0; j < N; j++) {
      m[i][j] += (1 - beta1) * (grad[i][j] - m[i][j]);
      v[i][j] += (1 - beta2) * (grad[i][j] * grad[i][j] - v[i][j]);
      data[i][j] -= lr * m[i][j] / (sqrtf(v[i][j]) + eps);
    }
    asm volatile("### update loop end");
  }
  return 0;
}

int x_update(long M, long N, float beta1, float beta2, float lr, float eps,
	     float grad[restrict M * N], float m[restrict M * N], float v[restrict M * N], float data[restrict M * N]) {
  for (long i = 0; i < M * N; i++) {
    m[i] += (1 - beta1) * (grad[i] - m[i]);
    v[i] += (1 - beta2) * (grad[i] * grad[i] - v[i]);
    data[i] -= lr * m[i] / (sqrtf(v[i]) + eps);
  }
  return 0;
}
#endif
