#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <x86intrin.h>

int max_length(long bn, long M, int begins[bn], int t[M]) {
  int ml = 0;
#pragma omp parallel for schedule(runtime) reduction(max:ml)
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
	      float x[restrict M][N], int t[restrict M], float W[restrict P][N],
	      int begins[restrict bn], int paths[restrict pn], float codes[restrict pn], 
	      float wxy_cache[restrict M][ml]) {
  float loss = 0.0;
#pragma omp parallel for schedule(runtime) reduction(+:loss)
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
      asm volatile("### forward: vectorized loop begins");
#pragma omp simd
      for (long j = 0; j < N; j++) {
	wx += W[p][j] * x[i][j];
      }
      asm volatile("### forward: vectorized loop ends");
      float wxy = wx * codes[k];
      wxy_cache[i][o] = wxy;
      l += logf(1.0 + expf(-wxy));
    }
    loss += l;
  }
  return loss;
}

float x_forward(long M, long N, long P, long bn, long pn, long ml, 
		float x[restrict M][N], int t[restrict M], float W[restrict P][N],
		int begins[restrict bn], int paths[restrict pn], float codes[restrict pn], 
		float wxy[restrict M][ml]) {
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

#if 0
static void check_scal_error(float x, float c) {
  float eps = 1e-4;
  float tol = 1e-4;
  float o = (c >= 0 ? eps : -eps);
  float err = fabs((x - c) / (c + o));
  (void)err;
  (void)tol;
  assert(err < tol);
}
#endif

#define REC_BACKWARD 0
#if REC_BACKWARD
typedef long long tsc_t;

typedef struct {
  tsc_t t[6];
  long count_i;
  long count_p;
} backward_rec;
#endif

int backward_(long M, long N, long P, long bn, long pn, long ml, float gl,
	     float x[restrict M][N], int t[restrict M], float W[restrict P][N],
	     float gx[restrict M][N], float gW[restrict P][N], 
	     int begins[restrict bn], int paths[restrict pn],
	     float codes[restrict pn], float wxy_cache[restrict M][ml]) {
#pragma omp parallel for
  for (long i = 0; i < M; i++) {
    int it = t[i];
    assert(it >= 0);
    assert(it + 1 < bn);
    int begin = begins[it];
    int end = begins[it + 1];
    assert(end - begin >= 0);
    assert(end - begin <= ml);
    for (long k = begin, o = 0; k < end; k++, o++) {
      assert(k >= 0);
      assert(k < pn);
      long p = paths[k];
      assert(p >= 0);
      assert(p < pn);
      float wxy_c = wxy_cache[i][o];
#if 0
      float wx = dot(M, N, P, p, i, W, x);
      float wx = 0.0;
      for (long j = 0; j < N; j++) {
        wx += W[p][j] * x[i][j];
      }
      float wxy_org = wx * codes[k];
      check_scal_error(wxy_c, wxy_org);
#endif
      float g = -gl * codes[k] / (1.0 + expf(wxy_c));
      for (long j = 0; j < N; j++) {
#pragma omp atomic
        gW[p][j] += g * x[i][j];
        gx[i][j] += g * W[p][j];
      }
    }
  }
  return 0;
}

void calc_p_assignment(int nw, long M, long P, long bn, long pn,
		       int t[M], int begins[bn], int paths[pn],
		       int assign[P]) {
  /* count how many times each path appears in the paths array */
#pragma omp for schedule(runtime) 
  for (long p = 0; p < P; p++) {
    assign[p] = 0;
  }
#pragma omp for schedule(runtime) 
  for (long i = 0; i < M; i++) {
    int it = t[i];
    int begin = begins[it];
    int end = begins[it + 1];
    for (int k = begin; k < end; k++) {
      int p = paths[k];
      assert(0 <= p);
      assert(p < P);
#pragma omp atomic
      assign[p]++;
    }
  }
#pragma omp master
  {
    int total = 0;
    int nz = 0;
    for (long p = 0; p < P; p++) {
      total += assign[p];
      nz += (assign[p] > 0);
    }
    int ps = 0;
#if 0
    printf("total path counts: %d\n", total);
    printf("non zero path counts: %d\n", nz);
#endif
    for (long p = 0; p < P; p++) {
      int c = assign[p];
      int w = (ps * nw) / total;
      assert(0 <= w);
      assert(w <= nw);
      if (w == nw) assert(c == 0);
      assign[p] = w;
#if 0
      printf("path %ld count %d -> %d\n", p, c, w);
#endif
      ps += c;
    }
  }
}

int backward(long M, long N, long P, long bn, long pn, long ml, float gl,
	      float x[restrict M][N], int t[restrict M], float W[restrict P][N],
	      float gx[restrict M][N], float gW[restrict P][N], 
	      int begins[restrict bn], int paths[restrict pn],
	      float codes[restrict pn], float wxy[restrict M][ml]) {
#if REC_BACKWARD
  int max_nw = omp_get_max_threads();
  backward_rec R[max_nw];
#endif
  int assign[P];
#if REC_BACKWARD
  R[0].t[0] = _rdtsc();
#endif
  
#pragma omp parallel
  {
    int rk = omp_get_thread_num();
    int nw = omp_get_num_threads();
#if REC_BACKWARD
    backward_rec * RR = &R[rk];
    RR->t[1] = _rdtsc();
    RR->count_i = 0;
    RR->count_p = 0;
#endif
    calc_p_assignment(nw, M, P, bn, pn, t, begins, paths, assign);

    for (long i = 0; i < M; i++) {
      int my_i = (rk * M  <= i * nw) && (i * nw < ((rk + 1) * M));
      int it = t[i];
      int begin = begins[it];
      int end = begins[it + 1];
      for (int k = begin, o = 0; k < end; k++, o++) {
	int p = paths[k];
	int my_p = (assign[p] == rk);
	if (!my_i && !my_p) continue;
	float g = -gl * codes[k] / (1.0 + expf(wxy[i][o]));
	if (my_i) {
#if REC_BACKWARD
	  RR->count_i++;
#endif
#pragma omp simd
	  for (int j = 0; j < N; j++) {
	    gx[i][j] += g * W[p][j];
	  }
	}
	if (my_p) {
#if REC_BACKWARD
	  RR->count_p++;
#endif
#pragma omp simd
	  for (int j = 0; j < N; j++) {
	    gW[p][j] += g * x[i][j];
	  }
	}
      }
    }
#if REC_BACKWARD
    RR->t[2] = _rdtsc();
#endif
  }
#if REC_BACKWARD
  R[0].t[3] = _rdtsc();
  {
    tsc_t start_t = R[0].t[0];
    tsc_t end_t = R[0].t[3];
    for (int i = 0; i < max_nw; i++) {
      printf("[%3d]: %lld %lld i=%ld p=%ld\n",
	    i, R[i].t[1] - start_t, R[i].t[2] - start_t, R[i].count_i, R[i].count_p);
    }
    printf("end: %lld\n", end_t - start_t);
  }
#endif
  return 0;
}

