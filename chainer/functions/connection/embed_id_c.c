#include <assert.h>
#include <omp.h>

static void calc_load_balance(long M, long N, long K, long L, long ignore_label,
			      int x[restrict M][N], int assign[restrict K], int nw) {
#pragma omp for
  for (long i = 0; i < K; i++) {
    assign[i] = 0;
  }
#pragma omp for collapse(2)
  for (long i = 0; i < M; i++) {
    for (long j = 0; j < N; j++) {
      int ix = x[i][j];
      if (ix == ignore_label) continue;
      assert(0 <= ix);
      assert(ix < K);
#pragma omp atomic
      assign[ix]++;
    }
  }
  int total = 0;
  for (long i = 0; i < K; i++) {
    total += assign[i];
  }
  int ps = 0;
  for (long i = 0; i < K; i++) {
    int c = assign[i];
    int w = (ps * nw) / total;
    assert(0 <= w);
    assert(w <= nw);
    if (w == nw) assert(c == 0);
    assign[i] = w;
    ps += c;
  }
}

static void update_gW(long M, long N, long K, long L,
		      long ix, long i, long j,
		      float gW[restrict K][L], float gy[restrict M][N][L]) {
  asm volatile("### update_gW loop begins");
#pragma omp simd
  for (long l = 0; l < L; l++) {
    gW[ix][l] += gy[i][j][l];
  }
  asm volatile("### update_gW loop ends");
}

int backward(long M, long N, long K, long L, long ignore_label,
	     int x[restrict M][N], float gW[restrict K][L], float gy[restrict M][N][L]) {
  int assign[K];
#pragma omp parallel
  {
    int nw = omp_get_num_threads();
    int rk = omp_get_thread_num();
    calc_load_balance(M, N, K, L, ignore_label, x, assign, nw);
    for (long i = 0; i < M; i++) {
      for (long j = 0; j < N; j++) {
	int ix = x[i][j];
	if (ix == ignore_label) continue;
	if (rk == assign[ix]) {
	  update_gW(M, N, K, L, ix, i, j, gW, gy);
	}
      }
    }
  }
  return 0;
}



int backward_(long M, long N, long K, long L, long ignore_label,
	      int x[restrict M][N], float gW[restrict K][L], float gy[restrict M][N][L]) {
  //#pragma omp parallel for schedule(runtime) 
  for (long i = 0; i < M; i++) {
    for (long j = 0; j < N; j++) {
      int ix = x[i][j];
      if (ix == ignore_label) continue;
      for (long l = 0; l < L; l++) {
	gW[ix][l] += gy[i][j][l];
      }
    }
  }
  return 0;
}
