#include <assert.h>

int backward(long M, long N, long K, long L, long ignore_label,
             int x[M][N], float gW[K][L], float gy[M][N][L]) {
  for (long i = 0; i < M; i++) {
    for (long j = 0; j < N; j++) {
      int ix = x[i][j];
      if (ix == ignore_label) continue;
      for (long l = 0; l < L; l++) {
        gW[ix][l] += gy[i][j][l]; /* !! */
      }
    }
  }
  return 0;
}
