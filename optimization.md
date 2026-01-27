Techiques applied:
Compiler flags: -O3 -march=native -ffast-math -flto (~3x improvement from original ~25ms to ~8ms)

LSTM loop restructuring: Combined all 4 gates into a single array, better memory access patterns

Sequence subsampling: Configurable via --subsample N flag - trades accuracy for speed

Pre-allocation: Vectors allocated once outside the loop, using std::swap instead of allocation