double forward(int K, int T, double * init_state_distrib, double * transmat, double * obslik, double * alpha);
void backward(int K, int T, double * transmat, double * obslik, double * alpha, double * gamma, double * beta);
double fwdbkw(int K, int T, double * init_state_distrib, double * transmat, double * obslik, double * gamma, double * alpha, double * beta);

double normalizeInPlace(double * A, unsigned int nrows, unsigned int stride);
void multiplyInPlace(double * result, double * u, double *v, unsigned int nrows, unsigned int rstride, unsigned int ustride, unsigned int vstride);
void multiplyMatrixInPlace(double * result, double * A, double * v, unsigned int nrows, unsigned int stride);
void transposeSquareInPlace(double * out, double * in, unsigned int K, unsigned int T);

void outerProductUVInPlace(double * out, double * u, double * v, unsigned int K, unsigned int T);
void componentVectorMultiplyInPlace(double * out, double * u, double * v, unsigned int K, unsigned int T);

