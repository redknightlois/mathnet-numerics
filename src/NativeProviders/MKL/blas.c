#include "mkl_cblas.h"
#include "wrapper_common.h"

#if GCC 
extern "C" { 
#endif
DLLEXPORT void s_axpy(const MKL_INT n, const float alpha, const float x[], const int xOffset, float y[], const int yOffset){
	auto* xo = x + xOffset;
	auto* yo = y + yOffset;
	cblas_saxpy(n, alpha, xo, 1, yo, 1);
}

DLLEXPORT void d_axpy(const MKL_INT n, const double alpha, const double x[], const int xOffset, double y[], const int yOffset){
	auto* xo = x + xOffset;
	auto* yo = y + yOffset;
	cblas_daxpy(n, alpha, xo, 1, yo, 1);
}

DLLEXPORT void c_axpy(const MKL_INT n, const MKL_Complex8 alpha, const MKL_Complex8 x[], const int xOffset, MKL_Complex8 y[], const int yOffset){
	auto* xo = x + xOffset;
	auto* yo = y + yOffset;
	cblas_caxpy(n, &alpha, xo, 1, yo, 1);
}

DLLEXPORT void z_axpy(const MKL_INT n, const MKL_Complex16 alpha, const MKL_Complex16 x[], const int xOffset, MKL_Complex16 y[], const int yOffset){
	auto* xo = x + xOffset;
	auto* yo = y + yOffset;
	cblas_zaxpy(n, &alpha, xo, 1, yo, 1);
}

DLLEXPORT void s_scale(const MKL_INT n, const float alpha, float x[], const int xOffset){
	auto* xo = x + xOffset;
	cblas_sscal(n, alpha, xo, 1);
}

DLLEXPORT void d_scale(const MKL_INT n, const double alpha, double x[], const int xOffset){
	auto* xo = x + xOffset;
	cblas_dscal(n, alpha, xo, 1);
}

DLLEXPORT void c_scale(const MKL_INT n, const MKL_Complex8 alpha, MKL_Complex8 x[], const int xOffset){
	auto* xo = x + xOffset;
	cblas_cscal(n, &alpha, xo, 1);
}

DLLEXPORT void z_scale(const MKL_INT n, const MKL_Complex16 alpha, MKL_Complex16 x[], const int xOffset){
	auto* xo = x + xOffset;
	cblas_zscal(n, &alpha, xo, 1);
}

DLLEXPORT float s_dot_product(const MKL_INT n, const float x[], const int xOffset, const float y[], const int yOffset){
	auto* xo = x + xOffset;
	auto* yo = y + yOffset;
	return cblas_sdot(n, xo, 1, yo, 1);
}

DLLEXPORT double d_dot_product(const MKL_INT n, const double x[], const int xOffset, const double y[], const int yOffset){
	auto* xo = x + xOffset;
	auto* yo = y + yOffset;
	return cblas_ddot(n, xo, 1, yo, 1);
}

DLLEXPORT MKL_Complex8 c_dot_product(const MKL_INT n, const MKL_Complex8 x[], const int xOffset, const MKL_Complex8 y[], const int yOffset){
	auto* xo = x + xOffset;
	auto* yo = y + yOffset;
	MKL_Complex8 ret;
	cblas_cdotu_sub(n, xo, 1, yo, 1, &ret);
	return ret;
}

DLLEXPORT MKL_Complex16 z_dot_product(const MKL_INT n, const MKL_Complex16 x[], const int xOffset, const MKL_Complex16 y[], const int yOffset){
	auto* xo = x + xOffset;
	auto* yo = y + yOffset;
	MKL_Complex16 ret;
	cblas_zdotu_sub(n, xo, 1, yo, 1, &ret);
	return ret;
}

DLLEXPORT void s_matrix_multiply(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, const MKL_INT m, const MKL_INT n, const MKL_INT k, const float alpha, const float x[], const int xOffset, const float y[], const int yOffset, const float beta, float c[], const int cOffset){
	MKL_INT lda = transA == CblasNoTrans ? m : k;
	MKL_INT ldb = transB == CblasNoTrans ? k : n;

	auto* xo = x + xOffset;
	auto* yo = y + yOffset;
	auto* co = c + cOffset;

	cblas_sgemm(CblasColMajor, transA, transB, m, n, k, alpha, xo, lda, yo, ldb, beta, co, m);
}

DLLEXPORT void d_matrix_multiply(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, const MKL_INT m, const MKL_INT n, const MKL_INT k, const double alpha, const double x[], const int xOffset, const double y[], const int yOffset, const double beta, double c[], const int cOffset){
	MKL_INT lda = transA == CblasNoTrans ? m : k;
	MKL_INT ldb = transB == CblasNoTrans ? k : n;

	auto* xo = x + xOffset;
	auto* yo = y + yOffset;
	auto* co = c + cOffset;

	cblas_dgemm(CblasColMajor, transA, transB, m, n, k, alpha, xo, lda, yo, ldb, beta, co, m);
}

DLLEXPORT void c_matrix_multiply(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, const MKL_INT m, const MKL_INT n, const MKL_INT k, const MKL_Complex8 alpha, const MKL_Complex8 x[], const int xOffset, const MKL_Complex8 y[], const int yOffset, const MKL_Complex8 beta, MKL_Complex8 c[], const int cOffset){
	MKL_INT lda = transA == CblasNoTrans ? m : k;
	MKL_INT ldb = transB == CblasNoTrans ? k : n;

	auto* xo = x + xOffset;
	auto* yo = y + yOffset;
	auto* co = c + cOffset;

	cblas_cgemm(CblasColMajor, transA, transB, m, n, k, &alpha, xo, lda, yo, ldb, &beta, co, m);
}

DLLEXPORT void z_matrix_multiply(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, const MKL_INT m, const MKL_INT n, const MKL_INT k, const MKL_Complex16 alpha, const MKL_Complex16 x[], const int xOffset, const MKL_Complex16 y[], const int yOffset, const MKL_Complex16 beta, MKL_Complex16 c[], const int cOffset){
	MKL_INT lda = transA == CblasNoTrans ? m : k;
	MKL_INT ldb = transB == CblasNoTrans ? k : n;

	auto* xo = x + xOffset;
	auto* yo = y + yOffset;
	auto* co = c + cOffset;

	cblas_zgemm(CblasColMajor, transA, transB, m, n, k, &alpha, xo, lda, yo, ldb, &beta, co, m);
}

#if GCC 
} 
#endif
