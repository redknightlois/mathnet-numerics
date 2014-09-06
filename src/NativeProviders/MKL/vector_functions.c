#include "mkl_vml.h"
#include "wrapper_common.h"

#if GCC
extern "C" {
#endif
	DLLEXPORT void s_vector_add(const int n, const float x[], const int xOffset, const float y[], const int yOffset, float result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vsAdd(n, xo, yo, ro);
}

	DLLEXPORT void s_vector_subtract(const int n, const float x[], const int xOffset, const float y[], const int yOffset, float result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vsSub(n, xo, yo, ro);
}

	DLLEXPORT void s_vector_multiply(const int n,const float x[], const int xOffset, const float y[], const int yOffset, float result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vsMul(n, xo, yo, ro);
}

	DLLEXPORT void s_vector_divide(const int n,const float x[], const int xOffset, const float y[], const int yOffset, float result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vsDiv(n, xo, yo, ro);
}

	DLLEXPORT void d_vector_add(const int n,const double x[], const int xOffset, const double y[], const int yOffset, double result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vdAdd(n, xo, yo, ro);
}

	DLLEXPORT void d_vector_subtract(const int n, const double x[], const int xOffset, const double y[], const int yOffset, double result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vdSub(n, xo, yo, ro);
}

	DLLEXPORT void d_vector_multiply(const int n,const double x[], const int xOffset, const double y[], const int yOffset, double result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vdMul(n, xo, yo, ro);
}

	DLLEXPORT void d_vector_divide(const int n,const double x[], const int xOffset, const double y[], const int yOffset, double result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vdDiv(n, xo, yo, ro);
}

	DLLEXPORT void c_vector_add(const int n,const MKL_Complex8 x[], const int xOffset, const MKL_Complex8 y[], const int yOffset, MKL_Complex8 result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vcAdd(n, xo, yo, ro);
}

	DLLEXPORT void c_vector_subtract(const int n,const MKL_Complex8 x[], const int xOffset, const MKL_Complex8 y[], const int yOffset, MKL_Complex8 result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vcSub(n, xo, yo, ro);
}

	DLLEXPORT void c_vector_multiply(const int n,const MKL_Complex8 x[], const int xOffset, const MKL_Complex8 y[], const int yOffset, MKL_Complex8 result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vcMul(n, xo, yo, ro);
}

	DLLEXPORT void c_vector_divide(const int n,const MKL_Complex8 x[], const int xOffset, const MKL_Complex8 y[], const int yOffset, MKL_Complex8 result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vcDiv(n, xo, yo, ro);
}

	DLLEXPORT void z_vector_add(const int n,const MKL_Complex16 x[], const int xOffset, const MKL_Complex16 y[], const int yOffset, MKL_Complex16 result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vzAdd(n, xo, yo, ro);
}

	DLLEXPORT void z_vector_subtract(const int n,const MKL_Complex16 x[], const int xOffset, const MKL_Complex16 y[], const int yOffset, MKL_Complex16 result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vzSub(n, xo, yo, ro);
}

	DLLEXPORT void z_vector_multiply(const int n,const MKL_Complex16 x[], const int xOffset, const MKL_Complex16 y[], const int yOffset, MKL_Complex16 result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vzMul(n, xo, yo, ro);
}

	DLLEXPORT void z_vector_divide(const int n,const MKL_Complex16 x[], const int xOffset, const MKL_Complex16 y[], const int yOffset, MKL_Complex16 result[], const int rOffset){
		auto* xo = x + xOffset;
		auto* yo = y + yOffset;
		auto* ro = result + rOffset;
		vzDiv(n, xo, yo, ro);
}
#if GCC
}
#endif
