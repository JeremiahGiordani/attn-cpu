#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "attn_api.h"
#include "gemm.h"

namespace py = pybind11;
using namespace attn;

py::array_t<float> mha_block_dense_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<float, py::array::c_style | py::array::forcecast> W_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> b_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> W_out,
    py::array_t<float, py::array::c_style | py::array::forcecast> b_out,
    int num_heads, bool causal = false) {
    if (x.ndim() != 2)
        throw std::invalid_argument("x must be [T,D]");
    if (W_in.ndim() != 2)
        throw std::invalid_argument("W_in must be [3D,D]");
    if (b_in.ndim() != 1)
        throw std::invalid_argument("b_in must be [3D]");
    if (W_out.ndim() != 2)
        throw std::invalid_argument("W_out must be [D,D]");
    if (b_out.ndim() != 1)
        throw std::invalid_argument("b_out must be [D]");

    const int T = (int)x.shape(0);
    const int D = (int)x.shape(1);
    if ((int)W_in.shape(0) != 3 * D || (int)W_in.shape(1) != D)
        throw std::invalid_argument("W_in shape must be [3D, D]");
    if ((int)b_in.shape(0) != 3 * D)
        throw std::invalid_argument("b_in shape must be [3D]");
    if ((int)W_out.shape(0) != D || (int)W_out.shape(1) != D)
        throw std::invalid_argument("W_out shape must be [D, D]");
    if ((int)b_out.shape(0) != D)
        throw std::invalid_argument("b_out shape must be [D]");
    if (num_heads <= 0 || (D % num_heads) != 0)
        throw std::invalid_argument("num_heads must divide D");

    py::array_t<float> y({T, D});
    attn::mha_block_dense(x.data(), T, D, W_in.data(), b_in.data(),
                          W_out.data(), b_out.data(), num_heads, causal,
                          y.mutable_data());
    return y;
}

py::array_t<float> gemm_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B,
    float alpha = 1.0f,
    float beta  = 0.0f,
    int Mb = 192, int Nb = 144, int Kb = 256,
    int mr = 16, int nr = 12, int ku = 4)
{
    if (A.ndim() != 2 || B.ndim() != 2)
        throw std::invalid_argument("A and B must be 2D: A[M,K], B[K,N]");

    const int M = static_cast<int>(A.shape(0));
    const int K = static_cast<int>(A.shape(1));
    const int Kb_B = static_cast<int>(B.shape(0));
    const int N = static_cast<int>(B.shape(1));
    if (K != Kb_B)
        throw std::invalid_argument("Inner dim mismatch: A[M,K] x B[K,N]");

    py::array_t<float> C({M, N});
    // If beta != 0, user should have passed an initialized C; for simplicity we just scale zeros→still zero.
    gemm::sgemm_blocked(
        A.data(), M, K,
        B.data(), N,
        C.mutable_data(),
        alpha, beta,
        Mb, Nb, Kb,
        mr, nr, ku
    );
    return C;
}

PYBIND11_MODULE(_attn_cpu, m) {
    m.doc() = "CPU Dense Attention (step + block)";
    m.def("mha_block_dense", &mha_block_dense_py, py::arg("x"), py::arg("W_in"),
          py::arg("b_in"), py::arg("W_out"), py::arg("b_out"),
          py::arg("num_heads"), py::arg("causal") = false,
          "Full MHA block: x:[T,D], W_in:[3D,D], b_in:[3D], W_out:[D,D], "
          "b_out:[D] -> y:[T,D]");
    m.def("gemm",
          &gemm_py,
          py::arg("A"), py::arg("B"),
          py::arg("alpha") = 1.0f,
          py::arg("beta")  = 0.0f,
          py::arg("Mb") = 256, py::arg("Nb") = 96, py::arg("Kb") = 288,
          py::arg("mr") = 16,  py::arg("nr") = 24, py::arg("ku") = 4,
          "Blocked SGEMM: C = alpha * A@B + beta*C, A:[M,K], B:[K,N] -> C:[M,N]\n"
          "Tunables default to AVX-512-friendly picks. Pass num_threads>0 to set OpenMP threads."
    );
}