#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "attn_api.h"

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

PYBIND11_MODULE(_attn_cpu, m) {
    m.doc() = "CPU Dense Attention (step + block)";
    m.def("mha_block_dense", &mha_block_dense_py, py::arg("x"), py::arg("W_in"),
          py::arg("b_in"), py::arg("W_out"), py::arg("b_out"),
          py::arg("num_heads"), py::arg("causal") = false,
          "Full MHA block: x:[T,D], W_in:[3D,D], b_in:[3D], W_out:[D,D], "
          "b_out:[D] -> y:[T,D]");
}