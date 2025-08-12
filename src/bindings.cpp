#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "attn.h"

namespace py = pybind11;
using namespace attn;

static inline int64_t stride_elems(const py::array &a, int axis) {
    // Convert byte strides to element strides (float32 assumed)
    auto s = a.strides()[axis];
    return static_cast<int64_t>(s / static_cast<ssize_t>(sizeof(float)));
}

py::array_t<float> attn_block_dense_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> Q,
    py::array_t<float, py::array::c_style | py::array::forcecast> K,
    py::array_t<float, py::array::c_style | py::array::forcecast> V,
    bool causal = false) {
    // Expect Q: [H, L, Dh], K/V: [H, T, Dh]
    if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3)
        throw std::invalid_argument("Q,K,V must be 3D");

    int H = (int)Q.shape(0);
    int L = (int)Q.shape(1);
    int Dh = (int)Q.shape(2);
    if ((int)K.shape(0) != H || (int)V.shape(0) != H)
        throw std::invalid_argument("H mismatch");
    if ((int)K.shape(2) != Dh || (int)V.shape(2) != Dh)
        throw std::invalid_argument("Dh mismatch");
    int T = (int)K.shape(1);
    if ((int)V.shape(1) != T)
        throw std::invalid_argument("K/V T mismatch");
    if (causal && L != T)
        throw std::invalid_argument("causal requires L==T");

    TensorF32 Qv{
        Q.data(),          H, L, Dh, stride_elems(Q, 0), stride_elems(Q, 1),
        stride_elems(Q, 2)};
    TensorF32 Kv{
        K.data(),          H, T, Dh, stride_elems(K, 0), stride_elems(K, 1),
        stride_elems(K, 2)};
    TensorF32 Vv{
        V.data(),          H, T, Dh, stride_elems(V, 0), stride_elems(V, 1),
        stride_elems(V, 2)};

    py::array_t<float> O({H, L, Dh});
    auto Obuf = O.request();
    MutableTensor3F32 Ov{(float *)Obuf.ptr, H,           L, Dh,
                         (int64_t)L * Dh,   (int64_t)Dh, 1};

    attn_block_dense(Qv, Kv, Vv, Ov, causal);
    return O;
}

py::array_t<float> attn_step_dense_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> Q_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> K_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> V_in) {
    // Expect Q: [H, Dh], K/V: [H, T, Dh]
    if (Q_in.ndim() != 2)
        throw std::invalid_argument("Q must be 2D [H, Dh]");
    if (K_in.ndim() != 3 || V_in.ndim() != 3)
        throw std::invalid_argument("K/V must be 3D [H, T, Dh]");

    int H = static_cast<int>(Q_in.shape(0));
    int Dh = static_cast<int>(Q_in.shape(1));

    if (static_cast<int>(K_in.shape(0)) != H ||
        static_cast<int>(V_in.shape(0)) != H)
        throw std::invalid_argument("K/V H must match Q H");
    if (static_cast<int>(K_in.shape(2)) != Dh ||
        static_cast<int>(V_in.shape(2)) != Dh)
        throw std::invalid_argument("K/V Dh must match Q Dh");

    int T = static_cast<int>(K_in.shape(1));
    if (static_cast<int>(V_in.shape(1)) != T)
        throw std::invalid_argument("K and V must have same T");

    TensorF32 Qv;
    Qv.data = Q_in.data();
    Qv.H = H;
    Qv.T = 1; // not used
    Qv.Dh = Dh;
    Qv.strideH = stride_elems(Q_in, 0);
    Qv.strideT = 0; // not used
    Qv.strideDh = stride_elems(Q_in, 1);

    TensorF32 Kv;
    Kv.data = K_in.data();
    Kv.H = H;
    Kv.T = T;
    Kv.Dh = Dh;
    Kv.strideH = stride_elems(K_in, 0);
    Kv.strideT = stride_elems(K_in, 1);
    Kv.strideDh = stride_elems(K_in, 2);

    TensorF32 Vv;
    Vv.data = V_in.data();
    Vv.H = H;
    Vv.T = T;
    Vv.Dh = Dh;
    Vv.strideH = stride_elems(V_in, 0);
    Vv.strideT = stride_elems(V_in, 1);
    Vv.strideDh = stride_elems(V_in, 2);

    // Output
    py::array_t<float> O_out({H, Dh});
    auto O_buf = O_out.request();
    MutableTensorF32 Ov;
    Ov.data = static_cast<float *>(O_buf.ptr);
    Ov.H = H;
    Ov.Dh = Dh;
    Ov.strideH = Dh; // row-major [H, Dh] for the created array
    Ov.strideDh = 1;

    attn_step_dense(Qv, Kv, Vv, Ov);
    return O_out;
}

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
    m.def("attn_step_dense", &attn_step_dense_py, "Step dense attention");
    m.def("attn_block_dense", &attn_block_dense_py, py::arg("Q"), py::arg("K"),
          py::arg("V"), py::arg("causal") = false,
          "Block dense attention: Q:[H,L,Dh], K/V:[H,T,Dh] -> O:[H,L,Dh]");
    m.def("mha_block_dense", &mha_block_dense_py, py::arg("x"), py::arg("W_in"),
          py::arg("b_in"), py::arg("W_out"), py::arg("b_out"),
          py::arg("num_heads"), py::arg("causal") = false,
          "Full MHA block: x:[T,D], W_in:[3D,D], b_in:[3D], W_out:[D,D], "
          "b_out:[D] -> y:[T,D]");
}