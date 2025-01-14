#pragma once

#include <co_math.h>
#include <common.h>
#include <torch_common.h>
#include <torch_tensor.h>

template <typename T>
struct PointCatForward {
    const Tensor2<T> in_features;  // n_elems x channels
    const Tensor1<int64_t> key;       // prefix_sum
    int n_cat;

    Tensor2<T> out_features;  // prefix_sum - 1 x n_cat . channels

    PointCatForward(const Tensor2<T> in_features, const Tensor1<int64_t> key,
                    int n_cat, Tensor2<T> out_features)
        : in_features(in_features),
          key(key),
          n_cat(n_cat),
          out_features(out_features) {}

    CPU_GPU_FUNCTION void operator()(int64_t idx) {
        // idx \in [0, prefix_sum]
        const int64_t channels = in_features.shape[1];
        int64_t key_from = key(idx);
        int64_t key_to = key(idx + 1);
        for (int64_t n = 0; n < n_cat && n < (key_to - key_from); ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                out_features(idx, n * channels + c) =
                    in_features(key_from + n, c);
            }
        }
    }
};

template <typename T>
struct PointCatBackward {
    const Tensor2<T> grad_out_features;  // prefix_sum - 1 x n_cat . channels
    const Tensor1<int64_t> key;             // prefix_sum
    int n_cat;

    Tensor2<T> grad_in_features;  // n_elems x channels

    PointCatBackward(const Tensor2<T> grad_out_features,
                     const Tensor1<int64_t> key, int n_cat,
                     Tensor2<T> grad_in_features)
        : grad_out_features(grad_out_features),
          key(key),
          n_cat(n_cat),
          grad_in_features(grad_in_features) {}

    CPU_GPU_FUNCTION void operator()(int64_t idx) {
        // idx \in [0, prefix_sum]
        const int64_t channels = grad_in_features.shape[1];
        int64_t key_from = key(idx);
        int64_t key_to = key(idx + 1);
        for (int64_t n = 0; n < n_cat && n < (key_to - key_from); ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                grad_in_features(key_from + n, c) =
                    grad_out_features(idx, n * channels + c);
            }
        }
    }
};
