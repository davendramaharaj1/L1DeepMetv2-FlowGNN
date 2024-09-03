#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "GraphMetNetwork.h"

namespace py = pybind11;

PYBIND11_MODULE(graphmetnetwork, m) {
    py::class_<GraphMetNetwork>(m, "GraphMetNetwork")


        // constructor
        .def(py::init<>())

        // forward layer
        .def("GraphMetNetworkLayers", [](GraphMetNetwork &gmn, py::array_t<float> x_cont, py::array_t<int> x_cat, py::array_t<int> batch, int num_nodes) {
            auto buf_x_cont = x_cont.request();
            auto buf_x_cat = x_cat.request();
            auto buf_batch = batch.request();

            if (buf_x_cont.ndim != 2 || buf_x_cat.ndim != 2) {
                throw std::runtime_error("Array dimensions mismatch.");
            }

            gmn.GraphMetNetworkLayers(
            reinterpret_cast<float (*)[CONT_DIM]>(buf_x_cont.ptr),
            reinterpret_cast<int (*)[CAT_DIM]>(buf_x_cat.ptr),
            reinterpret_cast<int *>(buf_batch.ptr),
            num_nodes
            );
        })

        // loading weights
        .def("load_weights", [](GraphMetNetwork &gmn, std::string weights){
            gmn.load_weights(weights);
            })
        
        // input getters
        .def("get_x_cont", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), CONT_DIM}, gmn.get_x_cont());
        })

        .def("get_x_cat", [](GraphMetNetwork &gmn) {
            return py::array_t<int>({gmn.get_num_nodes(), CAT_DIM}, gmn.get_x_cat());
        })

        .def("get_batch", [](GraphMetNetwork &gmn) {
            return py::array_t<int>({gmn.get_num_nodes()}, gmn.get_batch());
        })

        .def("get_num_nodes", [](GraphMetNetwork &gmn) {
            return gmn.get_num_nodes();
        })

        // internal variable getters
        .def("get_edge_index", [](GraphMetNetwork &gmn) {
            return py::array_t<int>({gmn.get_num_edges(), 2}, gmn.get_edge_index());
        })

        .def("get_etaphi", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), 2}, gmn.get_etaphi());
        })

        .def("get_num_edges", [](GraphMetNetwork &gmn) {
            return gmn.get_num_edges();
        })

        // intermediate getters
        .def("get_output", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes()}, gmn.get_output());
        })

        .def("get_emb_cont", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), HIDDEN_DIM/2}, gmn.get_emb_cont());
        })

        .def("get_emb_chrg", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), HIDDEN_DIM/4}, gmn.get_emb_chrg());
        })

        .def("get_emb_pdg", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), HIDDEN_DIM/4}, gmn.get_emb_pdg());
        })

        .def("get_emb_cat", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), HIDDEN_DIM/2}, gmn.get_emb_cat());
        })

        .def("get_emb", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), HIDDEN_DIM}, gmn.get_emb());
        })

        .def("get_emb1", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), HIDDEN_DIM}, gmn.get_emb1());
        })

        .def("get_emb2", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), HIDDEN_DIM}, gmn.get_emb2());
        })

        // Weight array getters
        .def("get_graphnet_embed_charge_weight", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({3, 8}, gmn.get_graphnet_embed_charge_weight());
        })

        .def("get_graphnet_embed_pdgid_weight", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({7, 8}, gmn.get_graphnet_embed_pdgid_weight());
        })

        .def("get_graphnet_embed_continuous_0_weight", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({16, 6}, gmn.get_graphnet_embed_continuous_0_weight());
        })

        .def("get_graphnet_embed_continuous_0_bias", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({16}, gmn.get_graphnet_embed_continuous_0_bias());
        })

        .def("get_graphnet_embed_categorical_0_weight", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({16, 16}, gmn.get_graphnet_embed_categorical_0_weight());
        })

        .def("get_graphnet_embed_categorical_0_bias", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({16}, gmn.get_graphnet_embed_categorical_0_bias());
        })

        .def("get_graphnet_encode_all_weight", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32, 32}, gmn.get_graphnet_encode_all_weight());
        })

        .def("get_graphnet_encode_all_bias", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32}, gmn.get_graphnet_encode_all_bias());
        })

        .def("get_graphnet_bn_all_weight", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32}, gmn.get_graphnet_bn_all_weight());
        })

        .def("get_graphnet_bn_all_bias", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32}, gmn.get_graphnet_bn_all_bias());
        })

        .def("get_graphnet_bn_all_running_mean", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32}, gmn.get_graphnet_bn_all_running_mean());
        })

        .def("get_graphnet_bn_all_running_var", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32}, gmn.get_graphnet_bn_all_running_var());
        })

        .def("get_graphnet_bn_all_batches_tracked", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({1}, gmn.get_graphnet_bn_all_batches_tracked());
        })

        .def("get_graphnet_conv_continuous_0_0_nn_0_weight", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32, 64}, gmn.get_graphnet_conv_continuous_0_0_nn_0_weight());
        })

        .def("get_graphnet_conv_continuous_0_0_nn_0_bias", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32}, gmn.get_graphnet_conv_continuous_0_0_nn_0_bias());
        })

        .def("get_graphnet_conv_continuous_0_1_weight", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32}, gmn.get_graphnet_conv_continuous_0_1_weight());
        })

        .def("get_graphnet_conv_continuous_0_1_bias", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32}, gmn.get_graphnet_conv_continuous_0_1_bias());
        })

        .def("get_graphnet_conv_continuous_0_1_running_mean", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32}, gmn.get_graphnet_conv_continuous_0_1_running_mean());
        })

        .def("get_graphnet_conv_continuous_0_1_running_var", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32}, gmn.get_graphnet_conv_continuous_0_1_running_var());
        })

        .def("get_graphnet_conv_continuous_0_1_num_batches_tracked", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({1}, gmn.get_graphnet_conv_continuous_0_1_num_batches_tracked());
        })

        .def("get_graphnet_conv_continuous_1_0_nn_0_weight", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32, 64}, gmn.get_graphnet_conv_continuous_1_0_nn_0_weight());
        })

        .def("get_graphnet_conv_continuous_1_0_nn_0_bias", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32}, gmn.get_graphnet_conv_continuous_1_0_nn_0_bias());
        })

        .def("get_graphnet_conv_continuous_1_1_weight", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32}, gmn.get_graphnet_conv_continuous_1_1_weight());
        })

        .def("get_graphnet_conv_continuous_1_1_bias", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32}, gmn.get_graphnet_conv_continuous_1_1_bias());
        })

        .def("get_graphnet_conv_continuous_1_1_running_mean", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32}, gmn.get_graphnet_conv_continuous_1_1_running_mean());
        })

        .def("get_graphnet_conv_continuous_1_1_running_var", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({32}, gmn.get_graphnet_conv_continuous_1_1_running_var());
        })

        .def("get_graphnet_conv_continuous_1_1_num_batches_tracked", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({1}, gmn.get_graphnet_conv_continuous_1_1_num_batches_tracked());
        })

        .def("get_graphnet_output_0_weight", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({16, 32}, gmn.get_graphnet_output_0_weight());
        })

        .def("get_graphnet_output_0_bias", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({16}, gmn.get_graphnet_output_0_bias());
        })

        .def("get_graphnet_output_2_weight", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({1, 16}, gmn.get_graphnet_output_2_weight());
        })

        .def("get_graphnet_output_2_bias", [](GraphMetNetwork &gmn) {
            return py::array_t<float>({1}, gmn.get_graphnet_output_2_bias());
        });

}


