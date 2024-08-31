#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "GraphMetNetwork.h"

namespace py = pybind11;

PYBIND11_MODULE(graphmetnetwork, m) {
    py::class_<GraphMetNetwork>(m, "GraphMetNetwork")

        .def(py::init<>())

        .def("GraphMetNetworkLayers", [](GraphMetNetwork &gmn, py::array_t<float> x_cont, py::array_t<int> x_cat, int num_nodes) {
            auto buf_x_cont = x_cont.request();
            auto buf_x_cat = x_cat.request();

            if (buf_x_cont.ndim != 2 || buf_x_cat.ndim != 2) {
                throw std::runtime_error("Array dimensions mismatch.");
            }

            gmn.GraphMetNetworkLayers(
            reinterpret_cast<float (*)[CONT_DIM]>(buf_x_cont.ptr),
            reinterpret_cast<int (*)[CAT_DIM]>(buf_x_cat.ptr),
            num_nodes
            );
        })

        .def("load_weights", &GraphMetNetwork::load_weights)

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

        .def("get_num_nodes", [](GraphMetNetwork &gmn) {
            return gmn.get_num_nodes();
        })

        .def("get_num_edges", [](GraphMetNetwork &gmn) {
            return gmn.get_num_edges();
        });
}


