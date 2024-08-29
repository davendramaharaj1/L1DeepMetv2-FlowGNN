#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "GraphMetNetwork.h"

namespace py = pybind11;

PYBIND11_MODULE(graphmetnetwork, m) {
    py::class_<GraphMetNetwork>(m, "GraphMetNetwork")
        .def(py::init<>())
        .def("GraphMetNetworkLayers", [](GraphMetNetwork &gmn, py::array_t<float> x_cont, py::array_t<int> x_cat, int num_nodes) {
            gmn.GraphMetNetworkLayers(x_cont.mutable_data(), x_cat.mutable_data(), num_nodes);
        })
        .def("get_output", [](const GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes()}, gmn.get_output());
        })
        .def("get_emb_cont", [](const GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), HIDDEN_DIM/2}, gmn.get_emb_cont());
        })
        .def("get_emb_chrg", [](const GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), HIDDEN_DIM/4}, gmn.get_emb_chrg());
        })
        .def("get_emb_pdg", [](const GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), HIDDEN_DIM/4}, gmn.get_emb_pdg());
        })
        .def("get_emb_cat", [](const GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), HIDDEN_DIM/2}, gmn.get_emb_cat());
        })
        .def("get_emb", [](const GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), HIDDEN_DIM}, gmn.get_emb());
        })
        .def("get_emb1", [](const GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), HIDDEN_DIM}, gmn.get_emb1());
        })
        .def("get_emb2", [](const GraphMetNetwork &gmn) {
            return py::array_t<float>({gmn.get_num_nodes(), HIDDEN_DIM}, gmn.get_emb2());
        });
}


