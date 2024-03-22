#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "GripperControl.cpp"

// Save module:
// c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` gripper_control_bindings.cpp -o gripper_control_module`python3-config --extension-suffix`

namespace py = pybind11;

PYBIND11_MODULE(gripper_control_module, m) {
    py::class_<Obstacle>(m, "Obstacle")
        .def(py::init<const std::vector<float>&, const std::vector<float>&, float, const std::vector<float>&, float>());

    py::class_<FindTrajectory>(m, "FindTrajectory")
        .def(py::init<const std::vector<float>&, const std::vector<float>&, const std::vector<Obstacle>&, const std::vector<std::pair<float, float>>&,
                      float, float, bool>(), 
            py::arg("start"), py::arg("goal"), py::arg("obstacles"), py::arg("env_dimension"),
            py::arg("stepSize"), py::arg("safetyMargin"), py::arg("gripperClosed"))
        .def("aStarSearch", &FindTrajectory::aStarSearch)
        .def("teleportSearch", &FindTrajectory::teleportSearch);
}
