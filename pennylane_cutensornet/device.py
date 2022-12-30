from typing import Any, Dict, Iterable, List, Literal, Union

import cuquantum
import numpy as np
import pennylane as qml


class cuTensorNetDevice(qml.QubitDevice):
    """Device for cuTensorNet"""
    name = "PennyLane plugin for cuTensorNet of NVIDIA cuQuantum"
    short_name = "cuquantum.cutensornet"
    pennylane_requires = "2"
    version = "0.0.0"
    author = "ymd-h"

    operations = {"RX", "CNOT"}
    observables = {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "SparseHamiltonian",
        "Hamiltonian",
        "Identity",
    }

    def __init__(self,
                 wires: Union[int, Iterable[int, str]],
                 shots: Union[None, int, List[int]],
                 mode: Literal["qiskit", "cirq"] = "qiskit"):
        super().__init__(wires=wires, shots=shots)

        if mode == "qiskit":
            self._dev = qml.device("qiskit.basicaer", wires=wires)
            self._conv = self._conv_qiskit
        elif mode == "cirq":
            self._dev = qml.device("cirq.simulator", wires=wires)
            self._conv = self._conv_cirq
        else:
            raise ValueError(f"{mode} is not supported.")


    def _conv_qiskit(self, operations, **kwargs):
        self._dev.create_circuit_object(operations, **kwargs)
        return cuquantum.CircuitToEinsum(self._dev._circuit)

    def _conv_cirq(self, operations, **kwargs):
        from pennylane_cirq.cirq_device import CirqDevice
        CirqDevice.apply(self._dev, operations, **kwargs)
        return cuquantum.CircuitToEinsum(self._dev.circuit)


    def apply(self, operations: List[qml.Operation], **kwargs):
        conv = self._conv(operations, **kwargs)
        expr, op = conv.state_vector()
        state = cuquantum.contract(expr, *op)
        self._state = np.array(state, copy=True)
        return state


    def analytic_probabilitiy(self, wires: Union[None,
                                                 Iterable[Union[int, str]],
                                                 int,
                                                 str,
                                                 qml.Wires] = None):
        if self._state is None:
            return None

        prob = self.marginal_prob(np.abs(self._state) ** 2, wires)


    @classmethod
    def capability(cls) -> Dict[str, Any]:
        return {
            **super().capability(),
        }
