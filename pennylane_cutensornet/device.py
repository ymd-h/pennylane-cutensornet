from typing import Any, Dict, Iterable, List, Literal, Union

import cuquantum
import numpy as np
import pennylane as qml


class cuTensorNetDevice(qml.QubitDevice):
    """
    Device for cuTensorNet

    Currently, this device first constructs quantum circuit with Qiskit or Cirq,
    then converts to Tensor Network by `CircuitToEinsum()` function.

    Parameters
    ----------
    wires : int or list of int or str
        The number of subsystems represented by the device,
        or iterable that contains unique labels for the subsystems
        as numbers (i.e., ``[-1, 0, 2]``)
        and/or strings (``["auxiliary", "q1", "q2"]``).
    shots : int or list of int, optional
        Number of circuit evaluations/random samples used
        to estimate probabilities, expectation values,
        variances of observables in non-analytic mode.
        If ``None``, the device calculates probability,
        expectation values, and variances analytically.
        If an integer, it specifies the number of samples
        to estimate these quantities.
        If a list of integers is passed, the circuit evaluations
        are batched over the list of shots.
    mode : "qiskit" or "cirq"
        Intermediate circuit type. Default is "qiskit"
    """
    name = "PennyLane plugin for cuTensorNet of NVIDIA cuQuantum"
    short_name = "cuquantum.cutensornet"
    pennylane_requires = "2"
    version = "0.0.0"
    author = "ymd-h"

    # Qiskit operations: https://github.com/PennyLaneAI/pennylane-qiskit/blob/v0.28.0/pennylane_qiskit/qiskit_device.py#L35
    # Cirq operations: https://github.com/PennyLaneAI/pennylane-cirq/blob/v0.28.0/pennylane_cirq/cirq_device.py#L136
    operations = {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "CNOT",
        "CZ",
        "SWAP",
        "ISWAP",
        # Adjoint(ISWAP), # Not supported by Qiskit
        # "SISWAP", # Not supported by Qiskit
        # Adjoint(SISWAP), # Not supported by Qiskit
        "RX",
        "RY",
        "RZ",
        # "Rot", # Not supported by Qiskit
        "S",
        # "Adjoint(S)", # Not supported by Cirq
        "T",
        # "Adjoint(T)", # Not supported by Cirq
        # "SX", # Not supported by Cirq
        # "Adjoint(SX)", # Not supported by Cirq
        "Identity",
        "CSWAP",
        "CRX",
        "CRY",
        "CRZ",
        # "CRot", # Not supported by Qiskit
        # "CSWAP", # Not supported by Qiskit
        "PhaseShift",
        # "ControlledPhaseShift", # Not supported by Qiskit
        # "QubitStateVector", Not supported by Cirq
        "Toffoli",
        "QubitUnitary",
        # "U1", # Not supported by Cirq
        # "U2", # Not supported by Cirq
        # "U3", # Not supported by Cirq
        # "IsingZZ", # Not supported by Cirq
        # "IsingYY", # Not supported by Cirq
        # "IsingXX", # Not supported by Cirq
    }

    # Qiskit observables: https://github.com/PennyLaneAI/pennylane-qiskit/blob/v0.28.0/pennylane_qiskit/qiskit_device.py#L116
    # Cirq observables: https://github.com/PennyLaneAI/pennylane-cirq/blob/v0.28.0/pennylane_cirq/cirq_device.py#L174
    observables = {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Identity",
        "Hadamard",
        # "Hermitian", # Not supported by Cirq
        "Projector",
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
        return prob


    @classmethod
    def capability(cls) -> Dict[str, Any]:
        return {
            **super().capability(),
            "model": "qubit",
            "tensor_observables": True,
            "inverse_operations": True,
        }
