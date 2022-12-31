# PennyLane Plugin for cuTensorNet of NVIDIA cuQuantum

This repository is 3rd party unofficial PennyLane plugin for
cuTensorNet of NVIDIA cuQuantum.

> **Note**
> The official PennyLane-Lightning-GPU plugin supports cuStateVec of cuQuantum

## Usage

This plugin provides "cuquantum.cutensornet" device.
The device can take optional argument `mode` to specify intermediate circuit.

```python
import pennylane as qml

n_qubits = 3
mode = "cirq" # or "qiskit"
dev = qml.device("cuquantum.cutensornet", wires=n_qubits, mode=mode)
```


## References
- [PennyLane](https://pennylane.ai/) ([GitHub](https://github.com/PennyLaneAI/pennylane))
- [PennyLane-Lightning-GPU Plugin](https://github.com/PennyLaneAI/pennylane-lightning-gpu) ([Blog Post](https://pennylane.ai/blog/2022/07/lightning-fast-simulations-with-pennylane-and-the-nvidia-cuquantum-sdk/))
- [NVIDIA cuQuantum](https://developer.nvidia.com/cuquantum-sdk) ([GitHub](https://github.com/NVIDIA/cuQuantum))
