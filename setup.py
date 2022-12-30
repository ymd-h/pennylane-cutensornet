from setuptools import setup, find_packages

setup(name="pennylane-cutensornet",
      description="PennyLane Plugin for cuTensorNet of NVIDIA cuQuantum",
      version="0.0.0",
      packages=find_packages(),
      install_require = ["numpy", "pennylane", "cuquantum-python-cu11"],
      extra_requires = {
          "qiskit": "pennylane-qiskit",
          "cirq": "pennylane-cirq"
      },
      entrypoints={"pennylane.plugins": [
          "cuquantum.cutensornet = pennylane_cutensornet.cuTensorNetDevice"
      ]})
