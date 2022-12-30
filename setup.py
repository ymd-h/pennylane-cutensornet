from setuptools import setup, find_packages

setup(name="pennylane-cutensornet",
      description="PennyLane Plugin for cuTensorNet of NVIDIA cuQuantum",
      version="0.0.0",
      packages=find_packages(),
      install_requires = ["numpy", "pennylane", "cuquantum-python-cu11"],
      extras_require = {
          "qiskit": ["pennylane-qiskit"],
          "cirq": ["pennylane-cirq"],
          "test": ["pytest", "pytest_mock", "flaky"],
      },
      entrypoints={"pennylane.plugins": [
          "cuquantum.cutensornet = pennylane_cutensornet.cuTensorNetDevice"
      ]})
