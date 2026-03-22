# master execution file
import os

print('Operator definition')
os.system('python ./tests/operators.py')

print('Consistency of quantization Sub-rountine')
os.system('python ./tests/consistency/backend.py')
os.system('python ./tests/consistency/basis.py')
os.system('python ./tests/consistency/eigensolvers.py')
os.system('python ./tests/consistency/precision.py')

print('Comparison : DeS-SCQ v/s scQubits')
os.system('python ./tests/comparison/scQ-fluxonium.py')
os.system('python ./tests/comparison/scQ-transmon.py')
os.system('python ./tests/comparison/scQ-flux-shunted.py')

print('Optimization')
os.system('python ./tests/optimization/spectrum-tuning.py')
os.system('python ./tests/optimization/box-qutrit-projection.py')
os.system('python ./tests/optimization/initialization-parallelism.py')