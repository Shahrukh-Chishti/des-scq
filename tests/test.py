# master execution file: runs every test script, reports failures, exits non-zero on any
import subprocess
import sys

SUITES = {
    'Operator definition': [
        'tests/operators.py',
    ],
    'Consistency of quantization Sub-rountine': [
        'tests/consistency/basis.py',
        'tests/consistency/eigensolvers.py',
        'tests/consistency/precision.py',
        'tests/consistency/junction-flux.py',
        'tests/consistency/parameter-update.py',
        'tests/consistency/sampling-seed.py',
    ],
    'Comparison : DeS-SCQ v/s scQubits': [
        'tests/comparison/scQ-fluxonium.py',
        'tests/comparison/scQ-transmon.py',
        'tests/comparison/scQ-flux-shunted.py',
        'tests/comparison/scQ-flux-shunted-flux.py',
        'tests/comparison/scQ-kerman-oscillators.py',
    ],
    'Optimization': [
        'tests/optimization/spectrum-tuning.py',
        'tests/optimization/box-qutrit-projection.py',
        'tests/optimization/initialization-parallelism.py',
    ],
}


def main():
    failed = []
    for title, scripts in SUITES.items():
        print(title, flush=True)
        for script in scripts:
            code = subprocess.run([sys.executable, script]).returncode
            if code != 0:
                print(f'FAILED ({code}): {script}', flush=True)
                failed.append(script)
    print()
    if failed:
        print(f'{len(failed)} script(s) failed:', *failed, sep='\n  ')
        sys.exit(1)
    print('All scripts passed')


if __name__ == '__main__':
    main()
