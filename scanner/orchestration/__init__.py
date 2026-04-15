"""scanner.orchestration — State machine and main scan loop.

Exports:
    ScannerState: enum of all valid scanner states.
    StateMachine: finite-state machine enforcing valid transitions.
    run_scan: execute a full 3D scan and return the exported file path.
"""

from scanner.orchestration.scan import run_scan
from scanner.orchestration.state_machine import ScannerState, StateMachine

__all__ = ["ScannerState", "StateMachine", "run_scan"]
