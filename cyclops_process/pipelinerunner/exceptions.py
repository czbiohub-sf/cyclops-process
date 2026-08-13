"""Exceptions the pipeline runner recognises as control-flow signals.

Step functions raise ``PipelineHalted`` to tell the orchestrator to stop the
whole pipeline gracefully (instead of moving on to the next step). Typical
use: a step detects that an upstream artifact it needs is missing and there
is no point letting downstream steps fire blind. The runner catches it,
prints the reason, and ends the pipeline with a non-error exit so the user
isn't presented with a stack trace or a retry prompt.
"""


class PipelineHalted(Exception):
    """Signal: stop the pipeline cleanly from inside a step.

    ``reason`` is a short human-readable message printed by the orchestrator
    when this propagates up.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
