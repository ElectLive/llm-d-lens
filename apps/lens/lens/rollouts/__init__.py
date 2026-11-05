"""In-place rollouts of a stack with one DisaggregatedSet (``llmd rollouts``).

A rollout moves a running instance from the objects it runs to the ones a source ref
renders. When a DisaggregatedSet's pod templates change, its controller (lws) starts a new
revision and replaces groups in place: with ``maxSurge: 0`` and ``maxUnavailable: N`` per
role, each step removes up to N old groups of every role and creates as many new ones.

llmd plans that roll, applies the change, and follows the controller step by step. The
llm-d guides' model servers drain themselves as their pods stop (a ``preStop`` sleep, then
vLLM's ``--shutdown-timeout``), so ``--no-drain`` rolls them. A stack that holds the
controller's scale-down (an admission policy) until a drain tool took a group out of the
pool gives that tool with ``--drain``: llmd runs it for each role while the roll is in
progress, and it is what lets each step proceed.
"""
