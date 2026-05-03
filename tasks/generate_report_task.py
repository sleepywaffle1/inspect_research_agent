from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import chain

from solvers.pre_scope_solver import pre_scope_solver
from solvers.research_supervisor_solver import research_supervisor_solver
from solvers.generate_report_solver import generate_report_solver


@task
def research_task():
    return Task(
        dataset=[
            Sample(
                input="What are the top coffee shops in San Francisco based on coffee quality? Focus on coffee beans quality and awards. Do not ask anymore clarification questions.",
                target="This is a dummy target for the first sample.",  # or expected final summary
            ),
            Sample(
                input="Compare Tesla vs BYD electric vehicles in 2025. Focus on technical specifications, performance, and market reception. Do not ask anymore clarification questions.",
                target="This is a dummy target for the second sample.",
            ),
        ],
        solver=chain(
            # optional if you already have research brief
            pre_scope_solver(),
            research_supervisor_solver(),
            generate_report_solver(),
        ),
    )