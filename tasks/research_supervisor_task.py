from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import chain

from solvers.research_supervisor_solver import research_supervisor_solver
from solvers.pre_scope_solver import pre_scope_solver

research_brief = """I want to identify and evaluate the coffee shops in San Francisco that are considered the best based specifically  
on coffee quality. My research should focus on analyzing and comparing coffee shops within the San Francisco area, 
using coffee quality as the primary criterion. I am open regarding methods of assessing coffee quality (e.g.,      
expert reviews, customer ratings, specialty coffee certifications), and there are no constraints on ambiance,      
location, wifi, or food options unless they directly impact perceived coffee quality. Please prioritize primary    
sources such as the official websites of coffee shops, reputable third-party coffee review organizations (like     
Coffee Review or Specialty Coffee Association), and prominent review aggregators like Google or Yelp where direct  
customer feedback about coffee quality can be found. The study should result in a well-supported list or ranking of
the top coffee shops in San Francisco, emphasizing their coffee quality according to the latest available data as  
of July 2025."""

# @task
# def research_supervisor_task():
#     return Task(
#         dataset=[
#             Sample(
#                 input="",  # not used
#                 metadata={
#                     "research_brief": research_brief,
#                 },
#                 target=None,
#             ),
#         ],
#         solver=research_supervisor_solver(mode="full"),
#         scorer=match(location="exact"),
#         model="openrouter/openai/gpt-4.1-mini"
#     )

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
        ),
    )