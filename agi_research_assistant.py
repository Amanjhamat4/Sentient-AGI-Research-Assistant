import dspy
from roma_dspy import Aggregator, Atomizer, Executor, Planner, Verifier, SubTask
from roma_dspy.toolkit.serper import SerperToolkit # Web search toolkit


# Load config (general profile for AGI research)
from roma_dspy.config import load_config
config = load_config(profile="general")


# Set up LLMs (using OpenRouter models for cost-efficiency)
executor_lm = dspy.LM("openrouter/openai/gpt-4o-mini", temperature=0.7, cache=True)
atomizer_lm = dspy.LM("openrouter/google/gemini-2.5-flash", temperature=0.6, cache=False)
planner_lm = dspy.LM("openrouter/openai/gpt-4o-mini", temperature=0.85, cache=True)
aggregator_lm = dspy.LM("openrouter/openai/gpt-4o-mini", temperature=0.65, cache=True)
verifier_lm = dspy.LM("openrouter/openai/gpt-4o-mini", temperature=0.0, cache=True)


# Initialize modules with AGI-focused context
context_defaults = {"track_usage": True, "domain": "AGI and open-source AI development"}


atomizer = Atomizer(lm=atomizer_lm, prediction_strategy="cot", context_defaults=context_defaults)
planner = Planner(lm=planner_lm, prediction_strategy="cot", context_defaults=context_defaults)


# Executor with ReAct strategy and web search tool
serper_toolkit = SerperToolkit() # Uses SERPER_API_KEY from .env
executor = Executor(lm=executor_lm, prediction_strategy="react", tools=[serper_toolkit.get_weather], context_defaults=context_defaults) # Add more tools if needed


aggregator = Aggregator(lm=aggregator_lm, prediction_strategy="cot")
verifier = Verifier(lm=verifier_lm)


def run_agi_research(goal: str) -> str:
    """Runs the ROMA pipeline for AGI research queries."""
    atomized = atomizer.forward(goal)
    if atomized.is_atomic or atomized.node_type.is_execute:
        execution = executor.forward(goal)
        candidate = execution.output
    else:
        plan = planner.forward(goal)
        results = []
        for idx, subtask in enumerate(plan.subtasks, start=1):
            sub_result = run_agi_research(subtask.goal) # Recursive call for depth
            results.append(SubTask(goal=subtask.goal, task_type=subtask.task_type, dependencies=subtask.dependencies, result=sub_result))
        aggregated = aggregator.forward(goal, results)
        candidate = aggregated.synthesized_result


    verdict = verifier.forward(goal, candidate)
    if verdict.verdict:
        return candidate
    return f"Verification failed: {verdict.feedback or 'No feedback provided'}.\nCandidate output: {candidate}"


# Example usage
if __name__ == "__main__":
    query = "Analyze the impact of Sentient's ROMA on open AGI development, including pros, cons, and future potential."
    print("Research Query:", query)
    result = run_agi_research(query)
    print("Result:\n", result)