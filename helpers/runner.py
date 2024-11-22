import time
from collections import Counter

from negmas.helpers import humanize_time
from rich import print
from helpers.seasonBetterSyncAgent import BetterSyncAgent   # agent we want to compare to 
from scml.std.agents import SyncRandomStdAgent
from scml.utils import anac2024_oneshot, anac2024_std
from tabulate import tabulate
from or_dist_redist_agent import DistRedistAgent        # agent we want to compare to 


def run_multiple(
    competitors=tuple(),
    competition="oneshot",
    reveal_names=True,
    n_steps=10,
    n_configs=2,
    n_runs=7,  # Number of runs
):
    """
    Runs multiple simulations and determines the agent with the most wins.

    Args:
        competitors: A list of competitor classes
        competition: The competition type to run (possibilities are oneshot, std).
        n_steps:     The number of simulation steps.
        n_configs:   Number of different world configurations to try.
                     Different world configurations will correspond to
                     different number of factories, profiles, production graphs etc.
        n_runs:      The number of runs to perform.

    Returns:
        None
    """
    # agents with seasonal modifications
    competitors = list(competitors) + [BetterSyncAgent, DistRedistAgent]

    win_counts = Counter()

    start = time.perf_counter()
    for run_id in range(1, n_runs + 1):
        print(f"\n[bold green]Run {run_id}/{n_runs}[/bold green]")
        if competition == "std":
            runner = anac2024_std
        else:
            runner = anac2024_oneshot

        results = runner(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
        )
        # just make names shorter
        results.total_scores.agent_type = results.total_scores.agent_type.str.split(  # type: ignore
            "."
        ).str[
            -1
        ]
        # determine the winner of this run
        top_score = results.total_scores["score"].max()
        winner = results.total_scores.loc[
            results.total_scores["score"] == top_score, "agent_type"
        ].iloc[0]
        win_counts[winner] += 1

        # display results for this run
        print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))  # type: ignore

    # Display final win counts
    print("\n[bold blue]Final Results After All Runs[/bold blue]")
    print(tabulate(win_counts.items(), headers=["Agent Type", "Wins"], tablefmt="psql"))
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    import sys

    run_multiple(
        sys.argv[1] if len(sys.argv) > 1 else "oneshot"  #Number of runs
    )
