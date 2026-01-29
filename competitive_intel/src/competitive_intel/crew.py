from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from typing import List


@CrewBase
class CompetitiveIntel():
    """Competitive Intelligence Monitor crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def trend_scanner(self) -> Agent:
        return Agent(
            config=self.agents_config['trend_scanner'],  # type: ignore[index]
            tools=[SerperDevTool()],
            verbose=True
        )

    @agent
    def company_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['company_analyst'],  # type: ignore[index]
            verbose=True
        )

    @agent
    def strategy_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config['strategy_advisor'],  # type: ignore[index]
            verbose=True
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['report_writer'],  # type: ignore[index]
            verbose=True
        )

    @task
    def scan_competitors(self) -> Task:
        return Task(
            config=self.tasks_config['scan_competitors'],  # type: ignore[index]
        )

    @task
    def analyze_findings(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_findings'],  # type: ignore[index]
        )

    @task
    def strategic_recommendations(self) -> Task:
        return Task(
            config=self.tasks_config['strategic_recommendations'],  # type: ignore[index]
        )

    @task
    def write_briefing(self) -> Task:
        return Task(
            config=self.tasks_config['write_briefing'],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CompetitiveIntel crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
