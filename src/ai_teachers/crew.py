from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool,
    ScrapeWebsiteTool,
    FileWriterTool
)
from typing import List

# https://docs.crewai.com/concepts/agents#agent-tools
# https://docs.crewai.com/concepts/tasks#overview-of-a-task
# https://docs.crewai.com/concepts/knowledge#what-is-knowledge
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/


@CrewBase
class AiTeachers():
    """AiTeachers crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    llm = LLM(
        model='ollama/deepseek-r1:8b',
        # model='ollama/llama3.2:3b',
        base_url='http://127.0.0.1:11434'
    )

    @agent
    def web_retreiver(self) -> Agent:
        return Agent(
            config=self.agents_config['web_retreiver'],
            tools=[SerperDevTool()],
            verbose=True,
            llm=self.llm
        )

    @agent
    def web_scrapper(self) -> Agent:
        return Agent(
            config=self.agents_config['web_scrapper'],
            tools=[ScrapeWebsiteTool()],
            verbose=True,
            llm=self.llm
        )

    @agent
    def web_content_writter(self) -> Agent:
        return Agent(
            config=self.agents_config['web_content_writter'],
            tools=[],
            verbose=True,
            llm=self.llm
        )

    @agent
    def file_writter(self) -> Agent:
        return Agent(
            config=self.agents_config['file_writter'],
            tools=[FileWriterTool()],
            llm=self.llm

        )

    @task
    def retreive_content_task(self) -> Task:
        return Task(
            config=self.tasks_config['retreive_content_task'],
        )

    @task
    def web_scrap_task(self) -> Task:
        return Task(
            config=self.tasks_config['web_scrap_task'],
        )

    @task
    def write_web_content_task(self) -> Task:
        return Task(
            config=self.tasks_config['write_web_content_task'],
        )

    @task
    def write_file_task(self) -> Task:
        return Task(
            config=self.tasks_config['write_file_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AiTeachers crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
