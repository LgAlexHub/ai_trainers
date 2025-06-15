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
        model='ollama/llama3.2:3b',
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
            config=self.agents_configs['web_scrapper'],
            tools=[ScrapeWebsiteTool()],
            verbose=True,
            llm=self.llm
        )
        
    @agent 
    def web_writter(self) -> Agent:
        return Agent(
            config = self.agents_config['web_writter'],
            tools=[FileWriterTool()]
        )
    
    @agent
    def subject_matter_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['subject_matter_expert'],
            verbose=True,
            llm=self.llm
        )
    
    @agent
    def graphist_ux_designer(self) -> Agent:
        return Agent(
            config=self.agents_config['graphist_ux_designer'],
            verbose=True,
            llm = self.llm
        )

    @agent
    def insturctionnal_designer(self) -> Agent:
        return Agent(
            config=self.agents_config['insturctionnal_designer'],
            verbose=True,
            llm = self.llm
        )

    @agent
    def content_writter(self) -> Agent:
        return Agent(
            config=self.agents_config['content_writter'],
            verbose=True,
            llm = self.llm
        )

    @task
    def fetch_the_internet_task(self) -> Task:
        return Task(
            config=self.tasks_config['fetch_the_internet'],
            output_file='internet.md'

        )

    @task
    def master_subject_task(self) -> Task:
        return Task(
            config=self.tasks_config['master_subject'],
            output_file='master.md'

        )
        
    @task
    def front_mastering_task(self) -> Task:
        return Task(
            config=self.tasks_config['front_mastering'],
            output_file='ui.md'

        )
        
    @task
    def harmonize_content_task(self) -> Task:
        return Task(
            config=self.tasks_config['harmonize_content'],
            output_file='enhanced.md'

        )
        
    @task
    def scribbling_task(self) -> Task:
        return Task(
            config=self.tasks_config['scribbling'],
            output_file='final.md'
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
