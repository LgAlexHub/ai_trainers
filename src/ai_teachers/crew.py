from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import PDFSearchTool, FileReadTool, FileWriterTool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Type
import os
from pypdf import PdfReader
import re

class PDFContentInput(BaseModel):
    """Input schema for PDFContentTool."""
    pdf_path: str = Field(..., description="Chemin vers le fichier PDF à analyser")

class PDFContentTool(BaseTool):
    name: str = "PDF Content Extractor"
    description: str = "Extrait et traite le contenu textuel d'un fichier PDF pour analyse pédagogique"
    args_schema: Type[BaseModel] = PDFContentInput

    def _run(self, pdf_path: str) -> str:
        """Exécute l'extraction du contenu PDF"""
        try:
            with open("C:\\Users\\alexl\\Documents\\4_CODE\\7_PYTHON\\crew\\ai_teachers\\src\\ai_teachers\\Référentiel_Activités_Compétences_Evaluation_TP_DWWM.pdf", "rb") as f:
                pdf = PdfReader(f)
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
                processed_text = re.sub(r"\s+", " ", text).strip()
                return processed_text
        except Exception as e:
            return f"Erreur lors de la lecture du PDF: {str(e)}"

@CrewBase
class AiTeachers():
    """Crew optimisé pour l'analyse de PDF pédagogiques"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Configuration du modèle
    llm = LLM(
        model='ollama/llama3.2:3b',
        base_url='http://127.0.0.1:11434',
        timeout=300,
    )
    
    # Chemin du PDF à analyser
    PDF_PATH = "C:\\Users\\alexl\\Documents\\4_CODE\\7_PYTHON\\crew\\ai_teachers\\src\\ai_teachers\\Référentiel_Activités_Compétences_Evaluation_TP_DWWM.pdf"

    @agent
    def pdf_analyst(self) -> Agent:
        """Agent spécialisé dans l'analyse de PDF avec configuration Ollama forcée"""
        return Agent(
            config=self.agents_config['pdf_analyst'],
            tools=[
                PDFContentTool(pdf=self.PDF_PATH),
            ],
            verbose=True,
            llm=self.llm,
            max_iter=3,
            allow_delegation=False
        )

    @agent
    def content_synthesizer(self) -> Agent:
        """Agent spécialisé dans la synthèse et structuration"""
        return Agent(
            config=self.agents_config['content_synthesizer'],
            tools=[FileWriterTool()],
            verbose=True,
            llm=self.llm,
            max_iter=2,
            allow_delegation=False
        )

    @agent
    def quality_reviewer(self) -> Agent:
        """Agent pour validation et amélioration du contenu"""
        return Agent(
            config=self.agents_config['quality_reviewer'],
            tools=[FileReadTool(), FileWriterTool()],
            verbose=True,
            llm=self.llm,
            max_iter=2
        )

    @task
    def pdf_extraction_task(self) -> Task:
        """Tâche d'extraction ciblée du PDF"""
        return Task(
            config=self.tasks_config['pdf_extraction_task'],
            agent=self.pdf_analyst(),
            output_file="extracted_content.md"
        )

    @task
    def content_analysis_task(self) -> Task:
        """Tâche d'analyse approfondie du contenu"""
        return Task(
            config=self.tasks_config['content_analysis_task'],
            agent=self.pdf_analyst(),
            context=[self.pdf_extraction_task()],
            output_file="analyzed_content.md"
        )

    @task
    def synthesis_task(self) -> Task:
        """Tâche de synthèse finale"""
        return Task(
            config=self.tasks_config['synthesis_task'],
            agent=self.content_synthesizer(),
            context=[self.pdf_extraction_task(), self.content_analysis_task()],
            output_file="./output/rapport_final.md"
        )

    @task
    def quality_check_task(self) -> Task:
        """Tâche de validation qualité"""
        return Task(
            config=self.tasks_config['quality_check_task'],
            agent=self.quality_reviewer(),
            context=[self.synthesis_task()],
            output_file="./output/rapport_valide.md"
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True
        )