from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import SerperDevTool
from crewai.tools import PDFSearchTool
# from tools.custom_tool import DocumentSearchTool
from agentic_rag.tools.custom_tool import DocumentSearchTool

# Initialize the tool with a specific PDF path for exclusive search within that document
pdf_tool = DocumentSearchTool(pdf='knowledge/demo.pdf')
web_search_tool = SerperDevTool()

@CrewBase
class AgenticRag():
	"""AgenticRag crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	# @agent
	# def routing_agent(self) -> Agent:
	# 	return Agent(
	# 		config=self.agents_config['routing_agent'],
	# 		verbose=True
	# 	)

	@agent
	def retriever_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['retriever_agent'],
			verbose=True,
			tools=[
				pdf_tool,
				web_search_tool
			]
		)

	@agent
	def response_synthesizer_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['response_synthesizer_agent'],
			verbose=True
		)

	@task
	def retrieval_task(self) -> Task:
		return Task(
			config=self.tasks_config['retrieval_task'],
		)

	@task
	def response_task(self) -> Task:
		return Task(
			config=self.tasks_config['response_task'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the AgenticRag crew"""


		return Crew(
			agents=self.agents, 
			tasks=self.tasks, 
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical,
		)
