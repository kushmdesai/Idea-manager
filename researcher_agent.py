# researcher_agent.py
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands_tools import calculator, current_time
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
import requests
import os
import json
import time
from typing import Optional, Dict, List
from dotenv import load_dotenv
import xml.etree.ElementTree as ET


class ResearcherAgent:
    """
    A comprehensive research agent that can analyze GitHub repos, 
    search academic papers, and gather technical information
    """
    
    def __init__(self):
        load_dotenv()
        self.model = self._setup_model()
        self.github_mcp = self._setup_github_mcp()
        self.tools = self._setup_tools()
        
    def _setup_model(self):
        """Initialize the Cohere model"""
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")
            
        return OpenAIModel(
            client_args={"api_key": api_key, "base_url": "https://api.cohere.ai/compatibility/v1"},
            model_id="command-a-03-2025",  # Increased for detailed research
            params={"temperature": 0.3, "stream_options": None, "mak_tokens": 1096}  # Lower temp for more focused research
        )
    
    def _setup_github_mcp(self):
        """Set up GitHub MCP server connection"""
        github_token = os.getenv("GITHUB_PAT_1")
        if not github_token:
            print("Warning: No GitHub token found. GitHub tools will be unavailable.")
            return None
            
        return MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command="podman",
                args=[
                    "run", 
                    "-i", 
                    "--rm",
                    "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
                    "-e", "GITHUB_TOOLSETS=repos,issues,pull_requests,actions,context",
                    "ghcr.io/github/github-mcp-server"
                ],
                env={"GITHUB_PERSONAL_ACCESS_TOKEN": github_token}
            )
        ))
    
    def _setup_tools(self):
        """Initialize all research tools"""
        
        @tool
        def arxiv_paper_search(query: str, max_results: int = 10) -> Dict:
            """
            Search for academic papers on arXiv
            
            Args:
                query: Search terms for papers
                max_results: Maximum number of results to return
                
            Returns:
                Dict containing paper titles, abstracts, authors, and URLs
            """
            try:
                # Clean and format query
                clean_query = query.replace(" ", "+")
                url = f"http://export.arxiv.org/api/query?search_query=all:{clean_query}&max_results={max_results}"
                
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                
                papers = []
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    title = entry.find('{http://www.w3.org/2005/Atom}title').text
                    summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
                    
                    # Get authors
                    authors = []
                    for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                        name = author.find('{http://www.w3.org/2005/Atom}name').text
                        authors.append(name)
                    
                    # Get arXiv ID and URL
                    arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text
                    
                    papers.append({
                        "title": title.strip(),
                        "abstract": summary.strip(),
                        "authors": authors,
                        "url": arxiv_id,
                        "arxiv_id": arxiv_id.split('/')[-1]
                    })
                
                return {
                    "success": True,
                    "papers": papers,
                    "total_found": len(papers)
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error searching arXiv: {str(e)}",
                    "papers": []
                }
        
        @tool
        def tech_trend_analyzer(technology: str) -> Dict:
            """
            Analyze trends and popularity of a technology
            
            Args:
                technology: Name of the technology to analyze
                
            Returns:
                Dict with trend analysis and recommendations
            """
            try:
                # This could be expanded to use GitHub API, Stack Overflow API, etc.
                # For now, providing structured analysis framework
                
                analysis = {
                    "technology": technology,
                    "analysis_type": "github_activity",
                    "recommendations": [
                        f"Research {technology} repositories with high star counts",
                        f"Look for recent commits and active maintenance",
                        f"Check for comprehensive documentation and examples",
                        f"Analyze issue resolution time and community engagement"
                    ],
                    "search_keywords": [
                        technology,
                        f"{technology} tutorial",
                        f"{technology} example",
                        f"{technology} best practices"
                    ]
                }
                
                return analysis
                
            except Exception as e:
                return {"error": f"Error analyzing technology trends: {str(e)}"}
        
        @tool
        def competitive_analysis(project_type: str, feature_focus: str = None) -> Dict:
            """
            Analyze existing projects in a domain to identify patterns and opportunities
            
            Args:
                project_type: Type of project (e.g., "todo app", "weather app", "chat bot")
                feature_focus: Specific features to focus analysis on
                
            Returns:
                Dict with competitive analysis and insights
            """
            try:
                analysis = {
                    "project_type": project_type,
                    "feature_focus": feature_focus,
                    "research_strategy": {
                        "github_search_terms": [
                            project_type,
                            f"{project_type} open source",
                            f"{project_type} example",
                            f"{project_type} template"
                        ],
                        "analysis_criteria": [
                            "Star count and popularity",
                            "Code quality and structure", 
                            "Documentation completeness",
                            "Recent activity and maintenance",
                            "Unique features or approaches"
                        ]
                    },
                    "next_steps": [
                        f"Search GitHub for '{project_type}' repositories",
                        "Analyze top 5-10 repositories by stars",
                        "Document common patterns and unique approaches",
                        "Identify gaps or improvement opportunities"
                    ]
                }
                
                return analysis
                
            except Exception as e:
                return {"error": f"Error in competitive analysis: {str(e)}"}
        
        @tool
        def research_summarizer(research_data: List[Dict]) -> str:
            """
            Summarize research findings into actionable insights
            
            Args:
                research_data: List of research findings from other tools
                
            Returns:
                Formatted summary with key insights and recommendations
            """
            try:
                if not research_data:
                    return "No research data provided to summarize."
                
                summary = "## Research Summary\n\n"
                
                for i, data in enumerate(research_data, 1):
                    summary += f"### Finding {i}\n"
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key != "raw_data":  # Skip raw data in summary
                                summary += f"- **{key.title()}**: {value}\n"
                    else:
                        summary += f"- {str(data)}\n"
                    summary += "\n"
                
                summary += "### Key Recommendations\n"
                summary += "- Prioritize solutions with active community support\n"
                summary += "- Focus on well-documented approaches for faster development\n"
                summary += "- Consider innovative features that differentiate from existing solutions\n"
                
                return summary
                
            except Exception as e:
                return f"Error summarizing research: {str(e)}"
        
        # Base tools
        tools = [
            calculator, 
            current_time, 
            arxiv_paper_search,
            tech_trend_analyzer,
            competitive_analysis,
            research_summarizer
        ]
        
        return tools
    
    def research(self, prompt: str, research_type: str = "comprehensive") -> str:
        """
        Main research function that coordinates different research capabilities
        
        Args:
            prompt: Research question or topic
            research_type: Type of research ("quick", "comprehensive", "academic", "technical")
            
        Returns:
            Research findings and recommendations
        """
        try:
            # Add research type context to the prompt
            context_prompt = f"""
            You are a specialized research agent. Conduct {research_type} research on: {prompt}

            Research Guidelines:
            - Use available tools to gather comprehensive information
            - For GitHub research, analyze popular repositories, recent activity, and code quality
            - For academic research, search relevant papers and summarize key findings
            - For technical research, focus on implementation details and best practices
            - Always provide actionable insights and recommendations
            - Structure your findings clearly with key takeaways

            Research Focus: {prompt}
            """
            
            # Initialize agent with tools
            if self.github_mcp:
                with self.github_mcp:
                    # Get GitHub tools
                    try:
                        github_tools = self.github_mcp.list_tools_sync()
                        print(f"Added {len(github_tools)} GitHub tools")
                        all_tools = self.tools + github_tools
                    except Exception as e:
                        print(f"Warning: Could not load GitHub tools: {e}")
                        all_tools = self.tools
                        
                    # Create and run agent
                    agent = Agent(model=self.model, tools=all_tools)
                    result = agent(context_prompt)
                    return result
            else:
                # Run without GitHub integration
                agent = Agent(model=self.model, tools=self.tools)
                result = agent(context_prompt)
                return result
                
        except Exception as e:
            return f"Research error: {str(e)}"
    
    def quick_research(self, prompt: str) -> str:
        """Quick research for time-sensitive queries"""
        return self.research(prompt, "quick")
    
    def academic_research(self, prompt: str) -> str:
        """Academic-focused research using papers and publications"""
        return self.research(prompt, "academic")
    
    def technical_research(self, prompt: str) -> str:
        """Technical implementation-focused research"""
        return self.research(prompt, "technical")


# Convenience function for easy import and usage
def researcher(prompt: str, research_type: str = "comprehensive") -> str:
    """
    Simple function interface for the researcher agent
    
    Args:
        prompt: What to research
        research_type: Type of research to conduct
        
    Returns:
        Research findings
    """
    try:
        agent = ResearcherAgent()
        return agent.research(prompt, research_type)
    except Exception as e:
        return f"Researcher initialization error: {str(e)}"


# Example usage and testing
if __name__ == "__main__":
    # Test the researcher
    test_queries = [
        "Research React-based dashboard frameworks for data visualization",
        "Find academic papers about multi-agent AI systems",
        "Analyze popular todo list applications on GitHub"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"RESEARCH QUERY: {query}")
        print(f"{'='*50}")
        
        result = researcher(query, "comprehensive")
        print(result)
        print("\n")
        
        # Add delay to avoid rate limiting
        time.sleep(2)