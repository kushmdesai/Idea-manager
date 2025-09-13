# planning_agent.py
from strands import Agent, tool
from strands.models.anthropic import AnthropicModel
from strands_tools import calculator, current_time
import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
import re


@dataclass
class Task:
    """Individual task in the roadmap"""
    id: str
    title: str
    description: str
    agent_type: str  # Which agent should handle this
    estimated_hours: float
    dependencies: List[str]  # Task IDs this depends on
    deliverables: List[str]
    priority: str  # "high", "medium", "low"
    complexity: str  # "simple", "moderate", "complex"
    tools_needed: List[str]
    acceptance_criteria: List[str]
    status: str = "pending"  # "pending", "in_progress", "completed", "blocked"


@dataclass
class Phase:
    """Project phase containing multiple tasks"""
    id: str
    name: str
    description: str
    tasks: List[Task]
    estimated_duration: str
    success_criteria: List[str]
    risk_factors: List[str]


@dataclass
class ProjectRoadmap:
    """Complete project roadmap"""
    id: str
    project_name: str
    description: str
    phases: List[Phase]
    total_estimated_hours: float
    required_agents: List[str]
    technology_stack: List[str]
    risk_assessment: Dict[str, str]
    success_metrics: List[str]
    created_at: str
    last_updated: str


class PlanningAgent:
    """
    AI Planning Agent that creates detailed, executable roadmaps
    for multi-agent systems to follow
    """
    
    def __init__(self):
        load_dotenv()
        self.model = self._setup_model()
        self.tools = self._setup_tools()
        self.agent_capabilities = self._define_agent_capabilities()
        
    def _setup_model(self):
        """Initialize the Anthropic model optimized for planning"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            
        return AnthropicModel(
            client_args={"api_key": api_key},
            max_tokens=4096,  # Higher limit for detailed planning
            model_id="claude-sonnet-4-20250514",
            params={"temperature": 0.2}  # Lower temp for more consistent planning
        )
    
    def _define_agent_capabilities(self):
        """Define what each agent type can do"""
        return {
            "researcher": {
                "capabilities": ["github_analysis", "paper_search", "tech_trends", "competitive_analysis"],
                "max_concurrent_tasks": 3,
                "avg_hours_per_task": 1.5
            },
            "code_generator": {
                "capabilities": ["frontend_code", "backend_code", "api_development", "database_setup"],
                "max_concurrent_tasks": 2,
                "avg_hours_per_task": 3.0
            },
            "designer": {
                "capabilities": ["ui_design", "wireframes", "prototyping", "user_experience"],
                "max_concurrent_tasks": 2,
                "avg_hours_per_task": 2.5
            },
            "qa_tester": {
                "capabilities": ["unit_testing", "integration_testing", "user_testing", "bug_fixing"],
                "max_concurrent_tasks": 4,
                "avg_hours_per_task": 2.0
            },
            "devops": {
                "capabilities": ["deployment", "ci_cd", "monitoring", "infrastructure"],
                "max_concurrent_tasks": 2,
                "avg_hours_per_task": 2.5
            },
            "project_manager": {
                "capabilities": ["coordination", "progress_tracking", "risk_management", "communication"],
                "max_concurrent_tasks": 5,
                "avg_hours_per_task": 1.0
            }
        }
    
    def _setup_tools(self):
        """Initialize planning-specific tools"""
        
        @tool
        def task_dependency_analyzer(tasks: List[Dict]) -> Dict:
            """
            Analyze task dependencies and identify potential bottlenecks
            
            Args:
                tasks: List of task dictionaries
                
            Returns:
                Analysis of dependencies and critical path
            """
            try:
                if not tasks:
                    return {"error": "No tasks provided"}
                
                # Build dependency graph
                dependency_graph = {}
                all_task_ids = set()
                
                for task in tasks:
                    task_id = task.get('id', '')
                    dependencies = task.get('dependencies', [])
                    
                    all_task_ids.add(task_id)
                    dependency_graph[task_id] = {
                        'dependencies': dependencies,
                        'title': task.get('title', ''),
                        'estimated_hours': task.get('estimated_hours', 0)
                    }
                
                # Find critical path (simplified)
                critical_path = []
                total_critical_hours = 0
                
                # Find tasks with no dependencies (starting points)
                starting_tasks = [tid for tid, info in dependency_graph.items() 
                                if not info['dependencies']]
                
                # Find tasks with most dependencies (potential bottlenecks)
                bottleneck_tasks = sorted(
                    dependency_graph.items(),
                    key=lambda x: len(x[1]['dependencies']),
                    reverse=True
                )[:3]
                
                return {
                    "total_tasks": len(tasks),
                    "starting_tasks": starting_tasks,
                    "potential_bottlenecks": [
                        {"id": task_id, "title": info['title'], "dependency_count": len(info['dependencies'])}
                        for task_id, info in bottleneck_tasks
                    ],
                    "recommendations": [
                        "Start with tasks that have no dependencies",
                        "Assign high-priority resources to bottleneck tasks",
                        "Consider parallel execution where possible"
                    ]
                }
                
            except Exception as e:
                return {"error": f"Dependency analysis failed: {str(e)}"}
        
        @tool
        def estimate_project_timeline(phases: List[Dict], team_size: int = 3) -> Dict:
            """
            Estimate realistic project timeline based on phases and team size
            
            Args:
                phases: List of project phases
                team_size: Number of agents/team members
                
            Returns:
                Timeline estimation with milestones
            """
            try:
                total_hours = 0
                timeline = []
                
                for phase in phases:
                    phase_hours = 0
                    for task in phase.get('tasks', []):
                        phase_hours += task.get('estimated_hours', 0)
                    
                    # Account for parallel work and team efficiency
                    parallel_factor = min(team_size, len(phase.get('tasks', [])))
                    if parallel_factor > 1:
                        adjusted_hours = phase_hours / (parallel_factor * 0.8)  # 80% efficiency
                    else:
                        adjusted_hours = phase_hours
                    
                    total_hours += adjusted_hours
                    
                    timeline.append({
                        "phase": phase.get('name', ''),
                        "estimated_hours": adjusted_hours,
                        "estimated_days": adjusted_hours / 8,  # 8 hour work days
                        "tasks_count": len(phase.get('tasks', []))
                    })
                
                # Calculate working days (excluding weekends)
                total_work_days = total_hours / (8 * team_size)
                calendar_days = total_work_days * 1.4  # Account for weekends
                
                return {
                    "total_estimated_hours": round(total_hours, 1),
                    "total_work_days": round(total_work_days, 1),
                    "calendar_days": round(calendar_days, 1),
                    "team_size": team_size,
                    "phases_timeline": timeline,
                    "milestones": [
                        f"Phase {i+1}: {phase['phase']} - Day {sum(p['estimated_days'] for p in timeline[:i+1]):.1f}"
                        for i, phase in enumerate(timeline)
                    ]
                }
                
            except Exception as e:
                return {"error": f"Timeline estimation failed: {str(e)}"}
        
        @tool
        def risk_assessor(project_description: str, technology_stack: List[str]) -> Dict:
            """
            Assess potential risks and mitigation strategies
            
            Args:
                project_description: Description of the project
                technology_stack: Technologies being used
                
            Returns:
                Risk assessment with mitigation strategies
            """
            try:
                # Common risk categories
                risks = {
                    "technical": [],
                    "timeline": [],
                    "resource": [],
                    "integration": []
                }
                
                # Analyze based on project type and tech stack
                project_lower = project_description.lower()
                
                # Technical risks
                if "real-time" in project_lower or "websocket" in project_lower:
                    risks["technical"].append({
                        "risk": "Real-time synchronization challenges",
                        "probability": "medium",
                        "impact": "high",
                        "mitigation": "Use proven WebSocket libraries, implement proper error handling"
                    })
                
                if "ai" in project_lower or "machine learning" in project_lower:
                    risks["technical"].append({
                        "risk": "AI model performance and accuracy",
                        "probability": "high",
                        "impact": "medium",
                        "mitigation": "Use pre-trained models, implement fallback mechanisms"
                    })
                
                # Timeline risks
                risks["timeline"].append({
                    "risk": "Feature creep during development",
                    "probability": "high",
                    "impact": "medium",
                    "mitigation": "Stick to MVP, document future features separately"
                })
                
                # Resource risks
                risks["resource"].append({
                    "risk": "Agent coordination complexity",
                    "probability": "medium",
                    "impact": "medium",
                    "mitigation": "Clear task definitions, regular progress checks"
                })
                
                # Integration risks
                if len(technology_stack) > 3:
                    risks["integration"].append({
                        "risk": "Complex technology integration",
                        "probability": "medium",
                        "impact": "high",
                        "mitigation": "Prototype integrations early, use proven combinations"
                    })
                
                return {
                    "risk_categories": risks,
                    "overall_risk_level": "medium",
                    "top_risks": [
                        "Feature creep affecting timeline",
                        "Agent coordination complexity",
                        "Technology integration challenges"
                    ],
                    "risk_mitigation_strategy": [
                        "Implement regular checkpoint reviews",
                        "Maintain clear agent communication protocols",
                        "Build and test integrations incrementally"
                    ]
                }
                
            except Exception as e:
                return {"error": f"Risk assessment failed: {str(e)}"}
        
        @tool
        def roadmap_validator(roadmap_data: Dict) -> Dict:
            """
            Validate roadmap completeness and feasibility
            
            Args:
                roadmap_data: Complete roadmap data structure
                
            Returns:
                Validation results with recommendations
            """
            try:
                issues = []
                recommendations = []
                score = 100
                
                # Check required fields
                required_fields = ['project_name', 'phases', 'technology_stack']
                for field in required_fields:
                    if not roadmap_data.get(field):
                        issues.append(f"Missing required field: {field}")
                        score -= 15
                
                # Check phases
                phases = roadmap_data.get('phases', [])
                if not phases:
                    issues.append("No phases defined")
                    score -= 30
                else:
                    for phase in phases:
                        if not phase.get('tasks'):
                            issues.append(f"Phase '{phase.get('name', 'Unknown')}' has no tasks")
                            score -= 10
                
                # Check for balanced workload
                total_tasks = sum(len(phase.get('tasks', [])) for phase in phases)
                if total_tasks < 5:
                    recommendations.append("Consider breaking down tasks further for better tracking")
                elif total_tasks > 50:
                    recommendations.append("Consider grouping tasks to reduce complexity")
                
                # Check timeline reasonableness
                total_hours = roadmap_data.get('total_estimated_hours', 0)
                if total_hours > 200:
                    recommendations.append("Timeline seems ambitious - consider reducing scope or extending deadline")
                elif total_hours < 20:
                    recommendations.append("Project might be underscoped - ensure all tasks are captured")
                
                return {
                    "validation_score": max(score, 0),
                    "issues": issues,
                    "recommendations": recommendations,
                    "is_valid": len(issues) == 0,
                    "completeness_level": "high" if score > 80 else "medium" if score > 60 else "low"
                }
                
            except Exception as e:
                return {"error": f"Roadmap validation failed: {str(e)}"}
        
        return [
            calculator,
            current_time,
            task_dependency_analyzer,
            estimate_project_timeline,
            risk_assessor,
            roadmap_validator
        ]
    
    def create_roadmap(self, project_description: str, research_context: str = None, 
                      constraints: Dict = None) -> ProjectRoadmap:
        """
        Create a comprehensive project roadmap
        
        Args:
            project_description: What to build
            research_context: Research findings from research agent
            constraints: Time, resource, or other constraints
            
        Returns:
            Complete ProjectRoadmap object
        """
        try:
            # Build comprehensive prompt
            planning_prompt = self._build_planning_prompt(
                project_description, research_context, constraints
            )
            
            # Create planning agent
            agent = Agent(model=self.model, tools=self.tools)
            
            # Get roadmap from agent
            roadmap_response = agent(planning_prompt)
            
            # Parse and structure the response
            roadmap = self._parse_roadmap_response(roadmap_response, project_description)
            
            return roadmap
            
        except Exception as e:
            # Return error roadmap
            return ProjectRoadmap(
                id=str(uuid.uuid4()),
                project_name=project_description,
                description=f"Error creating roadmap: {str(e)}",
                phases=[],
                total_estimated_hours=0,
                required_agents=[],
                technology_stack=[],
                risk_assessment={"error": str(e)},
                success_metrics=[],
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat()
            )
    
    def _build_planning_prompt(self, project_description: str, research_context: str, 
                              constraints: Dict) -> str:
        """Build comprehensive planning prompt"""
        
        agent_info = json.dumps(self.agent_capabilities, indent=2)
        
        prompt = f"""
        You are an expert AI Project Planning Agent. Create a detailed, executable roadmap for other AI agents to follow.

        PROJECT TO BUILD: {project_description}

        RESEARCH CONTEXT:
        {research_context if research_context else "No research context provided"}

        AVAILABLE AGENT TYPES AND CAPABILITIES:
        {agent_info}

        CONSTRAINTS:
        {json.dumps(constraints or {}, indent=2)}

        CREATE A COMPREHENSIVE ROADMAP INCLUDING:

        1. PROJECT ANALYSIS:
           - Break down the project into logical phases
           - Identify required technologies and tools
           - Determine which agent types are needed

        2. DETAILED PHASES:
           For each phase, create:
           - Clear phase name and description
           - 5-15 specific, actionable tasks
           - Task assignments to appropriate agent types
           - Realistic time estimates (in hours)
           - Task dependencies
           - Deliverables and acceptance criteria

        3. RISK ASSESSMENT:
           - Use the risk_assessor tool to identify potential risks
           - Provide mitigation strategies

        4. TIMELINE ESTIMATION:
           - Use estimate_project_timeline tool for realistic timelines
           - Account for agent capabilities and parallel work

        5. VALIDATION:
           - Use roadmap_validator tool to check completeness

        TASK BREAKDOWN GUIDELINES:
        - Tasks should be 2-8 hours each (not too big, not too small)
        - Each task should have clear deliverables
        - Assign tasks to agents based on their capabilities
        - Include testing and integration tasks
        - Consider user feedback and iteration cycles

        AGENT ASSIGNMENT RULES:
        - researcher: Market analysis, tech research, competitive analysis
        - code_generator: All coding tasks (frontend, backend, APIs)
        - designer: UI/UX design, wireframes, prototypes
        - qa_tester: All testing activities, bug fixes
        - devops: Deployment, CI/CD, infrastructure
        - project_manager: Coordination, progress tracking, documentation

        Return a structured analysis that I can parse into a proper roadmap format.
        """
        
        return prompt
    
    def _parse_roadmap_response(self, response: str, project_description: str) -> ProjectRoadmap:
        """Parse agent response into structured roadmap"""
        
        try:
            # This is a simplified parser - in production you'd want more robust parsing
            roadmap_id = str(uuid.uuid4())
            
            # Extract phases using simple pattern matching
            phases = []
            phase_pattern = r"Phase \d+:?\s*([^\n]+)"
            phase_matches = re.findall(phase_pattern, response, re.IGNORECASE)
            
            for i, phase_name in enumerate(phase_matches[:6]):  # Limit to 6 phases
                # Create sample tasks for demo purposes
                sample_tasks = self._generate_sample_tasks(phase_name, project_description, i)
                
                phase = Phase(
                    id=f"phase_{i+1}",
                    name=phase_name.strip(),
                    description=f"Phase {i+1} of the project focusing on {phase_name.lower()}",
                    tasks=sample_tasks,
                    estimated_duration=f"{sum(task.estimated_hours for task in sample_tasks):.1f} hours",
                    success_criteria=[f"Complete all tasks in {phase_name}"],
                    risk_factors=["Timeline delays", "Resource constraints"]
                )
                phases.append(phase)
            
            # Calculate totals
            total_hours = sum(
                sum(task.estimated_hours for task in phase.tasks) 
                for phase in phases
            )
            
            # Extract technology stack (simple keyword matching)
            tech_keywords = ["react", "python", "node", "docker", "aws", "mongodb", "postgresql"]
            found_tech = [tech for tech in tech_keywords 
                         if tech.lower() in response.lower()]
            
            return ProjectRoadmap(
                id=roadmap_id,
                project_name=project_description,
                description=f"AI-generated roadmap for {project_description}",
                phases=phases,
                total_estimated_hours=total_hours,
                required_agents=["researcher", "code_generator", "qa_tester", "devops"],
                technology_stack=found_tech or ["python", "react"],
                risk_assessment={
                    "overall": "medium",
                    "main_risks": ["timeline", "integration", "complexity"]
                },
                success_metrics=[
                    "All phases completed on time",
                    "Working prototype delivered",
                    "User acceptance criteria met"
                ],
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            # Fallback roadmap
            return self._create_fallback_roadmap(project_description, str(e))
    
    def _generate_sample_tasks(self, phase_name: str, project_description: str, phase_index: int) -> List[Task]:
        """Generate sample tasks for a phase"""
        
        task_templates = {
            0: [  # Research/Planning phase
                ("Research similar projects", "researcher", 2.0, ["github_analysis"]),
                ("Analyze technical requirements", "researcher", 1.5, ["competitive_analysis"]),
                ("Create technical specification", "project_manager", 2.0, ["documentation"])
            ],
            1: [  # Design phase
                ("Create wireframes", "designer", 3.0, ["ui_design"]),
                ("Design user interface", "designer", 4.0, ["prototyping"]),
                ("Review and iterate design", "designer", 2.0, ["user_experience"])
            ],
            2: [  # Development phase
                ("Set up project structure", "code_generator", 2.0, ["backend_code"]),
                ("Implement core features", "code_generator", 6.0, ["frontend_code"]),
                ("Create API endpoints", "code_generator", 4.0, ["api_development"])
            ],
            3: [  # Testing phase
                ("Write unit tests", "qa_tester", 3.0, ["unit_testing"]),
                ("Perform integration testing", "qa_tester", 4.0, ["integration_testing"]),
                ("User acceptance testing", "qa_tester", 2.0, ["user_testing"])
            ],
            4: [  # Deployment phase
                ("Set up CI/CD pipeline", "devops", 3.0, ["ci_cd"]),
                ("Deploy to staging", "devops", 2.0, ["deployment"]),
                ("Production deployment", "devops", 2.0, ["deployment"])
            ]
        }
        
        templates = task_templates.get(phase_index, task_templates[2])  # Default to dev tasks
        
        tasks = []
        for i, (title, agent, hours, tools) in enumerate(templates):
            task = Task(
                id=f"task_{phase_index}_{i}",
                title=title,
                description=f"{title} for {project_description}",
                agent_type=agent,
                estimated_hours=hours,
                dependencies=[f"task_{phase_index}_{i-1}"] if i > 0 else [],
                deliverables=[f"{title} completed"],
                priority="high" if i < 2 else "medium",
                complexity="moderate",
                tools_needed=tools,
                acceptance_criteria=[f"{title} meets requirements"]
            )
            tasks.append(task)
        
        return tasks
    
    def _create_fallback_roadmap(self, project_description: str, error: str) -> ProjectRoadmap:
        """Create a basic fallback roadmap when parsing fails"""
        
        basic_phases = [
            Phase(
                id="phase_1",
                name="Research & Planning",
                description="Initial research and project planning",
                tasks=[],
                estimated_duration="8 hours",
                success_criteria=["Research completed", "Plan approved"],
                risk_factors=["Scope creep"]
            ),
            Phase(
                id="phase_2", 
                name="Development",
                description="Core development work",
                tasks=[],
                estimated_duration="24 hours",
                success_criteria=["Core features implemented"],
                risk_factors=["Technical complexity"]
            ),
            Phase(
                id="phase_3",
                name="Testing & Deployment",
                description="Testing and production deployment",
                tasks=[],
                estimated_duration="8 hours", 
                success_criteria=["All tests pass", "Successfully deployed"],
                risk_factors=["Integration issues"]
            )
        ]
        
        return ProjectRoadmap(
            id=str(uuid.uuid4()),
            project_name=project_description,
            description=f"Basic roadmap for {project_description} (fallback due to parsing error: {error})",
            phases=basic_phases,
            total_estimated_hours=40.0,
            required_agents=["researcher", "code_generator", "qa_tester"],
            technology_stack=["python", "react"],
            risk_assessment={"error": error, "fallback": True},
            success_metrics=["Project completed", "Requirements met"],
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
    
    def export_roadmap(self, roadmap: ProjectRoadmap, format: str = "json") -> str:
        """Export roadmap in various formats"""
        
        if format == "json":
            return json.dumps(asdict(roadmap), indent=2)
        elif format == "markdown":
            return self._roadmap_to_markdown(roadmap)
        else:
            return str(roadmap)
    
    def _roadmap_to_markdown(self, roadmap: ProjectRoadmap) -> str:
        """Convert roadmap to markdown format"""
        
        md = f"""# {roadmap.project_name}

## Project Overview
{roadmap.description}

**Total Estimated Hours:** {roadmap.total_estimated_hours}
**Required Agents:** {', '.join(roadmap.required_agents)}
**Technology Stack:** {', '.join(roadmap.technology_stack)}

## Phases

"""
        for phase in roadmap.phases:
            md += f"### {phase.name}\n"
            md += f"{phase.description}\n\n"
            md += f"**Duration:** {phase.estimated_duration}\n\n"
            
            if phase.tasks:
                md += "#### Tasks:\n"
                for task in phase.tasks:
                    md += f"- **{task.title}** ({task.estimated_hours}h) - {task.agent_type}\n"
                    md += f"  - {task.description}\n"
                md += "\n"
        
        return md


# Convenience function for easy import
def project_planner(project_description: str, context: Dict = None) -> Dict:
    """
    Simple function interface for the planning agent
    
    Args:
        project_description: What to build
        context: Additional context (research data, constraints, etc.)
        
    Returns:
        Roadmap as dictionary
    """
    try:
        agent = PlanningAgent()
        research_context = context.get('research', '') if context else ''
        constraints = context.get('constraints', {}) if context else {}
        
        roadmap = agent.create_roadmap(project_description, research_context, constraints)
        return asdict(roadmap)
        
    except Exception as e:
        return {"error": f"Planning failed: {str(e)}"}


# Example usage and testing
if __name__ == "__main__":
    # Test the planning agent
    planner = PlanningAgent()
    
    test_projects = [
        "AI-powered task management app with real-time collaboration",
        "E-commerce platform with recommendation engine",
        "Social media dashboard with analytics"
    ]
    
    for project in test_projects:
        print(f"\n{'='*60}")
        print(f"CREATING ROADMAP FOR: {project}")
        print(f"{'='*60}")
        
        roadmap = planner.create_roadmap(
            project,
            research_context="Sample research data about similar projects",
            constraints={"timeline": "2 weeks", "team_size": 3}
        )
        
        print(planner.export_roadmap(roadmap, "markdown"))
        print("\n")