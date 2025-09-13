# agent_coordinator.py - FIXED VERSION
from researcher_agent import researcher
from planning_agent import project_planner, PlanningAgent
import json

class AgentCoordinator:
    def __init__(self):
        self.shared_context = {}
        self.execution_log = []
        self.planner = PlanningAgent()
    
    def _extract_text_from_result(self, result):
        """Extract text from AgentResult or return as-is if already string"""
        if hasattr(result, 'content'):
            return str(result.content)
        elif hasattr(result, 'text'):
            return str(result.text)
        else:
            return str(result)
    
    def build_project(self, project_idea):
        print("🔍 Phase 1: Research & Analysis")
        
        # Step 1: Comprehensive Research
        research_result = researcher(
            f"Research technologies, frameworks, and best practices for: {project_idea}",
            "comprehensive"
        )
        research = self._extract_text_from_result(research_result)
        
        # Step 2: Technical Deep Dive
        tech_result = researcher(
            f"Find technical implementation patterns and architecture recommendations for: {project_idea}",
            "technical"
        )
        tech_analysis = self._extract_text_from_result(tech_result)
        
        # Step 3: Competitive Analysis
        comp_result = researcher(
            f"Analyze existing solutions and identify opportunities for: {project_idea}",
            "quick"
        )
        competitive_analysis = self._extract_text_from_result(comp_result)
        
        # Combine research context
        self.shared_context = {
            'research': research,
            'technical_analysis': tech_analysis,
            'competitive_analysis': competitive_analysis,
            'constraints': {
                'timeline': '1 week',
                'team_size': 4,
                'complexity': 'moderate'
            }
        }
        
        print("📋 Phase 2: Roadmap Generation")
        
        # Step 4: Create Detailed Roadmap
        try:
            roadmap_dict = project_planner(project_idea, self.shared_context)
            
            # Step 5: Export for demo
            if 'error' not in roadmap_dict:
                roadmap_obj = self.planner.create_roadmap(
                    project_idea, 
                    research, 
                    self.shared_context.get('constraints')
                )
                
                # Generate different formats for demo
                markdown_roadmap = self.planner.export_roadmap(roadmap_obj, "markdown")
                json_roadmap = self.planner.export_roadmap(roadmap_obj, "json")
                
                return {
                    'project_idea': project_idea,
                    'research_summary': research[:500] + "..." if len(research) > 500 else research,
                    'technical_insights': tech_analysis[:500] + "..." if len(tech_analysis) > 500 else tech_analysis,
                    'roadmap': roadmap_dict,
                    'roadmap_markdown': markdown_roadmap,
                    'roadmap_json': json_roadmap,
                    'total_phases': len(roadmap_dict.get('phases', [])),
                    'estimated_hours': roadmap_dict.get('total_estimated_hours', 0),
                    'success': True
                }
            else:
                return {
                    'project_idea': project_idea,
                    'error': roadmap_dict['error'],
                    'success': False
                }
                
        except Exception as e:
            return {
                'project_idea': project_idea,
                'error': f"Roadmap generation failed: {str(e)}",
                'research_summary': research[:500] + "..." if len(research) > 500 else research,
                'success': False
            }


# Demo script
if __name__ == "__main__":
    coordinator = AgentCoordinator()
    
    # Demo projects perfect for Hack the North
    hackathon_projects = [
        "Real-time collaborative code editor with AI suggestions",
        # "Smart campus navigation app with AR features", 
        # "AI-powered mental health check-in platform"
    ]
    
    for project in hackathon_projects:
        print(f"\n🚀 BUILDING ROADMAP: {project}")
        print("="*80)
        
        try:
            result = coordinator.build_project(project)
            
            if result['success']:
                print(f"✅ Research completed")
                print(f"✅ Roadmap generated: {result['total_phases']} phases")
                print(f"⏱️ Total estimated time: {result['estimated_hours']} hours")
                
                # Show phases overview
                phases = result['roadmap']['phases']
                print(f"\n📊 PHASES:")
                for i, phase in enumerate(phases, 1):
                    print(f"  {i}. {phase['name']} ({len(phase['tasks'])} tasks)")
                
                # Show first few tasks as preview
                if phases:
                    first_phase = phases[0]
                    print(f"\n📋 Sample tasks from '{first_phase['name']}':")
                    for task in first_phase.get('tasks', [])[:3]:
                        print(f"  • {task['title']} ({task['estimated_hours']}h) - {task['agent_type']}")
                
                # Show required agents
                agents = set()
                for phase in phases:
                    for task in phase['tasks']:
                        agents.add(task['agent_type'])
                
                print(f"\n🤖 Required agents: {', '.join(sorted(agents))}")
                
                print(f"\n📄 Research Summary Preview:")
                print(result['research_summary'])
                
            else:
                print(f"❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
        
        print("-" * 80)