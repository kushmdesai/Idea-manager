from agent_coordinator import AgentCoordinator

def quick_demo():
    coordinator = AgentCoordinator()
    
    print("🎯 AI ROADMAP GENERATOR - HACK THE NORTH 2025")
    print("="*60)
    
    project = "AI-powered study buddy matching app"
    print(f"📝 Project: {project}\n")
    
    result = coordinator.build_project(project)
    
    if result['success']:
        print("🎉 SUCCESS!")
        print(f"📊 Generated {result['total_phases']} phases")
        print(f"⏱️ Total time: {result['estimated_hours']} hours")
        
        print(f"\n🗺️ ROADMAP PHASES:")
        for i, phase in enumerate(result['roadmap']['phases'], 1):
            task_count = len(phase['tasks'])
            print(f"  {i}. {phase['name']} - {task_count} tasks")
        
        print(f"\n🤖 AI agents will handle: research, coding, testing, deployment")
        print("✅ Ready for multi-agent execution!")
        
    else:
        print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    quick_demo()