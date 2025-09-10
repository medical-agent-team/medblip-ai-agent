#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt experimentation tool for testing different agent prompts
"""

import os
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.orchestrator.agent import OrchestratorAgent
from app.orchestrator.radiology_agent import RadiologyAnalysisAgent
from app.orchestrator.prompts.prompt_loader import prompt_loader, list_prompts


class PromptExperiment:
    """Class to manage prompt experiments"""
    
    def __init__(self):
        self.experiments_log = []
        self.current_orchestrator = None
        self.current_radiology = None
        
    def list_available_prompts(self):
        """List all available prompt versions"""
        prompts = list_prompts()
        print("\n🧪 Available Experimental Prompts:")
        print("=" * 50)
        
        for agent_type, prompt_list in prompts.items():
            print(f"\n📝 {agent_type.title()} Agent Prompts:")
            if not prompt_list:
                print("  No experimental prompts found")
                continue
                
            for prompt_name in prompt_list:
                info = prompt_loader.get_prompt_info(prompt_name)
                if info['exists']:
                    print(f"  ✅ {prompt_name}")
                    print(f"     Lines: {info['lines']}, Size: {info['size']} bytes")
                    print(f"     Preview: {info['preview'][:100]}...")
                else:
                    print(f"  ❌ {prompt_name} (not found)")
        print()
    
    def load_orchestrator_experiment(self, version: str) -> bool:
        """Load orchestrator agent with experimental prompt"""
        try:
            print(f"🔄 Loading Orchestrator Agent with prompt version: {version}")
            self.current_orchestrator = OrchestratorAgent(prompt_version=version)
            print(f"✅ Orchestrator loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load Orchestrator: {e}")
            return False
    
    def load_radiology_experiment(self, version: str) -> bool:
        """Load radiology agent with experimental prompt"""
        try:
            print(f"🔄 Loading Radiology Agent with prompt version: {version}")
            self.current_radiology = RadiologyAnalysisAgent(prompt_version=version)
            print(f"✅ Radiology Agent loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load Radiology Agent: {e}")
            return False
    
    def test_orchestrator_response(self, user_input: str, 
                                 conversation_stage: str = "greeting",
                                 collected_info: Dict = None,
                                 has_image: bool = False) -> Optional[Dict]:
        """Test orchestrator agent response"""
        if self.current_orchestrator is None:
            print("❌ No orchestrator agent loaded. Use load_orchestrator_experiment() first.")
            return None
        
        if collected_info is None:
            collected_info = {}
            
        try:
            print(f"\n🧪 Testing Orchestrator Response:")
            print(f"Input: {user_input}")
            print(f"Stage: {conversation_stage}")
            print(f"Has Image: {has_image}")
            
            result = self.current_orchestrator.process_conversation(
                user_input=user_input,
                has_image=has_image
            )
            
            print(f"\n📤 Response:")
            print(f"Decision: {result.get('decision', 'None')}")
            print(f"Reason: {result.get('reason', 'None')}")
            print(f"Message: {result.get('message', 'None')}")
            print(f"New Stage: {result.get('conversation_stage', 'None')}")
            
            # Log experiment
            experiment = {
                "timestamp": datetime.now().isoformat(),
                "agent": "orchestrator",
                "prompt_version": self.current_orchestrator.current_prompt_version,
                "input": user_input,
                "context": {
                    "stage": conversation_stage,
                    "collected_info": collected_info,
                    "has_image": has_image
                },
                "output": result
            }
            self.experiments_log.append(experiment)
            
            return result
            
        except Exception as e:
            print(f"❌ Error testing orchestrator: {e}")
            return None
    
    def test_radiology_response(self, medblip_result: str, 
                              patient_info: Dict = None) -> Optional[str]:
        """Test radiology agent response"""
        if self.current_radiology is None:
            print("❌ No radiology agent loaded. Use load_radiology_experiment() first.")
            return None
            
        if patient_info is None:
            patient_info = {
                "symptoms": "특별한 증상 없음",
                "basic_response": "30세 남성", 
                "medical_history": "특별한 병력 없음"
            }
        
        try:
            print(f"\n🧪 Testing Radiology Analysis:")
            print(f"MedBLIP Result: {medblip_result}")
            print(f"Patient Info: {patient_info}")
            
            result = self.current_radiology.provide_medical_consultation(
                medblip_result=medblip_result,
                patient_info=patient_info
            )
            
            print(f"\n📤 Medical Consultation:")
            print(result)
            
            # Log experiment
            experiment = {
                "timestamp": datetime.now().isoformat(),
                "agent": "radiology",
                "prompt_version": self.current_radiology.current_prompt_version,
                "input": {
                    "medblip_result": medblip_result,
                    "patient_info": patient_info
                },
                "output": result
            }
            self.experiments_log.append(experiment)
            
            return result
            
        except Exception as e:
            print(f"❌ Error testing radiology agent: {e}")
            return None
    
    def save_experiment_log(self, filename: str = None) -> bool:
        """Save experiment log to file"""
        if not self.experiments_log:
            print("❌ No experiments to save")
            return False
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prompt_experiment_log_{timestamp}.json"
        
        try:
            log_dir = os.path.join(os.path.dirname(__file__), "experiment_logs")
            os.makedirs(log_dir, exist_ok=True)
            
            log_path = os.path.join(log_dir, filename)
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(self.experiments_log, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Experiment log saved to: {log_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving experiment log: {e}")
            return False
    
    def compare_prompts(self, test_cases: List[Dict]) -> Dict:
        """Compare different prompt versions with same test cases"""
        comparison_results = {}
        
        print(f"\n🔬 Running Prompt Comparison with {len(test_cases)} test cases")
        print("=" * 60)
        
        # Get available prompts
        available_prompts = list_prompts()
        
        # Test orchestrator prompts
        if available_prompts['orchestrator']:
            print(f"\n📝 Testing Orchestrator Prompts:")
            for prompt_name in available_prompts['orchestrator']:
                version = prompt_name.replace('orchestrator_', '')
                print(f"\n🧪 Testing {prompt_name}...")
                
                if self.load_orchestrator_experiment(version):
                    results = []
                    for i, test_case in enumerate(test_cases):
                        if test_case.get('type') == 'orchestrator':
                            result = self.test_orchestrator_response(
                                user_input=test_case['input'],
                                conversation_stage=test_case.get('stage', 'greeting'),
                                collected_info=test_case.get('collected_info', {}),
                                has_image=test_case.get('has_image', False)
                            )
                            results.append(result)
                    
                    comparison_results[prompt_name] = {
                        "agent_type": "orchestrator", 
                        "results": results
                    }
        
        # Test radiology prompts
        if available_prompts['radiology']:
            print(f"\n🏥 Testing Radiology Prompts:")
            for prompt_name in available_prompts['radiology']:
                version = prompt_name.replace('radiology_', '')
                print(f"\n🧪 Testing {prompt_name}...")
                
                if self.load_radiology_experiment(version):
                    results = []
                    for i, test_case in enumerate(test_cases):
                        if test_case.get('type') == 'radiology':
                            result = self.test_radiology_response(
                                medblip_result=test_case['medblip_result'],
                                patient_info=test_case.get('patient_info', {})
                            )
                            results.append(result)
                    
                    comparison_results[prompt_name] = {
                        "agent_type": "radiology",
                        "results": results
                    }
        
        return comparison_results


def interactive_prompt_tester():
    """Interactive prompt testing interface"""
    experiment = PromptExperiment()
    
    print("🧪 Welcome to Prompt Experimentation Tool!")
    print("=" * 50)
    
    while True:
        print("\nAvailable commands:")
        print("1. list - List available prompts")
        print("2. load_orch <version> - Load orchestrator prompt")
        print("3. load_rad <version> - Load radiology prompt")
        print("4. test_orch - Test orchestrator response")
        print("5. test_rad - Test radiology response")
        print("6. save_log - Save experiment log")
        print("7. compare - Run prompt comparison")
        print("8. quit - Exit")
        
        command = input("\nEnter command: ").strip().lower()
        
        if command == "quit":
            break
        elif command == "list":
            experiment.list_available_prompts()
        elif command.startswith("load_orch"):
            parts = command.split()
            if len(parts) > 1:
                version = parts[1]
                experiment.load_orchestrator_experiment(version)
            else:
                print("❌ Please specify version: load_orch <version>")
        elif command.startswith("load_rad"):
            parts = command.split()
            if len(parts) > 1:
                version = parts[1]
                experiment.load_radiology_experiment(version)
            else:
                print("❌ Please specify version: load_rad <version>")
        elif command == "test_orch":
            user_input = input("Enter user input: ")
            stage = input("Enter conversation stage (default: greeting): ") or "greeting"
            has_image = input("Has image? (y/n, default: n): ").lower() == 'y'
            experiment.test_orchestrator_response(user_input, stage, {}, has_image)
        elif command == "test_rad":
            medblip_result = input("Enter MedBLIP analysis result: ")
            symptoms = input("Enter symptoms (optional): ") or "특별한 증상 없음"
            basic_info = input("Enter basic info (optional): ") or "30세 남성"
            patient_info = {
                "symptoms": symptoms,
                "basic_response": basic_info,
                "medical_history": "특별한 병력 없음"
            }
            experiment.test_radiology_response(medblip_result, patient_info)
        elif command == "save_log":
            experiment.save_experiment_log()
        elif command == "compare":
            print("Running sample comparison...")
            test_cases = [
                {
                    "type": "orchestrator",
                    "input": "안녕하세요, 병원에서 찍은 X-ray 사진이 있어요",
                    "stage": "greeting",
                    "has_image": False
                },
                {
                    "type": "radiology",
                    "medblip_result": "chest x-ray shows normal cardiopulmonary findings with clear lung fields",
                    "patient_info": {"symptoms": "가끔 기침", "basic_response": "30세 남성"}
                }
            ]
            experiment.compare_prompts(test_cases)
        else:
            print("❌ Unknown command")
    
    print("👋 Goodbye!")


if __name__ == "__main__":
    interactive_prompt_tester()