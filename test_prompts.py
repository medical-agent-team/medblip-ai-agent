#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick prompt testing script
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.orchestrator.prompt_experimenter import PromptExperiment


def quick_test():
    """Quick test of different prompt versions"""
    print("🧪 Quick Prompt Testing")
    print("=" * 40)
    
    experiment = PromptExperiment()
    
    # List available prompts
    experiment.list_available_prompts()
    
    # Test orchestrator prompts
    print("\n📝 Testing Orchestrator Prompts:")
    print("-" * 40)
    
    test_inputs = [
        "안녕하세요, 병원에서 찍은 X-ray 사진이 있어요",
        "30세 남성이고 가끔 기침이 나요",
        "이미지 업로드했습니다"
    ]
    
    for version in ["v1", "v2"]:
        print(f"\n🔍 Testing Orchestrator {version.upper()}:")
        if experiment.load_orchestrator_experiment(version):
            for test_input in test_inputs:
                print(f"\n>>> Input: {test_input}")
                result = experiment.test_orchestrator_response(test_input)
                if result:
                    print(f"Decision: {result.get('decision')}")
                    print(f"Message: {result.get('message', '')[:100]}...")
                print("-" * 30)
    
    # Test radiology prompts  
    print("\n🏥 Testing Radiology Prompts:")
    print("-" * 40)
    
    test_medblip = "chest x-ray shows normal cardiopulmonary findings with clear lung fields and normal heart size"
    test_patient_info = {
        "symptoms": "가끔 기침",
        "basic_response": "30세 남성",
        "medical_history": "특별한 병력 없음"
    }
    
    for version in ["v1", "v2"]:
        print(f"\n🔍 Testing Radiology {version.upper()}:")
        if experiment.load_radiology_experiment(version):
            result = experiment.test_radiology_response(test_medblip, test_patient_info)
            if result:
                print(f"Response length: {len(result)} characters")
                print(f"Preview: {result[:200]}...")
        print("-" * 30)
    
    # Save experiment log
    experiment.save_experiment_log()
    print("\n✅ Quick test completed!")


if __name__ == "__main__":
    quick_test()