#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt loading and management system for experimentation
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


class PromptLoader:
    """Dynamic prompt loading system for agent experimentation"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.experiments_dir = self.base_dir / "experiments"
        self._ensure_experiments_dir()
        
    def _ensure_experiments_dir(self):
        """Ensure experiments directory exists"""
        self.experiments_dir.mkdir(exist_ok=True)
    
    def list_available_prompts(self) -> Dict[str, List[str]]:
        """List all available prompt files by agent type"""
        prompts = {"orchestrator": [], "radiology": []}
        
        if not self.experiments_dir.exists():
            return prompts
            
        for file_path in self.experiments_dir.glob("*.txt"):
            filename = file_path.stem
            if filename.startswith("orchestrator_"):
                prompts["orchestrator"].append(filename)
            elif filename.startswith("radiology_"):
                prompts["radiology"].append(filename)
                
        return prompts
    
    def load_prompt_content(self, prompt_name: str) -> Optional[str]:
        """Load prompt content from file"""
        prompt_file = self.experiments_dir / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            return None
            
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading prompt {prompt_name}: {e}")
            return None
    
    def create_orchestrator_prompt(self, prompt_name: str = "orchestrator_v1") -> Optional[ChatPromptTemplate]:
        """Create orchestrator ChatPromptTemplate from file"""
        system_content = self.load_prompt_content(prompt_name)
        
        if system_content is None:
            return None
            
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_content),
            HumanMessagePromptTemplate.from_template(
                """
                [Input] {user_input}
                [Context] 
                현재 대화 단계: {conversation_stage}
                수집된 정보: {collected_info}

                이미지 첨부 여부: {has_image}
                이전 대화 내용: {conversation_history}
                """
            )
        ])
    
    def create_radiology_prompt(self, prompt_name: str = "radiology_v1") -> Optional[ChatPromptTemplate]:
        """Create radiology ChatPromptTemplate from file"""
        system_content = self.load_prompt_content(prompt_name)
        
        if system_content is None:
            return None
            
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_content),
            HumanMessagePromptTemplate.from_template(
                """
                MedBLIP 파인튜닝 모델 분석 결과: {image_analysis}
                환자 증상: {symptoms}
                환자 기본 정보: {basic_info}
                과거 병력: {medical_history}
                
                위 MedBLIP 분석 결과를 바탕으로 환자가 이해하기 쉬운 상담을 제공해주세요.
                """
            )
        ])
    
    def save_prompt_experiment(self, agent_type: str, version: str, content: str) -> bool:
        """Save new experimental prompt"""
        prompt_name = f"{agent_type}_{version}"
        prompt_file = self.experiments_dir / f"{prompt_name}.txt"
        
        try:
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error saving prompt {prompt_name}: {e}")
            return False
    
    def get_prompt_info(self, prompt_name: str) -> Dict:
        """Get information about a specific prompt"""
        content = self.load_prompt_content(prompt_name)
        if content is None:
            return {"exists": False}
            
        prompt_file = self.experiments_dir / f"{prompt_name}.txt"
        stat = prompt_file.stat()
        
        return {
            "exists": True,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "lines": len(content.split('\n')),
            "preview": content[:200] + "..." if len(content) > 200 else content
        }


# Global prompt loader instance
prompt_loader = PromptLoader()


def get_orchestrator_prompt(version: str = "v1") -> Optional[ChatPromptTemplate]:
    """Convenience function to get orchestrator prompt"""
    return prompt_loader.create_orchestrator_prompt(f"orchestrator_{version}")


def get_radiology_prompt(version: str = "v1") -> Optional[ChatPromptTemplate]:
    """Convenience function to get radiology prompt"""
    return prompt_loader.create_radiology_prompt(f"radiology_{version}")


def list_prompts() -> Dict[str, List[str]]:
    """Convenience function to list available prompts"""
    return prompt_loader.list_available_prompts()


if __name__ == "__main__":
    # Test the prompt loader
    print("Available prompts:")
    prompts = list_prompts()
    for agent_type, prompt_list in prompts.items():
        print(f"  {agent_type}: {prompt_list}")
    
    # Test loading orchestrator prompt
    orch_prompt = get_orchestrator_prompt("v1")
    if orch_prompt:
        print("✅ Orchestrator prompt v1 loaded successfully")
    else:
        print("❌ Failed to load orchestrator prompt v1")
        
    # Test loading radiology prompt
    rad_prompt = get_radiology_prompt("v1") 
    if rad_prompt:
        print("✅ Radiology prompt v1 loaded successfully")
    else:
        print("❌ Failed to load radiology prompt v1")