# -*- coding: utf-8 -*-
"""
Radiological Image Analysis Agent for minimal scenario
"""

import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from .prompts.prompt import RADIOLOGY_ANALYSIS_PROMPT


class RadiologyAnalysisAgent:
    """Specialized agent for medical consultation using finetuned MedBLIP results"""
    
    def __init__(self):
        # Load OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4",  # Use GPT-4 for better medical consultation
            temperature=0.3  # Lower temperature for more consistent medical explanations
        )
        self.prompt = RADIOLOGY_ANALYSIS_PROMPT
    
    def provide_medical_consultation(self, medblip_result: str, patient_info: Dict[str, Any]) -> str:
        """
        Provide medical consultation based on finetuned MedBLIP analysis results
        
        Args:
            medblip_result: Finetuned MedBLIP model analysis result
            patient_info: Patient's collected information
            
        Returns:
            Comprehensive medical consultation response
        """
        
        # Extract relevant patient information
        symptoms = patient_info.get("symptoms", "특별한 증상 없음")
        basic_info = patient_info.get("basic_response", "정보 미제공")
        medical_history = patient_info.get("medical_history", "특별한 병력 없음")
        
        # Execute analysis prompt
        chain = self.prompt | self.llm
        response = chain.invoke({
            "image_analysis": medblip_result,
            "symptoms": symptoms,
            "basic_info": basic_info,
            "medical_history": medical_history
        })
        
        return response.content
    
    def get_imaging_method_explanation(self, image_type: str = "X-ray") -> str:
        """Provide simple explanation of imaging method"""
        explanations = {
            "X-ray": """
X-ray 검사는 가장 기본적인 영상 검사 중 하나입니다.

**검사 방법:**
- X선을 몸에 투과시켜 뼈와 폐 등의 구조를 영상으로 만드는 검사입니다
- 검사 시간은 보통 5-10분 정도로 매우 짧습니다
- 방사선 노출량은 매우 적어 안전합니다

**무엇을 볼 수 있나요:**
- 뼈의 골절이나 변형
- 폐의 염증이나 이상 소견
- 심장의 크기나 모양
- 복부 장기의 전반적인 상태
            """,
            "CT": "CT 스캔은 X선을 이용해 몸의 단면 영상을 만드는 정밀한 검사입니다.",
            "MRI": "MRI는 자기장을 이용해 몸 속 연조직을 자세히 볼 수 있는 검사입니다."
        }
        
        return explanations.get(image_type, explanations["X-ray"])