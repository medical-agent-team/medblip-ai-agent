# -*- coding: utf-8 -*-
"""
Radiological Image Analysis Agent for minimal scenario
"""

import os
from typing import Dict, Any
from .prompts.prompt import RADIOLOGY_ANALYSIS_PROMPT

try:
    # Use centralized LLM factory with vLLM/Langfuse support
    from app.core.llm_factory import get_llm_for_agent
    from app.core.observability import get_callbacks
    LLM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    LLM_AVAILABLE = False


class RadiologyAnalysisAgent:
    """Specialized agent for medical consultation using finetuned MedBLIP results.

    If `OPENAI_API_KEY` is missing or `langchain_openai` is not installed,
    falls back to a local, template-based explanation (no network calls).
    """

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.prompt = RADIOLOGY_ANALYSIS_PROMPT

        if LLM_AVAILABLE:
            try:
                callbacks = get_callbacks()
                self.llm = get_llm_for_agent(
                    agent_type="radiology",
                    api_key=api_key,
                    callbacks=callbacks
                )
            except Exception:
                self.llm = None
        else:
            self.llm = None
    
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
        
        # Online path
        if self.llm is not None:
            chain = self.prompt | self.llm
            response = chain.invoke({
                "image_analysis": medblip_result,
                "symptoms": symptoms,
                "basic_info": basic_info,
                "medical_history": medical_history,
            })
            return response.content

        # Offline fallback (no network, simple template)
        return (
            "### 이미지 소견\n"
            f"{medblip_result}\n\n"
            "### 검사 방법 설명\n"
            "업로드하신 방사선 이미지를 기반으로 한 분석 결과입니다. \n"
            "촬영된 영상은 일반적으로 방사선(X-ray) 또는 CT/MRI 등을 활용하여 얻습니다. \n"
            "검사의 목적과 과정은 의료진의 판단에 따라 달라질 수 있습니다.\n\n"
            "### 가능한 상태들(참고용)\n"
            "해당 분석은 교육적 설명을 위한 참고 정보입니다. 명확한 진단을 대신하지 않습니다.\n\n"
            "### 관련 증상\n"
            f"사용자 제공 정보: {symptoms} / 기본정보: {basic_info} / 병력: {medical_history}\n\n"
            "### 권고사항\n"
            "정확한 진단과 치료를 위해서는 반드시 의료진과 상의하시기 바랍니다."
        )
    
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
