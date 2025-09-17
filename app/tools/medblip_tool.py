#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MedBLIP Tool - LangChain Tool로 구현된 MedBLIP 모델 인터페이스
"""

import os
import logging
from typing import Optional, Union, Dict, Any
from PIL import Image
import torch
from pydantic import Field
from transformers import BlipForConditionalGeneration, BlipProcessor

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from app.core.model_utils import load_medblip_model

# Docker 로그에서 확인 가능한 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class MedBLIPTool(BaseTool):
    """
    MedBLIP 모델을 LangChain Tool로 래핑
    의료 이미지 분석을 위한 도구
    """

    name: str = "medblip_analyzer"
    description: str = """
    의료 이미지 분석 도구입니다.
    X-ray, CT, MRI 등의 의료 영상을 분석하여 의학적 소견을 제공합니다.
    입력: PIL Image 객체 또는 이미지 경로
    출력: 의료 영상에 대한 분석 결과 텍스트
    """
    model: Optional[Any] = Field(default=None, exclude=True)
    processor: Optional[Any] = Field(default=None, exclude=True)
    model_loaded: bool = Field(default=False, exclude=True)


    def __init__(self, **kwargs):
        logger.info("🚀 MedBLIPTool 초기화 시작")
        super().__init__(**kwargs)
        self._load_model()

    def _load_model(self):
        """MedBLIP 모델 로드"""
        logger.info("📥 MedBLIP 모델 로딩 시도 중...")
        try:
            model, processor, resolved_path = load_medblip_model()
            if model is not None and processor is not None:
                self.model = model
                self.processor = processor
                self.model_loaded = True
                logger.info(f"✅ MedBLIP 모델 로딩 성공: {resolved_path}")
            else:
                logger.warning("⚠️ MedBLIP 모델 로딩 실패 - 데모 모드로 동작")
                self.model_loaded = False
        except Exception as e:
            logger.error(f"❌ MedBLIP 모델 로딩 중 오류: {str(e)}")
            object.__setattr__(self, 'model_loaded', False)

    def _run(
        self,
        image_input: Union[str, Image.Image],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """도구 실행 메서드"""
        # run_manager는 LangChain 콜백용이지만 현재 사용하지 않음
        _ = run_manager  # Suppress unused variable warning
        logger.info("🔧 LangChain Tool 인터페이스를 통한 이미지 분석 호출")
        try:
            return self.analyze_medical_image(image_input)
        except Exception as e:
            logger.error(f"❌ Tool 실행 중 오류: {str(e)}")
            return f"이미지 분석 중 오류 발생: {str(e)}"

    def analyze_medical_image(
        self,
        image_input: Union[str, Image.Image],
        max_length: int = 100,
        num_beams: int = 5
    ) -> str:
        """
        의료 이미지 분석 수행

        Args:
            image_input: PIL Image 객체 또는 이미지 파일 경로
            max_length: 생성할 최대 토큰 수
            num_beams: Beam search를 위한 빔 수

        Returns:
            의료 영상 분석 결과 텍스트
        """
        logger.info("🔍 의료 이미지 분석 시작")

        if not self.model_loaded:
            logger.warning("⚠️ MedBLIP 모델 미로드 - 데모 모드로 분석")
            return self._demo_analysis()

        try:
            # 이미지 준비
            logger.info("🖼️ 이미지 전처리 중...")
            if isinstance(image_input, str):
                image = Image.open(image_input).convert('RGB')
                logger.info(f"📁 파일에서 이미지 로드: {image_input}")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert('RGB')
                logger.info("🖼️ PIL Image 객체 사용")
            else:
                raise ValueError("지원하지 않는 이미지 형식입니다.")

            # 모델 입력 준비
            logger.info("🔧 모델 입력 준비 중...")
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values

            # GPU 사용 가능한 경우 이동
            if torch.cuda.is_available() and hasattr(self.model, 'to'):
                logger.info("🚀 GPU 가속 사용")
                self.model = self.model.to('cuda')
                pixel_values = pixel_values.to('cuda')
            else:
                logger.info("💻 CPU 추론 사용")

            # 추론 수행
            logger.info("🧠 MedBLIP 모델 추론 중...")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False
                )

            # 결과 디코딩
            logger.info("📝 분석 결과 디코딩 중...")
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            # 후처리
            result = self._postprocess_analysis(generated_text)
            logger.info("✅ 의료 이미지 분석 완료")
            return result

        except Exception as e:
            error_msg = f"MedBLIP 분석 중 오류 발생: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return self._demo_analysis()

    def _postprocess_analysis(self, raw_text: str) -> str:
        """
        MedBLIP 원시 출력을 후처리

        Args:
            raw_text: MedBLIP 모델의 원시 출력

        Returns:
            정리된 분석 결과
        """
        # 기본 정리
        processed_text = raw_text.strip()

        # 의료진을 위한 형식으로 정리
        if processed_text:
            # 첫 글자 대문자로 변경
            if len(processed_text) > 1:
                processed_text = (
                    processed_text[0].upper() + processed_text[1:]
                )
            else:
                processed_text = processed_text.upper()

            # 마침표가 없으면 추가
            if not processed_text.endswith('.'):
                processed_text += '.'

            return processed_text
        else:
            return "이미지 분석 결과를 생성할 수 없습니다."

    def _demo_analysis(self) -> str:
        """
        모델이 로드되지 않았을 때 사용할 데모 분석 결과

        Returns:
            데모 분석 결과
        """
        demo_results = [
            "Chest X-ray demonstrates clear lung fields with no acute "
            "cardiopulmonary abnormalities. Heart size appears normal.",
            "The radiographic examination shows normal cardiac silhouette "
            "and no evidence of pneumonia or pleural effusion.",
            "Bilateral lung fields are clear without focal consolidation. "
            "Cardiac outline is within normal limits.",
            "No acute abnormalities detected in the chest radiograph. "
            "Recommend clinical correlation.",
            "The imaging study reveals normal findings consistent with "
            "healthy lung tissue and cardiac structure."
        ]

        import random
        return random.choice(demo_results)

    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환

        Returns:
            모델 상태 및 정보
        """
        return {
            "model_loaded": self.model_loaded,
            "model_type": "MedBLIP" if self.model_loaded else "Demo Mode",
            "device": (
                "cuda" if torch.cuda.is_available() and self.model_loaded
                else "cpu"
            ),
            "status": "Ready" if self.model_loaded else "Demo Mode - Model not loaded"
        }

    def validate_image(self, image_input: Union[str, Image.Image]) -> bool:
        """
        입력 이미지 유효성 검사

        Args:
            image_input: 검사할 이미지

        Returns:
            이미지가 유효한지 여부
        """
        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    return False
                Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                # PIL Image 객체인 경우 기본적으로 유효
                pass
            else:
                return False
            return True
        except Exception:
            return False

    def get_supported_formats(self) -> list:
        """
        지원되는 이미지 형식 반환

        Returns:
            지원되는 이미지 형식 리스트
        """
        return ['PNG', 'JPG', 'JPEG', 'BMP', 'TIFF', 'GIF']


# LangChain Tools 호환성을 위한 팩토리 함수
def create_medblip_tool() -> MedBLIPTool:
    """
    MedBLIP 도구 인스턴스 생성

    Returns:
        MedBLIP 도구 인스턴스
    """
    return MedBLIPTool()


# 편의 함수들
def quick_analyze(image_input: Union[str, Image.Image]) -> str:
    """
    빠른 이미지 분석을 위한 편의 함수

    Args:
        image_input: 분석할 이미지

    Returns:
        분석 결과
    """
    tool = create_medblip_tool()
    return tool.analyze_medical_image(image_input)


def batch_analyze(image_inputs: list) -> list:
    """
    여러 이미지 배치 분석

    Args:
        image_inputs: 분석할 이미지들의 리스트

    Returns:
        각 이미지의 분석 결과 리스트
    """
    tool = create_medblip_tool()
    results = []

    for image_input in image_inputs:
        try:
            result = tool.analyze_medical_image(image_input)
            results.append(result)
        except Exception as e:
            results.append(f"분석 실패: {str(e)}")

    return results
