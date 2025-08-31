#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for MedBLIP model integration
"""

import os
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_medblip_model():
    """Test the finetuned MedBLIP model"""
    print("🔍 Testing MedBLIP model...")
    
    # 가능한 모델 경로들 (우선순위 순)
    possible_paths = [
        "/app/model/blip_model_finetuned",  # Docker 컨테이너 내부 경로
        "/app/blip_model_finetuned",        # 루트 디렉토리
        "./model/blip_model_finetuned",     # 상대 경로 1
        "./blip_model_finetuned",           # 원래 경로
        "blip_model_finetuned"              # 현재 디렉토리
    ]
    
    model = None
    processor = None
    model_path = None
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                print(f"📥 Loading MedBLIP model from: {path}")
                model = BlipForConditionalGeneration.from_pretrained(path, local_files_only=True)
                processor = BlipProcessor.from_pretrained(path, local_files_only=True)
                model_path = path
                print(f"✅ Model loaded successfully from: {path}")
                break
            else:
                print(f"⚠️  Path not found: {path}")
        except Exception as e:
            print(f"❌ Failed to load model from {path}: {str(e)}")
            continue
    
    if model is None or processor is None:
        print("❌ Could not load MedBLIP model from any path")
        return False
        
    # Test with sample image if available
    sample_image_paths = [
        f"{model_path}/sample_image.png",
        "./blip_model_finetuned/sample_image.png",
        "./sample_image.png"
    ]
    
    sample_found = False
    for sample_image_path in sample_image_paths:
        if os.path.exists(sample_image_path):
            print(f"📷 Testing with sample image: {sample_image_path}")
            image = Image.open(sample_image_path)
            
            # Run inference
            print("🧠 Running MedBLIP inference...")
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values
            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(f"📋 MedBLIP Analysis Result: {generated_caption}")
            sample_found = True
            break
    
    if not sample_found:
        print("⚠️  No sample image found in any location")
        print("✅ Model loading test passed, but no sample image available")
    
    return True

def test_openai_integration():
    """Test OpenAI API integration"""
    print("\n🔍 Testing OpenAI integration...")
    
    try:
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment variables")
            return False
            
        print("✅ OpenAI API key found")
        
        # Test basic connection
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=0.3
        )
        
        # Simple test query
        response = llm.invoke("Hello, this is a test.")
        print("✅ OpenAI API connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Error testing OpenAI integration: {str(e)}")
        return False

def test_agents():
    """Test the agent integration"""
    print("\n🔍 Testing agent integration...")
    
    try:
        import sys
        sys.path.append("./app")
        
        from app.orchestrator.agent import OrchestratorAgent
        from app.orchestrator.radiology_agent import RadiologyAnalysisAgent
        
        # Test orchestrator agent
        print("🤖 Testing OrchestratorAgent...")
        orchestrator = OrchestratorAgent()
        print("✅ OrchestratorAgent initialized successfully")
        
        # Test radiology agent
        print("🏥 Testing RadiologyAnalysisAgent...")
        radiology_agent = RadiologyAnalysisAgent()
        print("✅ RadiologyAnalysisAgent initialized successfully")
        
        # Test medical consultation function
        test_medblip_result = "chest x-ray shows normal lung fields with no acute findings"
        test_patient_info = {
            "symptoms": "가끔 기침",
            "basic_response": "30세 남성",
            "medical_history": "특별한 병력 없음"
        }
        
        print("🩺 Testing medical consultation...")
        consultation_result = radiology_agent.provide_medical_consultation(
            test_medblip_result, 
            test_patient_info
        )
        
        print(f"📋 Consultation result preview: {consultation_result[:100]}...")
        print("✅ Medical consultation test successful")
        return True
        
    except Exception as e:
        print(f"❌ Error testing agents: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting MedBLIP integration tests...")
    print("=" * 50)
    
    test_results = []
    
    # Test 1: MedBLIP model
    test_results.append(test_medblip_model())
    
    # Test 2: OpenAI integration
    test_results.append(test_openai_integration())
    
    # Test 3: Agent integration
    test_results.append(test_agents())
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(test_results)}/{len(test_results)} tests")
    
    if all(test_results):
        print("🎉 All tests passed! MedBLIP integration is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)