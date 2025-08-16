# medical-data-agent

# poetry : Python 및 package 설치 호환성 체크 후 추가 패키지 설치
# Docker : 컨테이너 기반 배포 환경 구성 동일하게 구성

# poetry 설치방법

# BLIP(finetuned) 사용방법
BLIP model 다운: https://drive.google.com/file/d/1ZxWOffhTbLqCAhn_Wz1kCZywpvHW_6x2/view?usp=sharing

folder에다가 저장 -> 예시: /blip_model_finetuned
```python
from PIL import Image

from transformers import BlipForConditionalGeneration, BlipProcessor
model = BlipForConditionalGeneration.from_pretrained("./blip_model_finetuned")
processor = BlipProcessor.from_pretrained("./blip_model_finetuned")

image_path =  "./blip_model_finetuned/sample_image.png"

image = Image.open(image_path)
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs.pixel_values
generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)
