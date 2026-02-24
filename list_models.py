"""List available Gemini models."""
from google import genai
import config

client = genai.Client(api_key=config.GEMINI_API_KEY)

print("Available Gemini models:")
print("=" * 50)

try:
    models = client.models.list()
    for model in models:
        print(f"- {model.name}")
        if hasattr(model, 'supported_generation_methods'):
            print(f"  Methods: {model.supported_generation_methods}")
except Exception as e:
    print(f"Error listing models: {e}")
