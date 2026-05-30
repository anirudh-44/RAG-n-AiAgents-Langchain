from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm  = HuggingFaceEndpoint(
        repo_id = "HuggingFaceTB/SmolLM3-3B",
        task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of France?")

print(result.content)