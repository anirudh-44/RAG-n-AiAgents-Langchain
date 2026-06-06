from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm1  = HuggingFaceEndpoint(
        repo_id = "Qwen/Qwen3-Coder-Next",
        task = "text-generation"
)

llm2  = HuggingFaceEndpoint(
        repo_id = "deepseek-ai/DeepSeek-V4-Flash",
        task = "text-generation"
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template = 'Generate short and simple notes from the following text. \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template = 'Generate a quiz containing 5 simple questions from the following text. \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document. \n notes -> {notes} and \n quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

final_chain = parallel_chain | merge_chain

text = """
FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.

Historically, async work in Python has been nontrivial (though its API has rapidly improved since Python 3.4) particularly with Flask. Essentially, Flask (on most WSGI servers) is blocking by default - work triggered by a request to a particular endpoint will hold the server entirely until that request is completed. Instead, Flask (or rather, the WSGI server running it, like gunicorn or uWSGI) achieve scaling by running multiple worker instances of the app in parallel, such that requests can be farmed to other workers while one is busy. Within a single worker, asynchronous work can be wrapped in a blocking call (the route function itself is still blocking), threaded (in newer versions of Flask), or farmed to a queue manager like Celery - but there isn’t a single consistent story where routes can cleanly handle asynchronous requests without additional tooling.

FastAPI is designed from the ground up to run asynchronously - thanks to its underlying starlette ASGI framework, route functions default to running within an asynchronous event loop. With a good ASGI server (FastAPI is designed to couple to uvicorn, running on top of uvloop) this can get us performance on par with fast asynchronous webservers in Go or Node, without losing the benefits of Python’s broader machine learning ecosystem.

In contrast to messing with threads or Celery queues to achieve asynchronous execution in Flask, running an endpoint asynchronously is dead simple in FastAPI - we simply declare the route function as asynchronous (with async def) and we’re ready to go! We can even do this if the route function isn’t conventionally asynchronous - that is, we don’t have any awaitable calls (like if the endpoint is running inference against an ML model). In fact, unless the endpoint is specifically performing a blocking IO operation (to a database, for example), it’s better to declare the function with async def (as blocking functions are actually punted to an external threadpool and then awaited anyhow).

"""
result = final_chain.invoke({'text':text})
print(result)

final_chain.get_graph().print_ascii()

