# https://medium.com/@s_eschweiler/revolutionizing-ai-interactions-unpacking-openais-function-calling-capability-in-the-chat-b0a6b71a9452
# https://platform.openai.com/docs/guides/gpt/function-calling
import json
import os
from dotenv import load_dotenv
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""

    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }

    return json.dumps(weather_info)


FUNCTIONS = {
    "get_current_weather": get_current_weather,
}


def check_function_response(response):
    finish_reason = response["choices"][0].get("finish_reason")
    return finish_reason == "function_call"


def handle_function_call(response):
    message = response["choices"][0]["message"]
    function_call = message["function_call"]

    arguments = function_call.get("arguments", {})
    if arguments:
        arguments = json.loads(arguments)

    name = function_call["name"]
    func = FUNCTIONS.get(name)

    if func:
        return name, func(**arguments)

    return name, None

def call_llm(messages):
    functions = [{
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["fahrenheit", "celsius"],
                }
            },
            "required": ["location"]
        }
    }]

    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto"
    )


def run_conversation(messages):
    response = call_llm(messages)

    if check_function_response(response):
        func_name, func_resp = handle_function_call(response)

        if func_resp:
            messages.append({
                "role": "function",
                "name": func_name,
                "content": func_resp
            })

            return run_conversation(messages)

    return response



def run():
    messages = [{
        "role": "user",
        "content": "What's the weather like in Boston?"
    }]

    result = run_conversation(messages)

    print("DONE", result)


if __name__ == "__main__":
    run()