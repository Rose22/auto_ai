import os
import sys
import openai
import json
import yaml
import msgpack
import inspect
import datetime
import time

if not os.path.exists("config.yaml"):
    print("Error: config.yaml not found! Please ensure there is a config file.")
    exit()

config = {}
try:
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f.read())
except:
    print("Error while loading config. Please check it for errors!")
    exit()

def log(event: str, msg: str, truncate: bool = True):
    event = event.upper()
    current_time = datetime.datetime.strftime(datetime.datetime.now(), "%d-%m-%Y %H:%M")

    msg_formatted = f"{current_time} | [{event}] {msg}\n"
    if truncate:
        print(f"[{event}] {msg[:100]}..")
    else:
        print(f"[{event}] {msg}..")
    with open(config.get("logfile"), "a") as f:
        f.write(msg_formatted)

class ConversationManager:
    def __init__(self, client):
        self._client = client
        self._ctx = []
        self._tools = []
        self._last_tool_call = None
        self._last_tool_call_args = None

    def add_tool_class(self, toolclass, api_key: str = None):
        """
        adds tools to the manager based on a class with functions
        to make tools, just make a class like so:
        class MyToolClass:
            def search_web(self, query: str):
                search_the_web_or_whatever(query)
        """

        self._toolclass = toolclass

        for func_name in dir(self._toolclass):
            if func_name.startswith("_"):
                # skip private methods and other private properties
                continue

            try:
                func_obj = getattr(toolclass, func_name)
            except:
                continue

            if not callable(func_obj):
                continue

            func_params = inspect.signature(func_obj).parameters
            func_params_translated = {}
            for param_name, param in func_params.items():
                param_split = str(param).split(":")
                param_name = param_split[0]
                param_type = "str"
                if len(param_split) > 1:
                    param_type = param_split[1].strip()
                
                param_type_map = {
                    "str": "string",
                    "int": "integer"
                }

                for word, replacement in param_type_map.items():
                    if param_type == word:
                        param_type = replacement

                func_params_translated[param_name] = {"type": param_type,  "description": None}

            docstring = ""
            if "__doc__" in dir(func_obj):
                docstring = func_obj.__doc__

            tool = {
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": docstring,
                    "parameters": {
                        "type": "object",
                        "properties": func_params_translated,
                        "required": [],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }

            self._tools.append(tool)

    def load_system_prompt(self):
        if not os.path.exists("system_prompt.md"):
            with open("system_prompt.md", 'w') as f:
                f.write("")

        with open("system_prompt.md", "r") as f:
            self.insert_context("system", f.read())

    
    def insert_context(self, role: str, msg: str):
        msg_stripped = msg.strip().replace("\n", " ")
        log("context", f"inserted into context as {role}: {msg_stripped}")
        return self._ctx.append({"role": role, "content": msg})

    def send(self, role: str, msg: str, silent=False):
        """send a message to the AI as the chosen role and stream the response"""

        if not silent:
            log("request to AI", f"{role}: {msg}")

        # send the request
        stream = self._client.chat.completions.create(
            model=config.get("model"),
            messages=self._ctx+[
                {"role": role, "content": msg}
            ],
            tools=self._tools,
            stream=True
        )

        # stream the response
        chunks = []
        final_tool_calls = {}
        for chunk in stream:
            streamed_token = chunk.choices[0].delta

            # print the latest token in the stream
            if streamed_token.content:
                chunks.append(streamed_token.content)
                if not silent:
                    print(streamed_token.content, end="", flush=True)

            # handle tool calls, if any
            if streamed_token.tool_calls:
                # take the streamed tool call bits and mesh them together into a completed tool call array
                for tool_call in streamed_token.tool_calls:
                    index = tool_call.index

                    if index not in final_tool_calls:
                        final_tool_calls[index] = tool_call

                    final_tool_calls[index].function.arguments += tool_call.function.arguments

        # call any tool calls based on the stored tool call function
        for index, tool_call in final_tool_calls.items():
            # does the method exist within the class?
            if hasattr(self._toolclass, tool_call.function.name):
                if tool_call.function.name == self._last_tool_call and tool_call.function.arguments == self._last_tool_call_args:
                    log("toolcall", "AI tried calling the same tool again")
                    self.send("tool", json.dumps({"error": f"ALERT!! You are entering an endless loop. Stop what you are doing and choose a different action. Details: You already called this tool before! look at your list of tools and call a tool that isn't {tool_call.function.name}."}))
                # then get the class method object
                func_callable = getattr(self._toolclass, tool_call.function.name)
                # format its arguments in a JSON format the llm will understand
                arg_obj = json.loads(tool_call.function.arguments)
                log("toolcall", f"calling tool {tool_call.function.name}")
                # call the class method
                func_response = func_callable(**arg_obj)
                # and add the method's return value to the LLM's context window as a tool call response
                self._ctx.append({"role": "tool", "tool_call_id": tool_call.id, "content": func_response})

                self._last_tool_call = tool_call.function.name
                self._last_tool_call_args = tool_call.function.arguments

        # return the final streamed text response
        result = "".join(chunks)
        if result:
            self._ctx.append({"role": "assistant", "content": result})
            log("AI", result)

        # trim context if it exceeds a certain amount of messages
        # if len(self._ctx) > 6:
        #     self._ctx = self._ctx[1:]

        # write current context to file for debugging
        with open("ctx", "w") as f:
            f.write(json.dumps(self._ctx, indent=2))
        return result

if __name__ == "__main__":
    client = openai.OpenAI(base_url="http://localhost:5001/v1", api_key="dummy")

    convo = ConversationManager(client)
    # load all custom tool classes from tools.py
    import tools
    for tool_class in inspect.getmembers(tools):
        class_name = tool_class[0]
        class_obj = tool_class[1]
        if inspect.isclass(class_obj) and not class_name.startswith("__"):
            print(f"Loading tool class [{class_name}]..")
            convo.add_tool_class(class_obj)

    convo.load_system_prompt()

    print("Starting up Automatic AI..")

    system_prompt_repeat_counter = 0
    while True:
        now = datetime.datetime.now().isoformat()
        try:
            convo.send("user", f"Current Time: {now} | Last tool used: {convo._last_tool_call} | What do you want to do next? You must ALWAYS call a tool. Choose a different tool than the last tool used.", silent=True)

            system_prompt_repeat_counter += 1
            if system_prompt_repeat_counter > 6:
                # insert system prompt again.. so that it never forgets
                convo.load_mem()
                convo.load_system_prompt()
                system_prompt_repeat_counter = 0

        except Exception as e:
            log("error", f"error while sending request to LLM: {e}. trying again.")

    print()
