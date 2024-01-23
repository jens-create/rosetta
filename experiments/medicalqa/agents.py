import json
import re
import typing as typ
from textwrap import wrap

import jinja2

from experiments.medicalqa.functions.findzebra import FindZebraAPI
from experiments.medicalqa.functions.wikipedia import Wikipedia
from experiments.medicalqa.prompt import QuestionAnswer, Shot, get_templates
from experiments.medicalqa.responses import extract_multiple_choice_prediction, extract_prediction
from experiments.medicalqa.shots import get_cot_shots, get_direct_shots, get_fixed_shots, get_react_shots
from src.clients.base import BaseClient
from src.components.models import ChatMessage, CompletionConfig


class Agent:
    """A base class for agents. An agent is a client + messages."""

    def __init__(self, client: BaseClient, dataset: str, prompt_type: int, request: dict[str, str]) -> None:
        self.client = client
        self.model = client.checkpoint  # type: ignore
        self.template = get_templates(dataset)
        self.dataset = dataset
        self.request = request
        self.rendered_request = jinja2.Template(self.template["input"]).render(**self.request)
        self.fewshot = False
        self.prompt_type = prompt_type
        self.system_message = ChatMessage(
            role="system",
            content="You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct.",
        )

    def run(self) -> str:
        """Run the agent."""
        raise NotImplementedError("This function should be implemented in the child class.")

    def query(self, messages: str | list[ChatMessage], guidance_str: str = "", completion_config: CompletionConfig = CompletionConfig()) -> str:  # type: ignore   # noqa: B008
        """Query the client. A wrapper for the client."""
        return self.client(messages, guidance_str, completion_config)

    def answer_question(self, context: str = "") -> dict[str, typ.Any]:
        """Query the client with a question and possibly a context using the AnswerQuestion model.
        The messages and completion_config are dependent on the language model.
        """
        if (
            self.model
            in ["meta-llama/Llama-2-70b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-7b-chat-hf"]
            and self.fewshot
        ):
            # fewshot
            system_instruction = "\n".join(
                [
                    "You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct.",
                    'You must respond with "function = " followed with valid JSON that follows the provided schema',
                ]
            )
            completion_config = CompletionConfig(stop=["\n\n"])  # type: ignore

        elif self.model in [
            "meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "codellama/CodeLlama-7b-Instruct-hf",
            "codellama/CodeLlama-13b-Instruct-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ]:
            system_instruction = "\n".join(
                [
                    "You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct.",
                    "You must respond with a function call.",  # "function = " followed with valid JSON that follows the provided schema.',
                    'Example: functions.QuestionAnswer({"explanation": <explanation>, "answer": <answer>})',
                    #'Example: functions.QuestionAnswer({"answer_option": (A)})',
                    "No other format is allowed.",
                    # "Keep the explanation simple, short and concise.",  # llama7 + 13chat zero shot: Keep the explanation simple, short and to the point without the use commas or full stops.
                ]
            )

            if self.prompt_type == 2:
                system_instruction = "\n".join(
                    [
                        "You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct.",
                        "You must respond with a function call.",  # "function = " followed with valid JSON that follows the provided schema.',
                        'Example: function = {"name": "QuestionAnswer", "arguments": {"explanation": <explanation>, "answer": <answer>}}',
                        "No other format is allowed.",
                    ]
                )

            completion_config = CompletionConfig()  # type: ignore 0.28, json: 0.8
        elif self.model in ["meta-llama/Llama-2-7b-hf"]:
            system_instruction = "\n".join(
                [
                    "You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct.",
                    'You have to answer in a JSON format with the following structure: function = {"name": "function_name", "arguments": {"argument_name": "argument_value"}}.',
                    "You can only answer in the above format. No other format is accepted.",
                ]
            )
            completion_config = CompletionConfig(presence_penalty=-0.5)  # type: ignore

        else:
            raise ValueError(f"Model {self.model} is not supported.")

        # prepare the messages
        system_message = ChatMessage(role="system", content=system_instruction + "\n" + QuestionAnswer.typescript2())
        question_message = ChatMessage(role="user", content=context + self.rendered_request)

        if self.fewshot:
            shots = [Shot(**s) for s in get_fixed_shots("medmcqa")]
            shot_messages = []
            for shot in shots:
                shot_messages.append(
                    ChatMessage(
                        role="user",
                        content=jinja2.Template(self.template["input"]).render(
                            question=shot.question, opa=shot.opa, opb=shot.opb, opc=shot.opc, opd=shot.opd
                        ),
                    )
                )

                extract_function = QuestionAnswer.from_shot(shot)
                content = f"function = {{\"name\": \"{extract_function.openai_schema['name']}\", \"arguments\": {extract_function.from_shot(shot).model_dump_json()}}}"  # type: ignore
                shot_messages.append(ChatMessage(role="assistant", content=content))

            messages = [system_message, *shot_messages, question_message]
        else:
            messages = [system_message, question_message]
        # query the client
        response = self.query(messages=messages, completion_config=completion_config)
        # extract the response and return it
        if self.prompt_type == 2:
            return extract_response(response, QuestionAnswer)
        else:
            return extract_response2(response, QuestionAnswer)

    def divide_and_summarize(
        self, content: str | None, max_segment_length: int = 8000, final_max_length: int = 5000
    ) -> str:
        """Divide the content into smaller segments, summarize each, and then summarize the combined summary if it's still too long."""
        # Divide the content into manageable segments
        if not content:
            return "<empty>"
        segments = wrap(content, max_segment_length)

        summarized_segments = []
        for segment in segments:
            summarized_segment = self.summarize(segment)
            summarized_segments.append(str(summarized_segment))

        # Combine the summarized segments
        combined_summary = " ".join(summarized_segments)

        # Further summarize if the combined summary is too long
        if len(combined_summary) > final_max_length:
            combined_summary = self.summarize(combined_summary)

        return combined_summary

    def summarize(self, content: str) -> str:
        """Summarize the content."""
        messages = [
            ChatMessage(
                role="system",
                content='You are a summarizer, your role is to summarize the content given the original question.\nSummarize the content concisely and accurately.\nIf the content provided is not relevant to the question, you must reply with "<empty>".',
            ),
            ChatMessage(
                role="user",
                content="Summarize this content: \n"
                + content
                + "\n given this original question: \n"
                + self.rendered_request,
            ),
        ]
        # Call the client
        response = self.query(messages=messages)
        return response


class FindZebraAgent(Agent):
    """An agent that calls the FindZebra API."""

    def __init__(self, client: BaseClient, dataset: str, request: dict[str, str]) -> None:
        self.findzebra = FindZebraAPI()
        super().__init__(client, dataset, request)

    def run(self) -> dict[str, typ.Any]:
        """Run the agent."""
        # Create search query for FindZebra API
        query = self.create_search_query()

        # print(query)
        # this is cheating...
        # if self.request["cop"] == 0:
        #     query = {"function": self.request["opa"]}
        # elif self.request["cop"] == 1:
        #     query = {"function": self.request["opb"]}
        # elif self.request["cop"] == 2:
        #     query = {"function": self.request["opc"]}
        # elif self.request["cop"] == 3:
        #     query = {"function": self.request["opd"]}

        context = self.divide_and_summarize(query["function"])

        # print(context)

        response = self.answer_question(context=context)

        # print(response)

        f = response.pop("function")
        prediction = f.answer.value if f else "None"
        explanation = f.explanation if f else "None"
        return {"prediction": prediction, "explanation": explanation, **response}

    def create_search_query(self) -> dict[str, typ.Any]:
        """Create a search query for the FindZebra API."""
        messages = [
            ChatMessage(
                role="system",
                content="\n".join(
                    [
                        "You are a search expert, your role is to create a search query for a medical database that will be used to answer the question.",
                        'You must respond with "function = " followed with valid JSON that follows the provided schema.',
                        "Example:",
                        'function = {"name": "SearchMedicalDatabase", "arguments": {"query": <query>}}',
                        "No other format is allowed.",
                        "",
                        "// Supported function definitions that must be used.",
                        "namespace functions {",
                        "",
                        "",
                        "// Query the FindZebra medical database.",
                        "type SearchMedicalDatabase = (_: {",
                        "// Query to search for in medical database",
                        "explanation: string,",
                        "}) => string;",
                        "",
                        "// namespace functions",
                    ]
                ),
            ),
            ChatMessage(role="user", content=self.rendered_request),
        ]
        # Call the client
        response = self.query(messages=messages)

        # print("query_raw", response)

        # clean the response by using the expected format
        # Regular expression pattern to find the function call
        pattern = r'function\s*=\s*\{"name":\s*"SearchMedicalDatabase",\s*"arguments":\s*\{"query":\s*".*?"\}\}'

        # Search for the pattern in the text
        match = re.search(pattern, response)
        if match:
            response = match.group(0)

        # I am expecting a function call as a response
        response = extract_response(response, self.findzebra.search)
        return response


class AnswerQuestionStructuredAgent(Agent):
    """An agent that utilizes the structure of the QuestionAnswer pydantic model."""

    def __init__(self, client: BaseClient, dataset: str, request: dict[str, str], prompt_type: int) -> None:
        self.findzebra = FindZebraAPI()
        super().__init__(client, dataset, prompt_type, request)

    def answer_question(self, context: str = "") -> tuple[str, dict[str, typ.Any]]:
        """Query the client with a question and possibly a context using the AnswerQuestion model.
        The messages and completion_config are dependent on the language model.
        """
        if self.model not in [
            "meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "codellama/CodeLlama-7b-Instruct-hf",
            "codellama/CodeLlama-13b-Instruct-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ]:
            raise ValueError(f"Model {self.model} is not supported.")

        # Model supported...

        if self.prompt_type == 1:
            system_instruction = "\n".join(
                [
                    "You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct.",
                    "You must respond with a function call.",
                    'Example: functions.QuestionAnswer({"explanation": <explanation>, "answer": <answer>})',
                    "No other format is allowed.",
                ]
            )
            system_message = ChatMessage(role="system", content=system_instruction + "\n" + QuestionAnswer.typescript())
            question_message = ChatMessage(role="user", content=context + self.rendered_request)
            messages = [system_message, question_message]

            response = self.query(messages=messages, guidance_str=" functions.", completion_config=CompletionConfig())  # type: ignore
            return extract_response3(response, QuestionAnswer)
        elif self.prompt_type == 2:
            system_instruction = "\n".join(
                [
                    "You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct.",
                    "You must respond with a function call.",
                    'Example: function = {"name": "QuestionAnswer", "arguments": {"explanation": <explanation>, "answer": <answer>}}',
                    "No other format is allowed.",
                ]
            )
            system_message = ChatMessage(
                role="system", content=system_instruction + "\n" + QuestionAnswer.typescript2()
            )
            question_message = ChatMessage(role="user", content=context + self.rendered_request)
            messages = [system_message, question_message]

            response = self.query(messages=messages, guidance_str=" function =", completion_config=CompletionConfig())  # type: ignore
            return extract_response2(response, QuestionAnswer)
        elif self.prompt_type == 3:
            system_instruction = "\n".join(
                [
                    "You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct.",
                    "You must respond with a function call.",
                ]
            )
            system_message = ChatMessage(
                role="system", content=system_instruction + "\n" + QuestionAnswer.typescript3()
            )
            question_message = ChatMessage(role="user", content=context + self.rendered_request)
            messages = [system_message, question_message]

            response = self.query(messages=messages, guidance_str=" functions.", completion_config=CompletionConfig())
            return extract_response0(response, QuestionAnswer)
        elif self.prompt_type == 4:
            system_instruction = "\n".join(
                [
                    "You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct.",
                    "You must respond with a function call.",
                    "Example: functions.<function_name>({<arguments>})",
                    "No other format is allowed.",
                ]
            )
            system_message = ChatMessage(
                role="system", content=system_instruction + "\n" + QuestionAnswer.typescript3()
            )
            question_message = ChatMessage(role="user", content=context + self.rendered_request)
            messages = [system_message, question_message]

            response = self.query(messages=messages, guidance_str=" functions.", completion_config=CompletionConfig())
            return extract_response3(response, QuestionAnswer)
        elif self.prompt_type == 12:
            system_instruction = "\n".join(
                [
                    "You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct.",
                    "You must respond with a function call.",
                ]
            )
            system_message = ChatMessage(
                role="system", content=system_instruction + "\n" + QuestionAnswer.typescript3()
            )
            question_message = ChatMessage(role="user", content=context + self.rendered_request)
            messages = [system_message, question_message]

            response = self.query(messages=messages, guidance_str=" functions.", completion_config=CompletionConfig())
            return extract_response0(response, QuestionAnswer)
        elif self.prompt_type == 11:
            system_instruction = "\n".join(
                [
                    "You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct.",
                    "You must respond with a function call.",
                    "Example: functions.<function_name> = {<arguments>};",
                    "No other format is allowed.",
                ]
            )
            system_message = ChatMessage(
                role="system", content=system_instruction + "\n" + QuestionAnswer.typescript3()
            )
            question_message = ChatMessage(role="user", content=context + self.rendered_request)
            messages = [system_message, question_message]

            response = self.query(messages=messages, guidance_str=" functions.", completion_config=CompletionConfig())
            return extract_response3(response, QuestionAnswer)
        elif self.prompt_type == -1:  # Not used anymore
            system_instruction = "\n".join(
                [
                    "You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct.",
                    "You must respond with a function call.",
                    'Example: functions.<function_name> = {"<argument1>": <value1>, "<argument2>": <value2>};',
                    "No other format is allowed.",
                ]
            )
            system_message = ChatMessage(
                role="system", content=system_instruction + "\n" + QuestionAnswer.typescript3()
            )
            question_message = ChatMessage(role="user", content=context + self.rendered_request)
            messages = [system_message, question_message]

            response = self.query(messages=messages, guidance_str=" functions.", completion_config=CompletionConfig())
            return extract_response3(response, QuestionAnswer)
        elif self.prompt_type == 0:
            system_instruction = "\n".join(
                [
                    "You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct.",
                    "You must respond with a function call.",
                    'Example: functions.AnswerQuestion = {"explanation": <explanation>, "answer": <answer>};',
                    "No other format is allowed.",
                ]
            )
            system_message = ChatMessage(role="system", content=system_instruction + "\n" + QuestionAnswer.typescript())
            question_message = ChatMessage(role="user", content=context + self.rendered_request)
            messages = [system_message, question_message]

            response = self.query(messages=messages, guidance_str=" functions.", completion_config=CompletionConfig())
            return extract_response0(response, QuestionAnswer)
        else:
            raise ValueError(f"Prompt type {self.prompt_type} is not supported.")

    def run(self) -> dict[str, typ.Any]:
        """Run the agent."""
        # Call the client
        raw_response, response = self.answer_question(context="")

        f = response.pop("function")
        prediction = f.answer.value if f else "None"
        explanation = f.explanation if f else raw_response
        return {"prediction": prediction, "explanation": explanation, **response}

    def request_correct_json(self, response: str) -> str:
        """Ask the model to correct the json response."""
        messages = [
            ChatMessage(
                role="system",
                content="""You are a JSON expert, your role is to correct the JSON response provided by the model. 
                You receive an invalid JSON string as input and you must return a valid JSON string as output. 
                You must reply with the corrected JSON string and only the corrected JSON string.
                No explanation is required.""",
            ),
            ChatMessage(role="user", content=response),
        ]
        # Call the client
        response = self.query(messages=messages)
        return response


def extract_response1(response: str, function: typ.Callable) -> tuple[str, dict[str, typ.Any]]:
    """Extract the response from the raw response."""
    valid_indicator = False
    valid_json = False
    valid_function_args = False
    f = None

    # print(response)

    if "functions.QuestionAnswer" in response:
        response = response.replace("functions.QuestionAnswer", "").replace("\n", "").strip(" ()")
        # print("VALID INDICATOR", response)
        valid_indicator = True
        if valid_json_check(response):
            valid_json = True
            d = json.loads(response)
            # print("VALID JSON: d: ", d)

            try:
                f = function(**d)
                valid_function_args = True
            except:  # noqa: E722
                # print("Invalid function arguments.")
                pass
        # else:
        # print("INVALID JSON: ", response)

        if not valid_json_check(response):
            pass
    return response, {
        "valid_indicator": valid_indicator,
        "valid_json": valid_json,
        "valid_function_args": valid_function_args,
        "function": f,
    }


def extract_response2(response: str, function: typ.Callable) -> tuple[str, dict[str, typ.Any]]:
    """Extract the response from the raw response."""
    valid_indicator = False
    valid_json = False
    valid_function_args = False
    f = None

    # print(response)

    if "function =" in response:
        response = response.replace("function =", "").strip()
        valid_indicator = True
        # print(response)
        if valid_json_check(response):
            valid_json = True
            d = json.loads(response)

            try:
                f = function(**d["arguments"])
                valid_function_args = True
            except:  # noqa: E722
                # print("Invalid function arguments.")
                pass

    return response, {
        "valid_indicator": valid_indicator,
        "valid_json": valid_json,
        "valid_function_args": valid_function_args,
        "function": f,
    }


def extract_response3(response: str, function: typ.Callable) -> tuple[str, dict[str, typ.Any]]:
    """Extract the response from the raw response."""
    valid_indicator = False
    valid_json = False
    valid_function_args = False
    f = None

    # print(response)

    if (
        "functions.AnswerQuestion" in response
        or "functions.answerQuestion" in response
        or "functions.QuestionAnswer" in response
    ):
        response = (
            response.replace("functions.AnswerQuestion", "")
            .replace("functions.answerQuestion", "")
            .replace("functions.QuestionAnswer", "")
        )
        response = response.replace("\n", "").strip(" ();'=")  # =
        # print("VALID INDICATOR", response)
        if "{explanation" in response:
            response = response.replace("{explanation", '{"explanation"')
        if "answer:" in response:
            response = response.replace("answer:", '"answer":')

        response = response.replace("`", '"')

        # response = response.replace("explanation", '"explanation"').replace("answer", '"answer"')
        if response.endswith(",}"):
            response = response.replace(",}", "}")
        valid_indicator = True
        if valid_json_check(response):
            valid_json = True
            d = json.loads(response)
            # print("VALID JSON: d: ", d)

            try:
                f = function(**d)
                valid_function_args = True
            except:  # noqa: E722
                # print("Invalid function arguments.")
                pass
        # else:
        # print("INVALID JSON: ", response)
        if not valid_json_check(response):
            pass

    return response, {
        "valid_indicator": valid_indicator,
        "valid_json": valid_json,
        "valid_function_args": valid_function_args,
        "function": f,
    }


def extract_response0(response: str, function: typ.Callable) -> tuple[str, dict[str, typ.Any]]:
    """Extract the response from the raw response."""
    valid_indicator = False
    valid_json = False
    valid_function_args = False
    f = None

    # print(response)

    if (
        "functions.AnswerQuestion" in response
        or "functions.answerQuestion" in response
        or "functions.QuestionAnswer" in response
    ):
        response = response.replace("\n", "")
        pattern = r"\{.*\}"
        match = re.search(pattern, response)
        if match:
            response = match.group(0)

        if "answer:" in response:
            response = response.replace("answer:", '"answer":')
        if "explanation:" in response:
            response = response.replace("explanation:", '"explanation":')
        if response.endswith(",}"):
            response = response.replace(",}", "}")
        response = response.replace("'", '"')

        valid_indicator = True
        if valid_json_check(response):
            valid_json = True
            d = json.loads(response)
            # print("VALID JSON: d: ", d)

            try:
                f = function(**d)
                valid_function_args = True
            except:  # noqa: E722
                # print("Invalid function arguments.")
                pass
        # else:
        # print("INVALID JSON: ", response)
    if not valid_json_check(response):
        pass

    return response, {
        "valid_indicator": valid_indicator,
        "valid_json": valid_json,
        "valid_function_args": valid_function_args,
        "function": f,
    }


def valid_json_check(response: str) -> bool:
    """Check if the response is a valid json."""
    try:
        json.loads(response)
    except json.JSONDecodeError:
        return False
    return True


class RavenAgent(Agent):
    """Raven agent."""

    def __init__(self, client: BaseClient, dataset: str, request: dict[str, str]) -> None:
        self.findzebra = FindZebraAPI()
        super().__init__(client, dataset, request)

    def run(self) -> dict[str, typ.Any]:  # type: ignore
        """Run the agent."""
        # Call the LLM
        raw_response = self.ask_LLM_with_tools()
        response = parse_function_calls(raw_response)
        if response is None:
            return {"explanation": "", "prediction": "NA", "tools": []}

        function_name, args = response[0]
        if function_name == "query_medical_database":
            query_response = self.findzebra.search(*args)

            # Call the LLM with the query response to get the reasoning
            response = parse_function_calls(self.ask_LLM_for_reasoning(context=query_response))
            if response is None:
                return {"explanation": "", "prediction": "NA", "tools": []}
            function_name, args = response[0]  # type: ignore
            if function_name == "reason_about_question":
                answer = self.ask_LLM_for_answer_option(thought=args)
                return {
                    "explanation": " ".join(args),
                    "prediction": answer,
                    "tools": ["medical_database_query", "reasoning", "answer"],
                }

        if function_name == "reason_about_question":
            if "query_medical_database" in " ".join(args):
                # execute the query_medical_database function
                # extract the query from the reasoning
                search_query = args[0].split("query_medical_database(query=")[1].strip().replace("'", "")
                query_response = self.findzebra.search(search_query)

                # Call the LLM with the query response to get the reasoning
                response = parse_function_calls(self.ask_LLM_for_reasoning(context=query_response))
                if response is None:
                    return {"explanation": "", "prediction": "NA", "tools": []}
                function_name, args = response[0]  # type: ignore
                if function_name == "reason_about_question":
                    answer = self.ask_LLM_for_answer_option(thought=args)
                    return {
                        "explanation": " ".join(args),
                        "prediction": answer,
                        "tools": ["medical_database_query", "reasoning", "answer"],
                    }

            answer = self.ask_LLM_for_answer_option(thought=args)
            return {"explanation": " ".join(args), "prediction": answer, "tools": ["reasoning", "answer"]}

    def ask_LLM_with_tools(self) -> str:
        prompt = f'''
Function:
def reason_about_question(reasoning):
    """
    Reason about the question.

    Args:
    reasoning (str): The reasoning that will be used to answer the question.
    """

Function:
def query_medical_database(query):
    """
    Search the rare disease medical database. 

    Args:
    query (str): The query to search for in the medical database.

    Returns:
    document (str): The document returned by the medical database.
    """


User Query: {self.rendered_request}<human_end>
'''

        response = self.query(messages=prompt, completion_config=CompletionConfig(temperature=0.001, stop=["<bot_end>"], skip_special_tokens=False))  # type: ignore
        cleaned_response = response.replace("Call:", "").strip()

        return cleaned_response

    def ask_LLM_for_reasoning(self, context="") -> str:
        prompt = f'''
Function:
def reason_about_question(reasoning):
    """
    Reason about the question and provide an explanation.

    Args:
    reasoning (str): The reasoning that will be used to answer the question.
    """

User Query: {context}\n{self.rendered_request}<human_end>
'''

        response = self.query(messages=prompt, completion_config=CompletionConfig(temperature=0.001, stop=["<bot_end>"], skip_special_tokens=False))  # type: ignore
        cleaned_response = response.replace("Call:", "").strip()

        return cleaned_response

    def ask_LLM_for_answer_option(self, thought: str = "") -> str:
        prompt: str = f'''
Function:
def answer_question(answer):
    """
    Answer the user's question in a structured way with an explanation and a final answer (A, B, C, or D)

    Args:
    answer (str): The answer to the question (A, B, C or D).
    """

User Query: {self.rendered_request}\n {thought}<human_end>
'''
        response = self.query(messages=prompt, completion_config=CompletionConfig(temperature=0.001, stop=["<bot_end>"], skip_special_tokens=False))  # type: ignore
        cleaned_response = response.replace("Call:", "").strip()
        # print(cleaned_response)

        dict_response: dict[str, str] = parse_function_call(cleaned_response)
        answer = dict_response.get("answer", "")
        # explanation = dict_response.get("explanation", "")
        # answer = dict_response.get("answer", "")
        return answer


def parse_function_call(string: str) -> dict:
    # Pattern to match function arguments and their values
    # This pattern assumes arguments are in the format: argument_name='value'
    pattern = r"(\w+)='([^']*)'"

    # Find all matches in the string
    matches = re.findall(pattern, string)

    # Convert matches to dictionary
    extracted_args = {arg_name: arg_value for arg_name, arg_value in matches}
    return extracted_args


def parse_function_calls(input_str):
    # Regex to match function calls
    function_pattern = r"(\w+)\(([^)]*)\)"

    def parse_args(args_str):
        # Recursive parsing of arguments
        args = []
        for match in re.finditer(function_pattern, args_str):
            function_name, arguments = match.groups()
            args.append((function_name, parse_args(arguments)))
        if not args:
            # If no more nested functions, split arguments by comma
            args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
        return args

    matches = re.finditer(function_pattern, input_str)
    result = []
    for match in matches:
        function_name, arguments = match.groups()
        result.append((function_name, parse_args(arguments)))
    if result:
        return result
    else:
        return None


# Function:
# def search_wikipedia(title):
#     """
#     Search Wikipedia for a query that is relevant to the question.
#
#     Args:
#     title (str): The title of the Wikipedia page to search for.
#     """


class ReActAgent(Agent):
    """Raven agent."""

    def __init__(self, client: BaseClient, dataset: str, request: dict[str, str]) -> None:
        self.wiki = Wikipedia()
        self.prompt = get_react_shots(dataset)
        super().__init__(client, dataset, request)

    def run(self) -> dict[str, typ.Any]:  # type: ignore
        """Run the agent."""
        # Call the LLM
        # {'accuracy': 0.332, 'react': {'success': 0.792, 'unknown_action_type': 0.001, 'too_long_chain': 0.203, 'action_and_thought_not_extracted': 0.004, 'avg_chain_length': 6.206}}

        message = self.rendered_request

        i = 1
        finish = False
        thoughts = []
        actions = []
        observations = []

        while finish is False:
            response = self.query(messages=self.prompt + message, guidance_str=f"Thought {i}:", completion_config=CompletionConfig(stop=["Observation"]))  # type: ignore

            # split the response into the thought and action
            parts = response.split("Action", 1)
            if len(parts) != 2:
                return {
                    "explanation": "",
                    "prediction": "NA",
                    "tools": [],
                    "code": {"status": 400, "message": "Not possible to extract thought and action from the response."},
                    "chain_length": i,
                    "thoughts": thoughts,
                    "actions": actions,
                    "last_response": response,
                }
            thought = response.split("Action", 1)[0]
            action = response.split("Action", 1)[1]

            thoughts.append(thought)
            actions.append(action)

            status, observation = self.execute_action(action)

            observations.append(observation)
            if status == 400:
                return {
                    "explanation": observation,
                    "prediction": "NA",
                    "tools": [],
                    "code": {"status": 400, "message": "Unknown action type."},
                    "chain_length": i,
                    "thoughts": thoughts,
                    "actions": actions,
                    "last_response": response,
                }
            elif status == 213:
                finish = True

            message += response + f"\nObservation {i}: " + observation

            i += 1

            if i > 10:
                return {
                    "explanation": "Too long chain.",
                    "prediction": "NA",
                    "tools": [],
                    "code": {"status": 400, "message": "Too long chain."},
                    "chain_length": i,
                    "thoughts": thoughts,
                    "actions": actions,
                    "last_response": response,
                }

        return {
            "explanation": message,
            "prediction": observation,
            "tools": [],
            "code": {"status": 200, "message": "Succesful."},
            "chain_length": i,
            "thoughts": thoughts,
            "actions": actions,
            "last_response": response,
        }  # type: ignore

    def execute_action(self, action: str) -> tuple[int, str]:
        """Execute an action based on the provided action string."""
        action_type, _, action_value = action.partition(":")[2].partition("[")
        action_value = action_value.strip("[]")

        if "Search" in action_type:
            return 200, self.wiki.search(action_value)
        elif "Lookup" in action_type:
            return 200, self.wiki.lookup(action_value)
        elif "Finish" in action_type:
            return 213, action_value
        else:
            return 400, f"Unknown action type. {action_type}"


class FewShotCoTAgent(Agent):
    def __init__(
        self,
        client: BaseClient,
        dataset: str,
        request: dict[str, str],
        context="",
        prompt_type=-1,
        two_step=False,
        summarize=False,
        sumfilter=False,
    ) -> None:
        self.context = context
        super().__init__(client, dataset, -1, request)

    def run(self) -> dict[str, typ.Any]:  # type: ignore
        message = self.rendered_request
        messages = [
            self.system_message,
            *get_cot_shots(self.dataset),
            ChatMessage(role="user", content=self.context + message),
        ]

        reasoning = self.query(messages=messages, guidance_str="\nAnswer: Let's think step by step", completion_config=CompletionConfig(max_tokens=1024, stop=["Question"]))  # type: ignore )

        # as foundation models return a CoT + final answer, and chat models needs to be prompted in a two-step manner
        if "chat" in self.model or "instruct" in self.model:
            # messages.append(ChatMessage(role="assistant", content="\nAnswer: Let's think step by step" + reasoning))
            # messages.append(ChatMessage(role="user", content="\nTherefore, among A through D the answer is"))
            answer = self.query(
                messages=messages, guidance_str="\nAnswer: Let's think step by step" + reasoning + "\nTherefore, among A through D the answer is", completion_config=CompletionConfig(max_tokens=20, stop=["Explanation", "Question"])  # type: ignore
            )
        else:
            answer = reasoning

        pattern = r"\(([ABCD])\)"
        prediction = extract_prediction(pattern=pattern, text=answer)

        return {
            "explanation": reasoning + answer,
            "prediction": prediction,
        }


class FewShotDirectAgent(Agent):
    def __init__(
        self,
        client: BaseClient,
        dataset: str,
        request: dict[str, str],
        context="",
        prompt_type=-1,
        two_step=False,
        summarize=False,
        sumfilter=False,
    ) -> None:
        super().__init__(client, dataset, -1, request)
        self.context = context

    def run(self) -> dict[str, typ.Any]:  # type: ignore
        # prompt = get_direct_shots(self.dataset)
        pattern = r"\(([ABCD])\)"
        message = self.rendered_request

        messages = [
            self.system_message,
            *get_direct_shots(self.dataset),
            ChatMessage(role="user", content=self.context + message),
        ]

        response = self.query(messages=messages, guidance_str="\nAnswer: The answer is", completion_config=CompletionConfig(stop=["Question"]))  # type: ignore )

        prediction = extract_prediction(text=response, pattern=pattern)

        return {
            "explanation": response,
            "prediction": prediction,
        }


class CoTAgent(Agent):
    def __init__(self, client: BaseClient, dataset: str, request: dict[str, str]) -> None:
        super().__init__(client, dataset, -1, request)

    def run(self) -> dict[str, typ.Any]:  # type: ignore
        """Run the agent."""
        # Call the LLM

        message = self.rendered_request

        # Parse message to a list of ChatMessages
        messages = [self.system_message, ChatMessage(role="user", content=message)]

        reasoning = self.query(
            messages=messages, guidance_str="\nAnswer: Let's think step by step", completion_config=CompletionConfig(max_tokens=1024, stop=["Question"])  # type: ignore
        )

        # messages.append(ChatMessage(role="assistant", content="\nAnswer: Let's think step by step" + reasoning))
        # messages.append(ChatMessage(role="user", content="\nTherefore, among A through D the answer is"))

        answer = self.query(
            messages=messages, guidance_str="\nAnswer: Let's think step by step" + reasoning + "\nTherefore, among A through D the answer is", completion_config=CompletionConfig(max_tokens=20, stop=["Explanation", "Question"])  # type: ignore
        )
        # message + "\nAnswer: Let's think step by step" + reasoning

        # get choices in raw text
        if self.dataset == "medmcqa":
            choices = [
                f"A) {self.request['opa']}",
                f"B) {self.request['opb']}",
                f"C) {self.request['opc']}",
                f"D) {self.request['opd']}",
            ]
        else:
            choices = [
                f"A) {self.request['options']['A']}",  # type: ignore
                f"B) {self.request['options']['B']}",  # type: ignore
                f"C) {self.request['options']['C']}",  # type: ignore
                f"D) {self.request['options']['D']}",  # type: ignore
            ]

        extract_answer = extract_multiple_choice_prediction(completion=answer, choices=choices)
        if extract_answer == -1 or extract_answer > 3:
            prediction = "NA"
        else:
            prediction = ["A", "B", "C", "D"][extract_answer]

        return {
            "explanation": reasoning + answer,
            "prediction": prediction,
        }


class DirectAgent(Agent):
    """Raven agent."""

    def __init__(
        self,
        client: BaseClient,
        dataset: str,
        request: dict[str, str],
        context="",
        prompt_type=-1,
        two_step=False,
        summarize=False,
        sumfilter=False,
    ) -> None:
        super().__init__(client, dataset, -1, request)
        self.context = context

    def run(self) -> dict[str, typ.Any]:  # type: ignore
        """Run the agent."""
        # Call the LLM

        message = self.rendered_request

        # Parse message to a list of ChatMessages
        messages = [self.system_message, ChatMessage(role="user", content=self.context + message)]

        response = self.query(
            messages=messages, guidance_str="\nAnswer: The answer is", completion_config=CompletionConfig(max_tokens=20, stop=["Explanation"])  # type: ignore
        )

        # get choices in raw text
        if self.dataset == "medmcqa":
            choices = [
                f"A) {self.request['opa']}",
                f"B) {self.request['opb']}",
                f"C) {self.request['opc']}",
                f"D) {self.request['opd']}",
            ]
        else:
            choices = [
                f"A) {self.request['options']['A']}",  # type: ignore
                f"B) {self.request['options']['B']}",  # type: ignore
                f"C) {self.request['options']['C']}",  # type: ignore
                f"D) {self.request['options']['D']}",  # type: ignore
            ]

        response = response.replace("\nAnswer: The answer is", "")

        extract_answer = extract_multiple_choice_prediction(completion=response, choices=choices)
        if extract_answer == -1 or extract_answer > 3:
            prediction = "NA"
        else:
            prediction = ["A", "B", "C", "D"][extract_answer]

        return {
            "explanation": response,
            "prediction": prediction,
        }
