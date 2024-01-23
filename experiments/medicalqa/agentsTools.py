import json
import re
import typing as typ
from textwrap import wrap
from typing import Any

from components.models import ChatMessage, CompletionConfig

from experiments.medicalqa.agents import Agent, DirectAgent, FewShotCoTAgent, FewShotDirectAgent, valid_json_check
from experiments.medicalqa.functions.findzebra import FindZebraAPI
from experiments.medicalqa.functions.wikipedia import WikipediaAPI
from experiments.medicalqa.prompt import QuestionAnswer
from src.clients.base import BaseClient


class BaseAgentTools(Agent):
    """Base agent for tools."""

    def __init__(
        self,
        client: BaseClient,
        dataset: str,
        request: dict[str, str],
        prompt_type: int,
        summarize: bool,
        two_step: bool,
        sumfilter: bool = False,
    ) -> None:
        self.findzebra = FindZebraAPI()
        self.wiki = WikipediaAPI(cache_dir=client.cache_dir, cache_reset=client.cache_reset)  # type: ignore
        self.system_message_healthcare = "You are a healthcare professional, your role is to provide expert responses to questions presented with four answer options where one stands out as the most correct."
        self.system_message_wiki = "You are a search expert, your role is to create a search query for Wikipedia that should return the most relevant article for the question."
        self.system_message_function = "\n".join(
            [
                "You must respond with a function call.",
                "Example: functions.<type_name>({<arguments>})",
                "No other format is allowed.",
            ]
        )
        self.valid = {}

        super().__init__(client, dataset, prompt_type, request)

    def answer_question(self, context: str = "") -> tuple[str, dict[str, Any]]:
        """Answer a question."""
        # return super().answer_question(context)
        system_message = ChatMessage(
            role="system",
            content=self.system_message_healthcare
            + "\n"
            + self.system_message_function
            + "\n"
            + QuestionAnswer.typescript(),
        )
        question_message = ChatMessage(role="user", content=context + self.rendered_request)
        messages = [system_message, question_message]

        response = self.query(
            messages=messages, guidance_str=" functions.", completion_config=CompletionConfig(stop=["functions"])  # type: ignore
        )

        return self.extract_response(response, QuestionAnswer, "QuestionAnswer")

    def search_wikipedia(
        self, divide_and_summarize: bool, two_step: bool = False, sumfilter: bool = False
    ) -> tuple[str, dict[str, Any]]:
        """ "Search wikipedia."""
        system_message = ChatMessage(
            role="system",
            content=self.system_message_wiki + "\n" + self.system_message_function + "\n" + self.wiki.typescript,
        )

        question_message = ChatMessage(role="user", content=self.rendered_request)
        messages = [system_message, question_message]

        response = self.query(
            messages=messages, guidance_str=" functions.", completion_config=CompletionConfig()  # type: ignore
        )
        if two_step:
            raw_query, query_result = self.extract_response(response, self.wiki.search_top_pages, "Wikipedia")
            if not query_result["function"]:
                # add dummy values to self.valid
                self.valid["WikipediaSelectTopArticle"] = {
                    "valid_indicator": False,
                    "valid_json": False,
                    "valid_function_args": False,
                    "not_none_output": False,
                }
                if sumfilter:
                    self.valid["AssessContext"] = {
                        "valid_indicator": False,
                        "valid_json": False,
                        "valid_function_args": False,
                        "not_none_output": False,
                    }
                return raw_query, query_result
            top_pages = query_result["function"]
            # ask LLM to select the top page
            system_message = ChatMessage(
                role="system",
                content="\n".join(
                    [
                        "You are a search expert, your role is to select the most relevant article for the question.",
                        self.system_message_function,
                        self.wiki.typescript_select_article,
                    ]
                ),
            )
            rendered_top_pages = "\n".join([f"{i}: {page}" for i, page in enumerate(top_pages)])
            question_message = ChatMessage(
                role="user", content="Articles: \n" + rendered_top_pages + "\n" + self.rendered_request
            )

            messages = [system_message, question_message]
            response = self.query(
                messages=messages, guidance_str=" functions.", completion_config=CompletionConfig()  # type: ignore
            )
            raw_top_page, top_page_result = self.extract_response(
                response, self.wiki.get_page_content_by_index, "WikipediaSelectTopArticle"
            )
            if not top_page_result["valid_function_args"]:
                if sumfilter:
                    self.valid["AssessContext"] = {
                        "valid_indicator": False,
                        "valid_json": False,
                        "valid_function_args": False,
                        "not_none_output": False,
                    }
                return raw_top_page, top_page_result
            result = top_page_result

        else:
            raw_query, query_result = self.extract_response(response, self.wiki.search, "Wikipedia")

            # if invalid function arguments, return raw query and query result -> context will be ""
            if not query_result["valid_function_args"] or not query_result["function"]:
                if sumfilter:
                    self.valid["AssessContext"] = {
                        "valid_indicator": False,
                        "valid_json": False,
                        "valid_function_args": False,
                        "not_none_output": False,
                    }
                return raw_query, query_result
            result = query_result

        if divide_and_summarize:
            context = self.divide_and_summarize_recursive(result["function"], sumfilter=sumfilter)
            # update query_result with context
            result["function"] = context

        return raw_query, result  # extract_response(response, self.wiki.search, "Wikipedia")

    def divide_and_summarize(
        self, content: str | None, max_segment_length: int = 8000, final_max_length: int = 5000, sumfilter: bool = False
    ) -> str:
        """Divide the content into smaller segments, summarize each, and then summarize the combined summary if it's still too long."""
        # Divide the content into manageable segments
        if not content:
            return ""
        segments = wrap(content, max_segment_length)

        summarized_segments = []
        for segment in segments:
            summarized_segment = self.summarize(segment)
            # if sumfilter:
            #     relevant = self.relevant_context(summarized_segment)
            #     if relevant:
            #         summarized_segments.append(str(summarized_segment))
            # else:
            summarized_segments.append(str(summarized_segment))

        # Combine the summarized segments
        combined_summary = " ".join(summarized_segments)

        # summarize the combined summaries.
        combined_summary = self.summarize(combined_summary)
        # if filter
        if sumfilter:
            relevant = self.relevant_context(combined_summary)
            if not relevant:
                combined_summary = ""

        return combined_summary

    def divide_and_summarize_recursive(
        self, content: str | None, final_max_length: int = 32000, sumfilter: bool = False
    ) -> str:
        """Divide the content into smaller segments, summarize each, and then summarize the combined summary if it's still too long."""
        # if no content, return empty string
        if not content:
            return ""

        if len(content) <= final_max_length:
            # summarize the content
            content = self.summarize(content)
            if sumfilter:
                relevant = self.relevant_context(content)
                if not relevant:
                    content = ""
            return content

        # divide the content into two segments
        max_segment_length = int(len(content) / 2)
        segments = wrap(content, max_segment_length)

        summarized_segments = []
        for segment in segments:
            summarized_segment = self.divide_and_summarize_recursive(segment, final_max_length, sumfilter)
            summarized_segments.append(str(summarized_segment))

        # Combine the summarized segments
        combined_summary = " ".join(summarized_segments)

        # summarize the combined summaries.
        combined_summary = self.summarize(combined_summary)
        if sumfilter:
            relevant = self.relevant_context(combined_summary)
            if not relevant:
                combined_summary = ""

        return combined_summary

    def summarize(self, content: str) -> str:
        """Summarize the content."""
        messages = [
            ChatMessage(
                role="system",
                content="\n".join(
                    [
                        "You are a summarizer, your role is to summarize the content given the question.",
                        "Summarize the content concisely and accurately.",
                        "If the content provided is not relevant to the question, respond with an empty string.",
                        "Do not answer the question.",
                    ]
                ),
            ),
            ChatMessage(
                role="user",
                content="Content: " + content + "\n given \n" + self.rendered_request,
            ),
        ]
        # Call the client

        response = self.query(messages=messages, guidance_str=" Summary: ", completion_config=CompletionConfig(stop=["Question", "Answer"]))  # type: ignore
        # remove the heading "Summary: " from the response
        response = response.replace(" Summary: ", "")
        return response

    def run(self) -> dict[str, typ.Any]:
        """Run the agent."""
        raw_response, response = self.answer_question(context="")
        f = response.pop("function")
        prediction = f.answer.value if f else "None"
        explanation = f.explanation if f else raw_response
        return {"prediction": prediction, "explanation": explanation, **response}

    def extract_response(self, raw_response: str, function: typ.Callable, function_name: str) -> tuple[str, dict]:
        """Extract the response from the raw response."""
        valid_indicator = False
        valid_json = False
        valid_function_args = False
        not_none_output = False
        f = None

        function_indicator = FUNCTION_INDICATORS[function_name]

        # test if one on the function indicators is in the response
        if any(indicator in raw_response for indicator in function_indicator):
            valid_indicator = True

            # sometimes the response ends with ) and not })
            if raw_response.endswith(")") and not raw_response.endswith("})"):
                raw_response = raw_response.replace(")", "})")

            # extract the function call with regex
            pattern = r"\{[\s\S]*\}"
            match = re.search(pattern, raw_response)
            if match:
                raw_response = match.group(0)
            response = raw_response

            # example string. '{\nexplanation: "The amount of "heat" required to change boiling water into vapor is referred to as Latent Heat of vaporization.",\nanswer: "A",\n}'
            # remove all double quotes, quotes, newlines and trailing commas
            response = response.replace('"', "").replace("\n", "").replace(",}", "}").replace("'", "").replace("`", "")
            # add double quotes to the arguments
            # now: '{explanation: The amount of heat required to change boiling water into vapor is referred to as Latent Heat of vaporization.,answer: A}'
            for i, args in enumerate(FUNCTION_ARGUMENTS[function_name]):
                if i == 0:
                    response = response.replace(args + ": ", '"' + args + '": "')
                else:
                    response = response.replace("," + args + ": ", '","' + args + '": "')
            response = response.replace("}", '"}')

            if valid_json_check(response):
                valid_json = True
                d = json.loads(response)
                # print("VALID JSON: d: ", d)

                try:
                    f = function(**d)
                    valid_function_args = True
                    if f is not None:
                        not_none_output = True
                except:  # noqa: E722, S110
                    # print("Invalid function arguments.")
                    pass  # noqa: E701
        self.valid[function_name] = {
            "valid_indicator": valid_indicator,
            "valid_json": valid_json,
            "valid_function_args": valid_function_args,
            "not_none_output": not_none_output,
        }

        # return result
        return raw_response, {
            "valid_indicator": valid_indicator,
            "valid_json": valid_json,
            "valid_function_args": valid_function_args,
            "not_none_output": not_none_output,
            "function": f,
        }

    def relevant_context(self, context: str) -> bool:
        # ask model if the summarized segment is relevant for the question
        system_message_content = "\n".join(
            [
                "You are a search expert, your role is to filter out irrelevant content for the question.",
                self.system_message_function,
                "namespace functions {",
                "",
                "// Assess whether the provided context is relevant for the question.",
                "type AssessContext = (_: {",
                "// Justification for the answer.",
                "justification: string,",
                "// Whether the context is relevant for the question.",
                "relevant: boolean,",
                "}) => any;",
                "",
                "} // namespace functions",
            ]
        )
        system_message = ChatMessage(role="system", content=system_message_content)
        question_message = ChatMessage(
            role="user", content="Context: " + context + "\n given\n" + self.rendered_request
        )
        messages = [system_message, question_message]
        response = self.query(
            messages=messages, guidance_str=" functions.", completion_config=CompletionConfig()  # type: ignore
        )
        raw_assessment, assessment_result = self.extract_response(response, self.assess_context, "AssessContext")

        relevant = assessment_result["function"]
        return relevant

    def assess_context(self, justification: str, relevant: str) -> str | None:
        """Assess whether the context is relevant for the question."""
        if relevant == "true":
            rel = "yes"
        elif relevant == "false":
            rel = None
        else:
            rel = None
        return rel


class WikipediaAgent(BaseAgentTools):
    def __init__(
        self,
        client: BaseClient,
        dataset: str,
        request: dict[str, str],
        summarize: bool,
        sumfilter: bool,
        two_step: bool,
        prompt_type: int = -1,
    ) -> None:
        super().__init__(
            client,
            dataset,
            request,
            prompt_type=prompt_type,
            summarize=summarize,
            two_step=two_step,
            sumfilter=sumfilter,
        )
        self.divide_and_summarize_toggle = summarize
        self.two_step = two_step
        self.sumfilter = sumfilter

    def run(self) -> dict[str, typ.Any]:
        """Run the agent."""
        # Ask for wikipedia search query
        query, wiki_response = self.search_wikipedia(
            divide_and_summarize=self.divide_and_summarize_toggle, two_step=self.two_step, sumfilter=self.sumfilter
        )

        # if successful wiki search, return the result as context
        context = wiki_response["function"] if wiki_response["function"] else ""

        raw_response, response = self.answer_question(context=context)
        f = response.pop("function")
        prediction = f.answer.value if f else "None"
        explanation = f.explanation if f else raw_response

        # if invalid function output - use FewShotCoT approach.
        # init fewshotcot agent
        # if prediction == "None":
        #     fewshotcot_agent = FewShotCoTAgent(
        #         self.client, self.dataset, self.request, context="Context: " + context + "\n"
        #     )
        #     explanation, prediction = fewshotcot_agent.run().values()
        return {"prediction": prediction, "explanation": explanation, "validity": self.valid}


class WikipediaCoTAgent(WikipediaAgent):
    def __init__(
        self,
        client: BaseClient,
        dataset: str,
        request: dict[str, str],
        prompt_type: int = -1,
        summarize: bool = False,
        sumfilter: bool = False,
        two_step: bool = False,
    ) -> None:
        super().__init__(client, dataset, request, summarize=summarize, two_step=two_step, sumfilter=sumfilter)

    def run(self) -> dict[str, typ.Any]:
        """Run the agent."""
        # Ask for wikipedia search query
        if (
            "A 37-year-old-woman presents to her primary care physician requesting a new form of birth control. She has been utilizing oral contraceptive pills (OCPs) for the past 8 years, but asks to switch to an intrauterine device (IUD)."
            in self.request["question"]
        ):
            print("here")

        query, wiki_response = self.search_wikipedia(
            divide_and_summarize=self.divide_and_summarize_toggle, two_step=self.two_step, sumfilter=self.sumfilter
        )

        # if successful wiki search, return the result as context
        context = wiki_response["function"] if wiki_response["function"] else ""

        fewshotcot_agent = FewShotCoTAgent(
            self.client, self.dataset, self.request, context="Context: " + context + "\n"
        )
        explanation, prediction = fewshotcot_agent.run().values()
        return {"prediction": prediction, "explanation": explanation, "validity": self.valid, "context": context}


class WikipediaFewShotDirectAgent(WikipediaAgent):
    def __init__(
        self,
        client: BaseClient,
        dataset: str,
        request: dict[str, str],
        prompt_type: int = -1,
        summarize: bool = False,
        sumfilter: bool = False,
        two_step: bool = False,
    ) -> None:
        super().__init__(client, dataset, request, summarize=summarize, two_step=two_step, sumfilter=sumfilter)

    def run(self) -> dict[str, typ.Any]:
        """Run the agent."""
        # Ask for wikipedia search query
        query, wiki_response = self.search_wikipedia(
            divide_and_summarize=self.divide_and_summarize_toggle, two_step=self.two_step, sumfilter=self.sumfilter
        )

        # if successful wiki search, return the result as context
        context = wiki_response["function"] if wiki_response["function"] else ""

        fewshotdirect_agent = FewShotDirectAgent(
            self.client, self.dataset, self.request, context="Context: " + context + "\n"
        )
        explanation, prediction = fewshotdirect_agent.run().values()
        return {"prediction": prediction, "explanation": explanation, "validity": self.valid, "context": context}


class WikipediaDirectAgent(WikipediaAgent):
    def __init__(
        self,
        client: BaseClient,
        dataset: str,
        request: dict[str, str],
        prompt_type: int = -1,
        summarize: bool = False,
        sumfilter: bool = False,
        two_step: bool = False,
    ) -> None:
        super().__init__(client, dataset, request, summarize=summarize, two_step=two_step, sumfilter=sumfilter)

    def run(self) -> dict[str, typ.Any]:
        """Run the agent."""
        # Ask for wikipedia search query
        query, wiki_response = self.search_wikipedia(
            divide_and_summarize=self.divide_and_summarize_toggle, two_step=self.two_step, sumfilter=self.sumfilter
        )

        # if successful wiki search, return the result as context
        context = wiki_response["function"] if wiki_response["function"] else ""

        direct_agent = DirectAgent(self.client, self.dataset, self.request, context="Context: " + context + "\n")
        explanation, prediction = direct_agent.run().values()
        return {"prediction": prediction, "explanation": explanation, "validity": self.valid, "context": context}


# class WikipediaSummarizeAgent(WikipediaAgent):
#     def __init__(
#         self,
#         client: BaseClient,
#         dataset: str,
#         request: dict[str, str],
#         prompt_type: int = -1,
#         divide_and_summarize: bool = False,
#         two_step: bool = False,
#     ) -> None:
#         super().__init__(client, dataset, request, divide_and_summarize=divide_and_summarize, two_step=two_step)


FUNCTION_INDICATORS = {
    "QuestionAnswer": [
        # "functions.AnswerQuestion",
        # "functions.answerQuestion",
        # "functions.QuestionAnswer",
        # "functions.questionAnswer",
        "functions.",
    ],
    "Wikipedia": [
        "functions.",
    ],
    "WikipediaSelectTopArticle": [
        "functions.",
    ],
    "AssessContext": [
        "functions.",
    ],
}
FUNCTION_ARGUMENTS = {
    "QuestionAnswer": ["explanation", "answer"],
    "Wikipedia": ["query"],
    "WikipediaSelectTopArticle": ["index"],
    "AssessContext": ["justification", "relevant"],
}
