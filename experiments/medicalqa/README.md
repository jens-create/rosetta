# 

Try prompt:


//Sys: "you are..."
{{question}}

Supported {{functions}}


//User:
Generate the json schema choosing the right function that will answer the question



commit where experiments are working: dfa349b


if no. samples not specified -> samples = 1000


llama7b-chat, 5-shot, valid questionAnswer only, "function = {"name": "QuestionAnswer", "arguments": {"explanation": "explain the answer", "answer": "B"}}"
{'accuracy': 0.315, 'avg_validJSON': 0.762, 'avg_validQuestionAnswer': 0.757}

llama13b-chat, 5-shot, valid questionAnswer only, "function = {"name": "QuestionAnswer", "arguments": {"explanation": "explain the answer", "answer": "B"}}"
{'accuracy': 0.419, 'avg_validJSON': 0.965, 'avg_validQuestionAnswer': 0.962}
{'accuracy': 0.411, 'avg_validJSON': 0.95, 'avg_validQuestionAnswer': 0.946}

llama70b-chat, 5-shot, valid questionAnswer only @100 samples, "function = {"name": "QuestionAnswer", "arguments": {"explanation": "explain the answer", "answer": "B"}}"
{'accuracy': 0.38, 'avg_validJSON': 0.87, 'avg_validQuestionAnswer': 0.87}

llama70b-chat, 5-shot, valid questionAnswer only @1000 samples, "function = {"name": "QuestionAnswer", "arguments": {"explanation": "explain the answer", "answer": "B"}}"
{'accuracy': 0.415, 'avg_validJSON': 0.811, 'avg_validQuestionAnswer': 0.806}


llama7b-chat 5-shot, valid questionAnswer only, "to=functions.QuestionAnswer: {"explanation": "explain the answer", "answer": "B"}
{'accuracy': 0.3, 'avg_validJSON': 0.745, 'avg_validQuestionAnswer': 0.74}

llama13b-chat 5-shot, valid questionAnswer only, "to=functions.QuestionAnswer: {"explanation": "explain the answer", "answer": "B"}
{'accuracy': 0.416, 'avg_validJSON': 0.908, 'avg_validQuestionAnswer': 0.905}

llama70b-chat 5-shot, valid questionAnswer only, "to=functions.QuestionAnswer: {"explanation": "explain the answer", "answer": "B"} (1000 samples)
{'accuracy': 0.049, 'avg_validJSON': 0.096, 'avg_validQuestionAnswer': 0.096}





Idéer: 
- Spørg den funtionen om at rette JSON til således at det bliver valid (virker til at det ret nemt kan blive valid.) Det er ikke så nemt haha.
- Prøv med 70b model
- Mine baselines skal kunne køre på den refaktoreringen.
- FindZebra context kan være for lang -> måske PCA (summarize the context). Paper: S2A https://arxiv.org/abs/2311.11829


\n\nExplanation:\nIf


