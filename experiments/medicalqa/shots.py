from components.models import ChatMessage

ANSWER_OPTION_KEYS = ["(A)", "(B)", "(C)", "(D)"]


def get_fixed_shots(dataset_name: str) -> list[dict]:
    """Get fixed shots."""
    return FIXED_SHOTS[dataset_name]


def get_react_shots(dataset: str) -> str:
    """Get fixed shots."""
    if dataset == "medmcqa":
        return format_shots_for_react(REACT)

    raise ValueError(f"Dataset {dataset} not supported.")


def get_empty_shots() -> list:
    """Get empty shots."""
    return []


def get_cot_shots(dataset: str) -> list[ChatMessage]:
    """Get fixed shots."""
    if dataset == "medmcqa":
        return format_shots_for_cot(MEDMCQA)
    if dataset == "GBaker/MedQA-USMLE-4-options":
        return format_shots_for_cot(MEDQA)

    raise ValueError(f"Dataset {dataset} not supported.")


def get_direct_shots(dataset: str) -> list[ChatMessage]:
    """Get fixed shots."""
    if dataset == "medmcqa":
        return format_shots_for_direct(MEDMCQA)
    if dataset == "GBaker/MedQA-USMLE-4-options":
        return format_shots_for_direct(MEDQA)

    raise ValueError(f"Dataset {dataset} not supported.")


def format_shots_for_direct(shots: list[dict]) -> list[ChatMessage]:
    """Format shots for direct."""
    # prompt = ""
    # for shot in shots:
    #     prompt += f"Question: {shot['question']}\n"
    #     prompt += f"Answer options: A) {shot['choices'][0]}\nB) {shot['choices'][1]}\nC) {shot['choices'][2]}\nD) {shot['choices'][3]}\n"
    #     prompt += f"Answer: {ANSWER_OPTION_KEYS[shot['target']]}\n"

    s = []
    for shot in shots:
        s.append(
            ChatMessage(
                role="user",
                content=f"Question: {shot['question']}\nAnswer options: A) {shot['choices'][0]}\nB) {shot['choices'][1]}\nC) {shot['choices'][2]}\nD) {shot['choices'][3]}\n",
            )
        )
        s.append(ChatMessage(role="assistant", content=f"Answer: The answer is {ANSWER_OPTION_KEYS[shot['target']]}\n"))

    return s


def format_shots_for_cot(shots: list[dict]) -> list[ChatMessage]:
    """Format shots for cot."""
    # prompt = ""
    # for shot in shots:
    #     prompt += f"Question: {shot['question']}\n"
    #     prompt += f"Answer options: A) {shot['opa']}\nB) {shot['opb']}\nC) {shot['opc']}\nD) {shot['opd']}\n"
    #     prompt += f"Answer: Let's think step by step {shot['explanation']}\n"
    #     prompt += f"Therefore, among A through D, the answer is {ANSWER_OPTION_KEYS[shot['target']]}\n"

    s = []
    for shot in shots:
        s.append(
            ChatMessage(
                role="user",
                content=f"Question: {shot['question']}\nAnswer options: A) {shot['choices'][0]}\nB) {shot['choices'][1]}\nC) {shot['choices'][2]}\nD) {shot['choices'][3]}\n",
            )
        )
        s.append(
            ChatMessage(
                role="assistant",
                content=f"Answer: Let's think step by step {shot['explanation']}\nTherefore, among A through D, the answer is {ANSWER_OPTION_KEYS[shot['target']]}\n",
            )
        )
        # s.append(ChatMessage(role="user", content="Therefore, among A through D, the answer is"))
        # s.append(ChatMessage(role="assistant", content=f"{ANSWER_OPTION_KEYS[shot['target']]}\n"))

    return s


def format_shots_for_react(shots: list[dict]) -> str:
    """Format shots for react."""
    prompt = ""
    for shot in shots:
        prompt += f"Question: {shot['question']}\n"
        prompt += f"Answer options: A) {shot['opa']}\nB) {shot['opb']}\nC) {shot['opc']}\nD) {shot['opd']}\n"
        for i, (thought, action, observation) in enumerate(
            zip(shot["react"]["thought"], shot["react"]["action"], shot["react"]["observation"])
        ):
            prompt += f"Thought {i + 1}: {thought}\n"
            prompt += f"Action {i + 1}: {action}\n"
            prompt += f"Observation {i + 1}: {observation}\n"

    return prompt


MEDQA = [
    {
        "question": "A 22-year-old male marathon runner presents to the office with the complaint of right-sided rib pain when he runs long distances. Physical examination reveals normal heart and lung findings and an exhalation dysfunction at ribs 4-5 on the right. Which of the following muscles or muscle groups will be most useful in correcting this dysfunction utilizing a direct method?",
        "choices": ["Anterior scalene", "Latissimus dorsi", "Pectoralis minor", "quadratus lumborum"],
        "target": 2,
        "explanation": "We refer to Wikipedia articles on medicine for help. Among the options, only pectoralis minor muscle origins from the outer surfaces of the 3rd to 5th ribs.",
    },
    {
        "question": "A 36-year-old male presents to the office with a 3-week history of low back pain. He denies any recent trauma but says that he climbs in and out of his truck numerous times a day for his job. Examination of the patient in the prone position reveals a deep sacral sulcus on the left, a posterior inferior lateral angle on the right, and a lumbosacral junction that springs freely on compression. Which of the following is the most likely diagnosis?",
        "choices": [
            "Left-on-left sacral torsion",
            "Left-on-right sacral torsion",
            "Right unilateral sacral flexion",
            "Right-on-right sacral torsion",
        ],
        "target": 3,
        "explanation": "We refer to Wikipedia articles on medicine for help. The deep sulcus on the left, a posterior ILA on the right, with a negative spring test suggests a right-on-right sacral torsion. All other options have a deep sulcus on the right.",
    },
    {
        "question": "A 44-year-old man comes to the office because of a 3-day history of sore throat, nonproductive cough, runny nose, and frontal headache. He says the headache is worse in the morning and ibuprofen does provide some relief. He has not had shortness of breath. Medical history is unremarkable. He takes no medications other than the ibuprofen for pain. Vital signs are temperature 37.4°C (99.4°F), pulse 88/min, respirations 18/min, and blood pressure 120/84 mm Hg. Examination of the nares shows erythematous mucous membranes. Examination of the throat shows erythema and follicular lymphoid hyperplasia on the posterior oropharynx. There is no palpable cervical adenopathy. Lungs are clear to auscultation. Which of the following is the most likely cause of this patient's symptoms?",
        "choices": ["Allergic rhinitis", "Rhinovirus", "Epstein-Barr virus", "Mycoplasma pneumonia"],
        "target": 1,
        "explanation": "We refer to Wikipedia articles on medicine for help. The symptoms, especially the headache, suggest that the most likely cause is Rhinovirus. Epstein-Barr virus will cause swollen lymph nodes but there is no palpable cervical adenopathy. Lungs are clear to auscultation suggests it's not Mycoplasma pneumonia.",
    },
    {
        "question": "A previously healthy 32-year-old woman comes to the physician 8 months after her husband was killed in a car crash. Since that time, she has had a decreased appetite and difficulty falling asleep. She states that she is often sad and cries frequently. She has been rechecking the door lock five times before leaving her house and has to count exactly five pieces of toilet paper before she uses it. She says that she has always been a perfectionist but these urges and rituals are new. Pharmacotherapy should be targeted to which of the following neurotransmitters?",
        "choices": ["Serotonin", "Dopamine", "Glutamate", "Norepinephrine"],
        "target": 0,
        "explanation": "We refer to Wikipedia articles on medicine for help. The patient feels sad and among the options, only Dopamine and Serotonin can help increase positive emotions. Serotonin also affects digestion and metabolism, which can help the patient's decreased appetite and sleep difficulty.",
    },
    {
        "question": "A 42-year-old man comes to the office for preoperative evaluation prior to undergoing adrenalectomy scheduled in 2 weeks. One month ago, he received care in the emergency department for pain over his right flank following a motor vehicle collision. At that time, blood pressure was 160/100 mm Hg and CT scan of the abdomen showed an incidental 10-cm left adrenal mass. Results of laboratory studies, including complete blood count, serum electrolyte concentrations, and liver function tests, were within the reference ranges. The patient otherwise had been healthy and had never been told that he had elevated blood pressure. He takes no medications. A follow-up visit in the office 2 weeks ago disclosed elevated urinary normetanephrine and metanephrine and plasma aldosterone concentrations. The patient was referred to a surgeon, who recommended the adrenalectomy. Today, vital signs are temperature 36.6°C (97.9°F), pulse 100/min, respirations 14/min, and blood pressure 170/95 mm Hg. Physical examination discloses no significant findings. Initial preoperative preparation should include treatment with which of the following?",
        "choices": ["Labetalol", "A loading dose of potassium chloride", "Phenoxybenzamine", "Nifedipine"],
        "target": 2,
        "explanation": "We refer to Wikipedia articles on medicine for help. The symptoms and the adrenal mass suggested pheochromocytoma, and the blood pressure indicates hypertension. Phenoxybenzamine is used to treat hypertension caused by pheochromocytoma.",
    },
]


MEDMCQA = [
    {
        "question": "In Gaucher's diseases, there is deficiency of:",
        "choices": ["Glucocerebrosidase", "Glucokinase", "Sphingomyelinase", "G-6PD"],
        "opa": "Glucocerebrosidase",
        "opb": "Glucokinase",
        "opc": "Sphingomyelinase",
        "opd": "G-6PD",
        "target": 0,
        "explanation": "Gaucher's disease is a rare genetic disorder that results from a deficiency of the enzyme glucocerebrosidase. The disease is characterized by the accumulation of a type of fat called glucocerebroside in cells and organs, particularly in the liver, spleen, and bone marrow. As a result, affected individuals may experience enlargement of these organs, anemia, low platelet count, and bone pain or fractures.",
    },
    {
        "question": "Which muscle is an abductor of the vocal cords?",
        "choices": ["Transverse arytenoid", "Oblique arytenoid", "Lateral thyroarytenoid", "Posterior Cricoarytenoid"],
        "opa": "Transverse arytenoid",
        "opb": "Oblique arytenoid",
        "opc": "Lateral thyroarytenoid",
        "opd": "Posterior Cricoarytenoid",
        "target": 3,
        "explanation": "The following movements are associated with specific muscles: elevation of the larynx is performed by the thyrohyoid and mylohyoid muscles, depression of the larynx is performed by the sternothyroid and sternohyoid muscles, opening the inlet of the larynx is performed by the thyroepiglottic muscle, and closing the inlet of the larynx is performed by the aryepiglottic muscle. The abductor of the vocal cords is the posterior cricoarytenoid muscle, while the lateral cricoarytenoid, transverse, and oblique arytenoid muscles are responsible for adducting the vocal cords.",
    },
    {
        "question": "Inner ear is present in which bone:",
        "choices": [
            "Parietal bone",
            "Petrous part of temporal bone",
            "Occipital bone",
            "Petrous part of squamous bone",
        ],
        "opa": "Parietal bone",
        "opb": "Petrous part of temporal bone",
        "opc": "Occipital bone",
        "opd": "Petrous part of squamous bone",
        "target": 1,
        "explanation": 'The inner ear is housed within a structure known as the bony labyrinth or otic capsule, which is a type of cartilaginous bone that forms from a cartilage model through endochondral bone formation. The bony labyrinth or otic capsule is located in the petrous part of the temporal bone. The petrous part is called "petrous" because it is one of the densest bones in the body, though not the strongest.',
    },
    {
        "question": "Blood flow in intervillous space at term:",
        "choices": ["150 ml", "250 ml", "350 ml", "450 ml"],
        "opa": "150 ml",
        "opb": "250 ml",
        "opc": "350 ml",
        "opd": "450 ml",
        "target": 0,
        "explanation": "The mature placenta has a total of 500 ml of blood. Out of this, 150 ml of blood is present in the intervillous space, while the remaining 350 ml of blood is present in the villi system.",
    },
    {
        "question": "Spuriously high BP is seen in A/E:",
        "choices": ["Auscultatory gap", "Small cuff", "Thick calcified vessels", "Obesity"],
        "opa": "Auscultatory gap",
        "opb": "Small cuff",
        "opc": "Thick calcified vessels",
        "opd": "Obesity",
        "target": 0,
        "explanation": "Spuriously high blood pressure readings can be observed in various conditions. These include the auscultatory gap, which is a brief silence heard during measurement due to the disappearance of Korotkoff sounds, often found in hypertensive patients. Additionally, using a small cuff that is not appropriately sized for the patient can lead to falsely elevated readings. Obese individuals may also falsely experience high blood pressure due to the thick layer of fat that can dissipate cuff pressure, resulting in inaccurate readings. Lastly, patients with thick and calcified vessels may experience falsely high blood pressure readings.",
    },
]

FIXED_SHOTS = {
    "medqa": MEDQA,
    "medmcqa": MEDMCQA,
    "dummy_medmcqa": MEDMCQA,
}


REACT = [
    {
        "id": "51250c61-30fe-47d0-b5cf-e9ce5abcac6e",
        "split": "train",
        "question": "Increase in pauwel's angle indicate?",
        "opa": "Good prognosis",
        "opb": "Impaction",
        "opc": "More chances of displacement",
        "opd": "Trabecular alignment displacement",
        "target": 2,
        "react": {
            "thought": [
                "The question is about an increase in pauwel's angle. Therefore, I need to search pauwel’s angle",
                "The search result does not mention increase in angle. Perhaps I can find the relevant information if I lookup increase.",
                "As an increasing angle leads to a more unstable fracture, it does not indicate a good prognosis. Therefore answer option A is ruled out. In this context, impaction refers to the degree of which the fractured bone ends are driving into each other. As the pauwel’s angle indicates the fracture’s orientation rather than the degree of impaction, answer option B is also ruled out. The answer is either C or D, I’ll have to search for Trabecular.",
                "It does not seem like Trabecular alignment displacement is not directly related to the implications of an increased pauwel’s angle. Therefore, the answer must be C) More chances of displacement",
            ],
            "action": ["Search[pauwel’s angle]", "Lookup[increase]", "Search[Trabecula]", "Finish[C]"],
            "observation": [
                "Pauwel's angle is the angle between the line of a fracture of the neck of the femur and the horizontal as seen on an anterio-posterior radiograph. Pauwel's angle is named after the German orthopedist Friedrich Pauwels.     Introduced in 1935, this system was the first biomechanical classification for femoral neck fractures, and is still in use.",
                "(Result 1 / 1) == Clinical Use ==\nAn increasing angle leads to a more unstable fracture and an increase in the shear stress at the fracture site",
                "A trabecula (pl.: trabeculae, from Latin for 'small beam') is a small, often microscopic, tissue element in the form of a small beam, strut or rod that supports or anchors a framework of parts within a body or organ. A trabecula generally has a mechanical function, and is usually composed of dense collagenous tissue (such as the trabecula of the spleen).  It can be composed of other material such as muscle and bone. In the heart, muscles form trabeculae carneae and septomarginal trabeculae",
                "",
            ],
        },
    },
    {
        "id": "053c4aff-1541-4fbd-8f3c-32fa0921e82a",
        "split": "train",
        "question": "Which of the following is the main enzyme responsible for activation of xenobiotics?",
        "opa": "Cytochrome P-450",
        "opb": "Glucuronyl transferase",
        "opc": "Glutathione S-transferase",
        "opd": "NADPH cytochrome P-450-reductase",
        "target": 0,
        "react": {
            "thought": [
                "Let's think step by step. The question is about which enzyme is responsible for the activation of xenobiotics. Therefore, I need to search xenobiotics and find which enzyme mainly activates it.",
                "I need to choose one of the available search titles. I see that the page is Xenobiotic, therefore I need to search Xenobiotic",
                "The search result does not help me in answering the question. Therefore I lookup enzyme.",
                "The lookup mentions that hepatic enzymes are responsible for the metabolism of xenobiotics. Furthermore, it mentions that an example of a group of enzymes involved in xenobiotic metabolism is hepatic microsomal cytochrome P450. Therefore, the answer is A) Cytochrome P-450.",
            ],
            "action": ["Search[xenobiotics]", "Search[Xenobiotic]", "Lookup[enzyme]", "Finish[A]"],
            "observation": [
                "Could not find [xenobiotics]. Similar: ['Xenobiotic', 'Drug metabolism', 'Environmental xenobiotic', 'Xenobiotic-transporting ATPase', 'Metabolism']",
                "A xenobiotic is a chemical substance found within an organism that is not naturally produced or expected to be present within the organism.  It can also cover substances that are present in much higher concentrations than are usual. Natural compounds can also become xenobiotics if they are taken up by another organism, such as the uptake of natural human hormones by fish found downstream of sewage treatment plant outfalls, or the chemical defenses produced by some organisms as protection against predators. The term xenobiotic is also used to refer to organs transplanted from one species to another.\nThe term xenobiotics, however, is very often used in the context of pollutants such as dioxins and polychlorinated biphenyls and their effect on the biota, because xenobiotics are understood as substances foreign to an entire biological system, i",
                "(Result 1 / 3) Hepatic enzymes are responsible for the metabolism of xenobiotics by first activating them (oxidation, reduction, hydrolysis, and/or hydration of the xenobiotic), and then conjugating the active secondary metabolite with glucuronic acid, sulfuric acid, or glutathione, followed by excretion in bile or urine. (Result 2 / 3) An example of a group of enzymes involved in xenobiotic metabolism is hepatic microsomal cytochrome P450. (Result 3 / 3) These enzymes that metabolize xenobiotics are very important for the pharmaceutical industry because they are responsible for the breakdown of medications",
                "",
            ],
        },
    },
    {
        "id": "fa2a40f5-7d07-473b-9175-f891e8923e19",
        "split": "train",
        "question": "Congenital hydrocele is best treated by?",
        "opa": "Eversion of sac",
        "opb": "Excision of sac",
        "opc": "Lord's procedure",
        "opd": "Herniotomy",
        "target": 3,
        "react": {
            "thought": [
                "I need to search Congenital hydrocele",
                "I need to choose one of the search results. Hydrocele seems to be the most appropiate search result",
                "The search result does not seem to be directly related to the question. Therefore, I need to look up the treatment",
                "The lookup mentions that the treatment of choice is surgery and the operation is conducted via an open access technique aiming to excise the hydrocele sac. Therefore, the answer is D) Herniotomy.",
            ],
            "action": [
                "Search[Congenital hydrocele]",
                "Search[Hydrocele]",
                "Lookup[treatment]",
                "Finish[D]",
            ],
            "observation": [
                "Could not find [Congenital hydrocele]. Similar: ['Hydrocele', 'Hydrocele testis', 'Male genital examination', 'Multiple congenital anomalies-hypotonia-seizures syndrome', 'Vaginal process']",
                "A hydrocele is an accumulation of serous fluid in a body cavity. A hydrocele testis, the most common form of hydrocele, is the accumulation of fluids around a testicle. It is often caused by fluid collecting within a layer wrapped around the testicle, called the tunica vaginalis, which is derived from peritoneum. Provided there is no hernia present, it goes away without treatment in the first year. Although hydroceles usually develop in males, rare instances have been described in females in the canal of Nuck",
                "(Result 1 / 4) Provided there is no hernia present, it goes away without treatment in the first year. (Result 2 / 4) (exception -  funicular hydrocele, encysted hydrocele)\n\n\n== Treatment ==\nMost hydroceles appearing in the first year of life resolve without treatment. (Result 3 / 4) Hydroceles that persist after the first year or occur later in life require treatment only in selected cases, such as patients who are symptomatic with pain or a pressure sensation, or when the scrotal skin integrity is compromised from chronic irritation; the treatment of choice is surgery and the operation is conducted via an open access technique aiming to excise the hydrocele sac. (Result 4 / 4) These alternative treatments are generally regarded as unsatisfactory because of the high incidence of recurrences and the frequent necessity for repetition of the procedure",
                "",
            ],
        },
    },
    {
        "id": "be77f10a-1e08-4ccb-9d32-23f49a68712d",
        "split": "train",
        "question": "Asymmetric tonic neck reflex (ATNR) disappears at ______?",
        "opa": "6 months",
        "opb": "5 months",
        "opc": "9 months",
        "opd": "8 months",
        "target": 0,
        "react": {
            "thought": [
                "I need to search Asymmetric tonic neck reflex",
                "The search engine does not seem to be able to find the exact term. Therefore, I need to reformulate my search query to one of the similar pages; Asymmetrical tonic neck reflex",
                "ATNR is a primitive reflex found in newborn humans that normally vanishes around 6 months of age. Therefore, the answer is A) 6 months.",
            ],
            "action": [
                "Search[Asymmetric tonic neck reflex]",
                "Search[Asymmetrical tonic neck reflex]",
                "Finish[A]",
            ],
            "observation": [
                "Could not find [Asymmetric tonic neck reflex]. Similar: ['Primitive reflexes', 'Asymmetrical tonic neck reflex', 'Reflex', 'Symmetrical tonic neck reflex', 'Tonic labyrinthine reflex']",
                'The asymmetrical tonic neck reflex (ATNR) is a primitive reflex found in newborn humans that normally vanishes around 6 months of age.\nIt is also known as the bow and arrow or "fencing reflex" because of the characteristic position of the infant\'s arms and head, which resembles that of a fencer.  When the face is turned to one side, the arm and leg on that side extend, and the arm and leg on the opposite side flex. It is more likely to be seen in premature infants than full-term babies. It is rare in newborns but can be stimulated from infants to up to 3 months old',
                "",
            ],
        },
    },
    {
        "id": "9587b40f-5950-423e-aaa2-7a05f813f832",
        "split": "train",
        "question": "The drug used in petit mal seizures and has a narrow spectrum of antiepileptic activity is?",
        "opa": "Lamotrigine",
        "opb": "Ethosuximide",
        "opc": "Phenytoin",
        "opd": "Primidone",
        "target": 1,
        "react": {
            "thought": [
                "I need to search petit mal seizures",
                "Petit mal seizure is not found. I need to search absence serizure as it is another word for petit mal seizure.",
                "The search result does not seem to be directly related to the question. Therefore, I need to look up the drug",
                "The lookup mentions that phenytoin can cause absence seizures. Therefore, the answer is not C) Phenytoin. I'll have to look up the other drugs.",
                "The lookup indicates that both ethosuximide and lamotrigin are used to treat absence seizures. Therefore, it can either be A) Lamotrigine or B) Ethosuximide. I'll have to look up Ethosuximide.",
                "The lookup mentions that absence seizures are mainly treated with ethosuximide and valproic acid. As valproic acid is not an option, the answer is B) Ethosuximide.",
            ],
            "action": [
                "Search[petit mal seizures]",
                "Search[absence seizure]",
                "Lookup[drug]",
                "Lookup[Lamotrigine]" "Lookup[Ethosuximide]",
                "Finish[B]",
            ],
            "observation": [
                "Could not find [petit mal seizures]. Similar: ['Absence seizure', 'Thought blocking', 'Kenneth Bianchi', 'Convulsion', 'Disorders of consciousness']",
                'Absence seizures are one of several kinds of generalized seizures. In the past, absence epilepsy was referred to as "pyknolepsy," a term derived from the Greek word "pyknos," signifying "extremely frequent" or "grouped". These seizures are sometimes referred to as petit mal seizures (from the French for "little illness", a term dated to the late 18th century); however, usage of this terminology is no longer recommended. Absence seizures are characterized by a brief loss and return of consciousness, generally not followed by a period of lethargy (i.e',
                "(Result 1 / 4) Some specific anticonvulsant drugs such as phenytoin, carbamazepine, and vigabatrin have been identified to raise the chances of experiencing absence seizures. (Result 2 / 4) Antiepileptic drugs such as Gabapentin, Tiagabine and Vigabatrin cause inhibition of GABA resulting in exacerbation of absence seizures. (Result 3 / 4) There were no significant differences between the three drugs with regard to discontinuation because of adverse events. (Result 4 / 4) If monotherapy fails or unacceptable adverse reactions appear, replacement of one by another of the three antiepileptic drugs is the alternative",
                "(Result 1 / 4) (2010), who studied the effects of ethosuximide, valproic acid, and lamotrigine in children with newly diagnosed childhood absence epilepsy. (Result 2 / 4) After 16 weeks of therapy, the freedom-from-failure rates for ethosuximide and valproic acid were similar and were higher than the rate for lamotrigine. (Result 3 / 4) Adding small doses of lamotrigine to sodium valproate may be the best combination in resistant cases. (Result 4 / 4) Similarly, lamotrigine treats multiple seizure types including partial seizures and generalized seizures, therefore it is also an option for patients with multiple seizure types",
                "(Result 1 / 6) == Treatment ==\nTreatment of patients with absence seizures only is mainly with ethosuximide or valproic acid, which are of equal efficacy controlling absences in around 75% of patients. (Result 2 / 6) (2010), who studied the effects of ethosuximide, valproic acid, and lamotrigine in children with newly diagnosed childhood absence epilepsy. (Result 3 / 6) After 16 weeks of therapy, the freedom-from-failure rates for ethosuximide and valproic acid were similar and were higher than the rate for lamotrigine. (Result 4 / 6) Attentional dysfunction was more common with valproic acid than with ethosuximide. (Result 5 / 6) Although ethosuximide is effective in treating only absence seizures, valproic acid is effective in treating multiple seizure types including tonic-clonic seizure and partial seizure, suggesting it is a better choice if a patient is exhibiting multiple types of seizures. (Result 6 / 6) A 2019 Cochrane review found that ethosuximide was the best mono-therapy for children and adolescents but noted that if absence seizures co-exist with tonic-clonic seizures then valproate should be preferred",
                "",
            ],
        },
    },
]


FRAGMENTS = {
    "version": "0.01",
    "instruction": {
        "qa-01": "Answer the following question through step-by-step reasoning.",
        "qa-02": "Answer the following question through careful, concise step-by-step reasoning.",
        "qa-03": 'Answer the following question through careful, concise step-by-step reasoning. Avoid making up wrong statements. If the question does not make sense or cannot be answered, write "I cannot answer the question". If you do not have a good answer, write "I do not have a good answer". If you are uncertain, write "I am uncertain about this".',
        "qa-04": 'Answer the following question through careful, concise step-by-step reasoning. Avoid making up wrong statements. Generate sub-questions that are required to answer the original question, answer them until you can answer the original question. If the question does not make sense or cannot be answered, write "I cannot answer the question". If you do not have a good answer, write "I do not have a good answer". If you are uncertain, write "I am uncertain about this".',
        "qa-05": "Answer the following question through step-by-step reasoning. Prefixed context may contain relevant knowledge.",
        "qa-06": "Using step-by-step reasoning, deduce the answer to the following question. Prefixed context may contain relevant knowledge.",
    },
    "cot_trigger": {
        "kojima-01": "Let's think step by step:",
        "kojima-02": "Answer: We should think about this step by step.",
        "kojima-03": "Answer: First,",
        "kojima-04": "Answer: Before we dive into the answer,",
        "kojima-05": "Answer: Proof followed by the answer.",
        "kojima-06": "Answer: Let's think step by step in a realistic way.",
        "kojima-07": "Answer: Let's think step by step using common sense and knowledge.",
        "kojima-08": "Answer: Let's think like a detective step by step.",
        "kojima-09": "Answer: Let's think about this logically.",
        "kojima-10": "Answer: Let's think step by step. First,",
        "kojima-11": "Answer: Let's think",
        "kojima-12": "Answer: Let's solve this problem by splitting it into steps.",
        "kojima-13": "Answer: The answer is after the proof.",
        "kojima-14": "Answer: Let's be realistic and think step by step.",
        "lievin-01": "Answer: Let's derive the differential diagnosis step by step.",
        "lievin-02": "Answer: Let's use step by step inductive reasoning, given the medical nature of the question.",
        "lievin-03": "Answer: Let's differentiate using step by step reasoning like a medical expert.",
        "lievin-04": "Answer: Let's think step by step using deductive reasoning.",
        "lievin-05": "Answer: Let's differentiate using step by step reasoning .",
        "lievin-06": "Answer: Let's think step by step to arrive at one of the options.",
        "lievin-07": "Answer: Let's break the problem into multiple steps.",
        "lievin-08": "Answer: Let's use step by step deductive reasoning, given the medical nature of the question.",
        "lievin-09": "Answer: Let's think step by step like a doctor.",
        "lievin-10": "Answer: Let's think step by step like a medical expert.",
        "lievin-11": "Answer: Let's summarize the facts step by step.",
        "lievin-12": "Answer: Let's think step by step using inductive reasoning.",
        "lievin-13": "Answer: Let's think step by step using deductive reasoning like a medical expert.",
        "lievin-14": "Answer: Let's be concise and think step by step.",
        "lievin-15": "Answer: Let's differentiate using step by step deductive reasoning like a medical expert.",
        "lievin-16": "Answer: Let's argue step by step.",
        "lievin-17": "Answer: Let's think step by step like a clinician.",
        "lievin-18": "Answer: Let's reflect on each answer option step by step.",
        "lievin-19": "Answer: Let's reason and differentiate options step by step like a medical expert.",
        "lievin-20": "Answer: Let's differentiate using step by step inductive reasoning like a medical expert.",
        "lievin-21": "Answer: Let's think step by step given every option equal consideration.",
        "lievin-22": "Answer: Let's think step by step like a scientist.",
        "lievin-23": "Answer: Let's use step by step inductive reasoning.",
        "lievin-24": "Answer: Let's work by elimination step by step.",
        "lievin-25": "Answer: Let's use step by step deductive reasoning.",
        "lievin-26": "Answer: Let's follow a Bayesian step by step approach.",
        "lievin-27": "Answer: Let's reflect on each option from the least likely to the most likely.",
        "lievin-28": "Answer: Let's use step by step Bayesian reasoning, given the medical nature of the question.",
        "motzfeldt-1": "Let's explore step by step why the answer is true.",
    },
    "answer_extraction": {
        "kojima-01": "Therefore, the answer is",
        "kojima-02": "Therefore,",
        "kojima-03": "The answer is",
        "kojima-numerals": "Therefore, the answer (arabic numerals) is",
        "kojima-yes-no": "Therefore, the answer (Yes or No) is",
        "kojima-A-C": "Therefore, among A through C, the answer is",
        "kojima-A-D": "Therefore, among A through D, the answer is",
        "kojima-A-E": "Therefore, among A through E, the answer is",
        "kojima-A-F": "Therefore, among A through F, the answer is",
    },
}
