def get_prompt(prompt_type="basic"):
    if prompt_type == "basic":
        return system_template, basic_template
    elif prompt_type == "plan":
        return system_template, plan_template


basic_template = "You will be provided with an abstract of a scientific document and other references papers in triple quotes. Your task is to write the related work section of the document using only the provided abstracts and other references papers. Please write the related work section creating a cohesive storyline by doing a critical analysis of prior work comparing the strengths and weaknesses while also motivating the proposed approach. You should cite the other related documents as [#] whenever you are referring it in the related work. Do not write it as Reference #. Do not cite abstract. Do not include any extra notes or newline characters at the end. Do not copy the abstracts of reference papers directly but compare and contrast to the main work concisely. Do not provide the output in bullet points. Do not provide references at the end. Please cite all the provided reference papers. Provide the output in maximum {max_tokens} words."

plan_template = "You will be provided with an abstract of a scientific document and other references papers in triple quotes. Your task is to write the related work section of the document using only the provided abstracts and other references papers. Please write the related work section creating a cohesive storyline by doing a critical analysis of prior work comparing the strengths and weaknesses while also motivating the proposed approach. You are also provided a sentence plan mentioning the total number of lines and the citations to refer in different lines. You should cite all the other related documents as [#] whenever you are referring it in the related work. Do not cite abstract. Do not include any extra notes or newline characters at the end. Do not copy the abstracts of reference papers directly but compare and contrast to the main work concisely. Do not provide the output in bullet points. Do not provide references at the end. Please cite all the provided reference papers. Please follow the plan when generating sentences, especially the number of lines to generate."

plan = (
    "Generate the output in 200 words using 5 sentences. Cite [1] on line 2. Cite [2], [3] on line 3. Cite [4] on line 5.",
)

system_template = "You are a helpful research assistant who is helping with literature review of a research idea."
