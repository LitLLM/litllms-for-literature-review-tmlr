import gradio as gr
import json


def button_callback():
    # You can perform some action here when the button is clicked
    pass


class GradioApp:
    def __init__(self):
        self.input_file_path = "input.jsonl"
        self.output_file_path = "sample_output.jsonl"
        # self.idx = 0
        with open(self.input_file_path) as f:
            self.input_data_lines = f.readlines()
        self.input_data = json.loads(self.input_data_lines[0]) # Initial data
        self.q1_feedback = None

    @staticmethod
    def get_box_html(text):
        html_txt = f"<p style='border-width:3px; border-style:solid; border-color:#4b5563; padding: 1em; " \
                   f"font-size: 18px;'> {text} </p>"
        return html_txt

    @staticmethod
    def get_text_html(text):
        html_txt = f"<p style='font-size: 18px;'> {text} </p>"
        return html_txt

    @staticmethod
    def append_to_jsonl(data, file_path, do_json_dump: bool = True, mode="a"):
        with open(file_path, 'a') as jsonl_file:
            if do_json_dump:
                json_string = json.dumps(data)
            else:
                json_string = data
                # TypeError: can't concat str to bytes. TODO find a better way
            jsonl_file.write(json_string + '\n')
    
    def skip_question(self, button, idx):
        idx += 1
        self.q1_feedback = None
        if idx < len(self.input_data_lines):
            nxt_input_data = json.loads(self.input_data_lines[idx])
            return f"Paper {idx+1} \n\n {nxt_input_data['abstract']}", nxt_input_data['citation_text'], nxt_input_data["response_a"], nxt_input_data["response_b"], idx
        return "Task Completed. Thanks!", None, None, idx
    
    def store_temp_values(self, button):
        self.q1_feedback = button            
        return f"You have selected: {self.q1_feedback}"

    def submit_question(self, button, idx, rank_1, rank_2, hallucinate_1, hallucinate_2):
        if idx <= len(self.input_data_lines):
            output_data = json.loads(self.input_data_lines[idx])
            # output_data["relevance_a"] = relevance_1
            # output_data["relevance_b"] = relevance_2
            # output_data["relevance_c"] = relevance_3
            # output_data["coherence_a"] = coherence_1
            # output_data["coherence_b"] = coherence_2
            # output_data["coherence_c"] = coherence_3
            output_data["rank_a"] = rank_1
            output_data["rank_b"] = rank_2
            output_data["hallucinate_a"] = hallucinate_1
            output_data["hallucinate_b"] = hallucinate_2
            # output_data["hallucinate_c"] = hallucinate_3
            output_data["index"] = idx
            self.append_to_jsonl(output_data, self.output_file_path)
            idx += 1
            if idx < len(self.input_data_lines):
                nxt_input_data = json.loads(self.input_data_lines[idx])
        #         return f"# {idx+1} \n\n {nxt_input_data['abstract']}", nxt_input_data['citation_text'], nxt_input_data["response_a"], nxt_input_data["response_b"], nxt_input_data["response_c"], idx, None, None, None, None, None, None, None, None, None
        # return "Task Completed. Thanks!", None, None, None, None, idx, None, None, None, None, None, None, None, None, None
                return f"Paper {idx+1} \n\n {nxt_input_data['abstract']}", nxt_input_data['citation_text'], nxt_input_data["response_a"], nxt_input_data["response_b"], idx, None, None, None, None
        return "Task Completed. Thanks!", None, None, None, None, idx, None, None, None, None
            

    def store_feedback(self, button, idx):
        if idx <= len(self.input_data_lines):
            output_data = json.loads(self.input_data_lines[idx])
            output_data["q1_feedback"] = self.q1_feedback
            output_data["q2_feedback"] = button
            output_data["index"] = idx
            self.append_to_jsonl(output_data, self.output_file_path)
            idx += 1
            self.q1_feedback = None
            if idx < len(self.input_data_lines):
                nxt_input_data = json.loads(self.input_data_lines[idx])
                return f"Paper {idx+1} \n\n {nxt_input_data['data_input']}", nxt_input_data["response_a"], nxt_input_data["response_b"], None, idx
        return "Task Completed. Thanks!", None, None, None, idx

    def main_fn(self):
        # Define the custom CSS style for the button when it's selected
        selected_style = """
        background-color: #e74c3c;
        color: #ffffff;
        """
        with gr.Blocks(theme=gr.themes.Base()) as demo:
            idx = gr.State(0)
            with gr.Row():
                gr.Markdown(f"<h2><center> Evaluate better Literature Review! </center></h2>")
            with gr.Row():
                gr.Markdown(f"Please provide ranks for model generated related work (1 best - you can also select both responses as 1s or 2s). Also mark hallucinations.")
            with gr.Accordion("Instructions (click to expand)", open=False):
                gr.Markdown(
                    """ 
                    * Goal: We would like to request your feedback on the performance of AI assistants in **generating a paragraph** of related work section of a scientific paper.
                    * Instructions: You will be provided with: (On left) a) The abstract of a paper; b) The abstracts of each reference paper it has to cite and (On right) c) candidate-related work sections for that paper which needs to be evaluated  
                    * Q1: Please provide the ranks for all the candidate which provides the best related works section. 
                        - 1 denotes the winner, while 2 means the loser. You can select 1 for both if you think both are equally good and 2 for both if they are equally bad.
                        - Eg. Model X is not fluent or misses to cover the citation @cite_y, give it a rank of 2. 
                        - If you cannot make a judgment you can select "skip" to move to the next example.
                    - **Please do not consider the length of the responses.** Please act as an impartial judge.
                    * Q2: Please also evaluate hallucinations - ie. - Which output contain information not mentioned in the abstracts?
                    Eg. The output cites a made up reference @cite_x not provided in the list of references. Or cites XYZ et al. without any information about this paper in references or abstract.

                    Thanks for volunteering to help with our work!
                    """
                )

            with gr.Row():
                with gr.Column(min_width=0, scale=1):
                    # input_txt = gr.Textbox(value=f"# 1: \n\n {self.input_data['data_input']}", label=f"Abstract + Reference", max_lines=25)
                    abstract_txt = gr.Textbox(value=f"Paper 1: \n {self.input_data['abstract']}", label=f"Abstract", max_lines=11)
                    cite_txt = gr.Textbox(value=f"{self.input_data['citation_text']}", label=f"Reference", max_lines=15)
                with gr.Column(min_width=0, scale=1):
                    with gr.Row():
                        result_1 = gr.Textbox(value=self.input_data["response_a"], label="Model A", scale=8)                        
                        with gr.Column(min_width=0, scale=3):
                            # rank_1 = gr.Dropdown([1, 2], label="Rank", scale=1, min_width=0)
                            rank_1 = gr.Radio(choices=[1, 2], label="Rank", scale=1, min_width=0)
                            # relevance_1 = gr.Slider(1, 5, step=1, label="Relevance", info="1: low", scale=1, min_width=0)
                            # coherence_1 = gr.Slider(1, 5, step=1, label="Coherence", scale=1, min_width=0)
                            hallucinate_1 = gr.Radio(choices=["True", "False"], label="Hallucination", scale=2, min_width=0)
                    with gr.Row():
                        result_2 = gr.Textbox(value=self.input_data["response_b"], label="Model B", scale=8)                        
                        with gr.Column(min_width=0, scale=3):
                            # rank_2 = gr.Dropdown([1, 2], label="Rank", scale=1, min_width=0)
                            rank_2 = gr.Radio(choices=[1, 2], label="Rank", scale=1, min_width=0)
                            # relevance_2 = gr.Slider(1, 5, step=1, label="Relevance", info="1: low", scale=1, min_width=0)
                            # coherence_2 = gr.Slider(1, 5, step=1, label="Coherence", scale=1, min_width=0)
                            hallucinate_2 = gr.Radio(choices=["True", "False"], label="Hallucination", scale=2, min_width=0)

                    # with gr.Row():
                    #     result_3 = gr.Textbox(value=self.input_data["response_c"], label="Model C", scale=8)
                    #     with gr.Column(min_width=0, scale=3):
                    #         relevance_3 = gr.Slider(1, 5, step=1, label="Relevance", info="1: low", scale=1, min_width=0)
                    #         coherence_3 = gr.Slider(1, 5, step=1, label="Coherence", scale=1, min_width=0)
                    #         hallucinate_3 = gr.Radio(choices=["True", "False"], label="Hallucination", scale=2, min_width=0)
            
            with gr.Row():
                gr.Column(scale=1)
                with gr.Column(scale=1):
                    with gr.Row():
                        select_skip_btn = gr.Button(value="Skip", scale=0)
                        select_submit_btn = gr.Button(value="Submit", scale=0)
            # select_skip_btn.click(self.skip_question, inputs=[select_skip_btn, idx], outputs=[abstract_txt, cite_txt, result_1, result_2, result_3, idx])
            # select_submit_btn.click(self.submit_question, inputs=[select_submit_btn, idx, relevance_1, relevance_2, relevance_3, coherence_1, coherence_2, coherence_3, hallucinate_1, hallucinate_2, hallucinate_3], 
            #                         outputs=[abstract_txt, cite_txt, result_1, result_2, result_3, idx, relevance_1, relevance_2, relevance_3, coherence_1, coherence_2, coherence_3, hallucinate_1, hallucinate_2, hallucinate_3])
            select_skip_btn.click(self.skip_question, inputs=[select_skip_btn, idx], outputs=[abstract_txt, cite_txt, result_1, result_2, idx])
            select_submit_btn.click(self.submit_question, inputs=[select_submit_btn, idx, rank_1, rank_2, hallucinate_1, hallucinate_2], 
                                    outputs=[abstract_txt, cite_txt, result_1, result_2, idx, rank_1, rank_2, hallucinate_1, hallucinate_2])


        # demo.queue(concurrency_count=16)
        demo.launch(server_name="0.0.0.0", share=False) # max_threads=4


gradio_app = GradioApp()
gradio_app.main_fn()