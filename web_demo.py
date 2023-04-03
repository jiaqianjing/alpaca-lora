from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import torch
from peft import PeftModel
import transformers
import gradio as gr

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"


BASE_MODEL = "./hf_weight/llama-7b/"
LORA_WEIGHTS = "lora-alpaca-202304021101"

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL, load_in_8bit=True, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(
    model, LORA_WEIGHTS, torch_dtype=torch.float16)


# prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:
# {instruction}

# ### Response:{response}"""

prompt_template = """
### Human:
{instruction}

### AI:
{response}
"""

model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)

device = "cuda"


def evaluate(instruction, history=None, temperature=0.1, top_p=0.75, top_k=1, num_beams=1, max_tokens=2048, **kwargs):
    if history is None:
        history = []
    if not history:
        prompt = prompt_template.format(instruction=instruction, response="")
        print("first session: ", prompt)
    else:
        prompt = ""
        for old_instruction, response in history:
            print(f"old_instruction: {old_instruction}, response: {response}")
            prompt += prompt_template.format(instruction=old_instruction, response=response) + "\n"
        prompt += prompt_template.format(instruction=instruction, response="")
    print("prompt: " + prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    response = output.split("### AI:")[-1].strip()
    new_history = history + [(instruction, response)]
    yield response, new_history


MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2


def predict(input, max_length, top_p, temperature, history=None):
    if history is None:
        print("init history...")
        history = []
    print("input: ", input)
    for response, history in evaluate(input, history, temperature=temperature, top_p=top_p, max_tokens=max_length):
        updates = []
        print(f"reponse: {response}, history: {history}")
        for query, response in history:
            updates.append(gr.update(visible=True, value="User: " + query))
            updates.append(gr.update(visible=True, value="Alpatent: " + response))
        if len(updates) < MAX_BOXES:
            updates = updates + [gr.Textbox.update(visible=False)] * (MAX_BOXES - len(updates))
        yield [history] + updates


with gr.Blocks() as demo:
    state = gr.State([])
    print(f"state: {state}, type: {type(state)}")
    text_boxes = []
    for i in range(MAX_BOXES):
        if i % 2 == 0:
            text_boxes.append(gr.Markdown(visible=False, label="User: "))
        else:
            text_boxes.append(gr.Markdown(visible=False, label="Alpatent: "))

    with gr.Row():
        with gr.Column(scale=4):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", lines=11).style(container=False)
        with gr.Column(scale=1):
            max_length = gr.Slider(0, 2048, value=256, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
            button = gr.Button("Generate")
    button.click(predict, [txt, max_length, top_p, temperature, state], [state] + text_boxes)
demo.queue(concurrency_count=1).launch(share=True, inbrowser=True, server_name='0.0.0.0', server_port=6006)
