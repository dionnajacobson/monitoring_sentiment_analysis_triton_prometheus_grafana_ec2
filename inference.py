import numpy as np
import tritonclient.http as tritonhttpclient
from scipy.special import softmax
from transformers import AutoModel, AutoTokenizer
VERBOSE=False
R_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english',padding_side='left')
input_name = ['input__0']
output_name = 'output__0'
#url='127.0.0.1:8000'
def run_inference(premise, model_name='distilbert', url='54.84.220.218:8000', model_version='1'):
    triton_client = tritonhttpclient.InferenceServerClient(
        url=url, verbose=VERBOSE)
    model_metadata = triton_client.get_model_metadata(
        model_name=model_name, model_version=model_version)
    model_config = triton_client.get_model_config(
        model_name=model_name, model_version=model_version)
    # I have restricted the input sequence length to 256
    input_ids = R_tokenizer.encode(premise, max_length=256, truncation=True, padding='max_length')
    input_ids = np.array(input_ids, dtype=np.int32)
    input_ids = input_ids.reshape(1, 256)
    input0 = tritonhttpclient.InferInput(input_name[0], (1,  256), 'INT32')
    input0.set_data_from_numpy(input_ids, binary_data=False)
    output = tritonhttpclient.InferRequestedOutput(output_name,  binary_data=False)
    response = triton_client.infer(model_name, model_version=model_version, inputs=[input0], outputs=[output])
    logits = response.as_numpy('output__0')
    logits = np.asarray(logits, dtype=np.float32)
    probs = softmax(logits)
    true_prob = probs[:,1].item() * 100
    print(f'Probability that the label is positive: {true_prob:0.2f}%')
# topic classification premises
if __name__ == '__main__':
    run_inference('This is the best')