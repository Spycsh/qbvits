from neural_compressor.utils.pytorch import load
from neural_compressor import PostTrainingQuantConfig, quantization
from huggingface_hub import hf_hub_download
from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.tts_multilang import MultilangTextToSpeech
from transformers import AutoConfig

# model loading
t2s = MultilangTextToSpeech()
from transformers import BertForMaskedLM
bert_cn = BertForMaskedLM(config=AutoConfig.from_pretrained("hfl/chinese-roberta-wwm-ext-large"))
t2s.bert_vits_model.cn_bert_model = load("vits_bert_cn", bert_cn)
t2s.bert_vits_model.vits = load("vits_int8", t2s.bert_vits_model.vits)

path = t2s.text2speech("欢迎来到英特尔，welcome to Intel。こんにちは！", "aaa.wav")