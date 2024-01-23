from neural_compressor.utils.pytorch import load
from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.tts_multilang import MultilangTextToSpeech
from transformers import AutoConfig

# model loading
t2s = MultilangTextToSpeech()
from transformers import BertForMaskedLM, DebertaV2Model
bert_cn = BertForMaskedLM(config=AutoConfig.from_pretrained("hfl/chinese-roberta-wwm-ext-large"))
t2s.bert_vits_model.cn_bert_model = load("vits_bert_cn/best_model.pt", bert_cn)

# bert_en = DebertaV2Model(config=AutoConfig.from_pretrained("microsoft/deberta-v3-large"))
# t2s.bert_vits_model.en_bert_model = load("vits_bert_en", bert_en)

# bert_jp = DebertaV2Model(config=AutoConfig.from_pretrained("ku-nlp/deberta-v2-large-japanese-char-wwm"))
# t2s.bert_vits_model.jp_bert_model = load("vits_bert_jp", bert_jp)

t2s.bert_vits_model.vits = load("vits_int8/best_model.pt", t2s.bert_vits_model.vits)

path = t2s.text2speech("欢迎来到英特尔，welcome to Intel。こんにちは！", "aaa.wav")