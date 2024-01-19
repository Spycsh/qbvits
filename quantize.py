from neural_compressor import PostTrainingQuantConfig, quantization
from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.tts_multilang import MultilangTextToSpeech

# model loading
t2s = MultilangTextToSpeech()
texts_cn = [
    '绿是阳春烟景，大块文章的底色，四月的林峦，更是绿得鲜活、秀媚、诗意盎然。',
    '他仅凭腰部的力量，在泳道上下翻腾，蛹动蛇行，状如海豚，一直以一头的优势领先。',
    '企业依靠技术挖潜增效，他负责全厂产品质量与技术培训，成了厂里的大忙人。',
    '菜做好了，一碗清蒸武昌鱼，一碗蕃茄炒鸡蛋，一碗榨菜干子炒肉丝。',
    '她看看夜己很深，白天的炎热已给夜凉吹散，吩咐大家各自安息，明天继续玩乐。',
    '有一家个体制品厂，本该用完整的型材生产门窗，却用半截材打结凑合。',
    '久居闹市，常常忘了山之外、水之外、身之外，还有沃野平畴，还有光风丽日。',
    '旷野的风要往这儿刮，那儿刮，你能命令风四面八方全刮一点吗。',
    '羊卓雍湖抽水蓄能电厂是目前世界上海拔最高的抽水蓄能电厂，也是西藏最大的能源基地。',
    '放眼望去，永定河两旁人声鼎沸，彩旗飘扬，推土机、挖土机、运土车正紧张地忙碌着。',
    '炮眼打好了，炸药怎么装，岳正才咬了咬牙，倏地脱去衣服，光膀子冲进了水窜洞。',
]
texts_en = [
    "Many of Shakespeare's plays were published in editions of varying quality and accuracy during his lifetime.",
    "Build your chatbot within minutes on your favorite device; offer SOTA compression techniques for LLMs",
    "Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project."
]

# do quantize
def calib_func_bert_cn(model):
    for text in texts_cn[:1]:
        t2s.text2speech(text, "tmp.wav")

conf = PostTrainingQuantConfig(
    # approach="dynamic",
    # op_type_dict=op_type_dict
    )
model = quantization.fit(t2s.bert_vits_model.cn_bert_model, conf, calib_func=calib_func_bert_cn)
model.save(f"vits_bert_cn")


def calib_func_vits(model):
    for text in texts_cn + texts_en:
        t2s.text2speech(text, "tmp.wav")

op_type_dict = {
        # q/dq 357+5+7=369
        # "Conv1d": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},   # 357
        "ConvTranspose1d": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},  # 5
        "Linear": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},   # 7
        # "Embedding": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},    # 4
        }

conf = PostTrainingQuantConfig(
    # approach="dynamic",
    op_type_dict=op_type_dict
    )
model = quantization.fit(t2s.bert_vits_model.vits, conf, calib_func=calib_func_vits)

model.save(f"vits_int8")
