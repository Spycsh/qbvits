
This is some experiment code to support Bert-VITS2 int8 static quantization by using Intel Neural Compressor and Intel Extension For Transformers.
Bert-VITS2 is a great repo on multilanguage TTS, but it has not been optimized in inference speed or memory space. This repo target at the inference speed and memory space optimization. This repo is not an official project from Intel and should be experimental. Future features may be integrated into Intel Extension For Transformers NeuralChat.

```
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers
git checkout spycsh/bvits-tts
cd ..
export PYTHONPATH=$(pwd)/intel-extension-for-transformers

python quantize.py
python infer.py
python reduce_nosie.py
```