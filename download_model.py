# pip install "modelscope==1.7.2rc0" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
from modelscope import snapshot_download

model_dir = snapshot_download("modelscope/Llama-2-7b-chat-ms", revision='v1.0.1', 
                              ignore_file_pattern = [r'\w+\.safetensors'])
print(model_dir)