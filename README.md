# LiveKit Agent · gpt-4o-transcribe (Cascade)

本地 console 模式跑通的 LiveKit voice agent：

- **STT**: Azure OpenAI `gpt-4o-transcribe`
- **LLM**: Azure OpenAI Chat（默认 `gpt-4o-mini`）
- **TTS**: Azure Speech（默认 `zh-CN-XiaoxiaoNeural`）
- **VAD**: Silero

## 1. 准备环境

```powershell
cd livekit-4o-trans
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. 配置 .env

```powershell
copy .env.example .env
# 编辑 .env，至少填写 AZURE_OPENAI_ENDPOINT、AZURE_SPEECH_REGION、AZURE_SPEECH_KEY
```

认证方式二选一：
- **Key**: 设置 `AZURE_OPENAI_API_KEY`
- **Entra ID**：留空 key，提前 `az login`，使用 `DefaultAzureCredential`

## 3. 下载 VAD 模型（首次）

```powershell
python agent.py download-files
```

## 4. 本地 console 运行

```powershell
python agent.py console
```

直接通过麦克风对话，扬声器播放回复，无需启动 LiveKit 服务器。

## 5. 连到 LiveKit 房间（可选）

设置 `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` 后：

```powershell
python agent.py dev
```
