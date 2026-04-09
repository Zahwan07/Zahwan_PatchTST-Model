# Full setup: Git clone → Ollama → Run Qwen (models on D:)

Use this guide to clone the Qwen repo (optional), install Ollama, store models on D:, and run Qwen.

---

## Part 1: Clone Qwen3.5 repo to D: (optional reference)

The repo has docs and links; model weights are downloaded separately (e.g. via Ollama).

1. Open **PowerShell** or **Command Prompt**.

2. Go to D: and create a folder:
   ```powershell
   D:
   mkdir D:\Qwen3.5
   cd D:\Qwen3.5
   ```

3. Clone the repo:
   ```powershell
   git clone https://github.com/QwenLM/Qwen3.5.git .
   ```
   (The `.` puts files in the current folder `D:\Qwen3.5`.)

4. If you don’t have Git: install from https://git-scm.com/download/win then run the commands again.

You now have the Qwen3.5 repo on D: for reference. You don’t need it to run Ollama; Ollama downloads its own model files.

---

## Part 2: Install Ollama

1. Go to **https://ollama.com/download**.

2. Download **Ollama for Windows** and run the installer.

3. Complete the installer (default options are fine).

4. Check that Ollama is installed:
   - Open a **new** PowerShell or Command Prompt.
   - Run:
     ```powershell
     ollama --version
     ```
   - You should see a version number. If `ollama` is not recognized, close and reopen the terminal or add Ollama to your PATH.

---

## Part 3: Store Ollama models on D:

So models don’t use your C: drive:

1. Create a folder on D: for Ollama models:
   ```powershell
   mkdir D:\OllamaModels
   ```

2. Set the environment variable **permanently** (recommended):
   - Press **Win + R**, type `sysdm.cpl`, Enter.
   - Go to **Advanced** tab → **Environment Variables**.
   - Under **User variables** (or **System** if you want it for all users), click **New**.
   - Variable name: `Ollama_Model`
   - Variable value: `D:\OllamaModels`
   - OK → OK → OK.

3. **Restart your terminal** (and restart Ollama if it’s running in the background).

4. Check (in a new terminal):
   ```powershell
   echo $env:Ollama_Model
   ```
   You should see `D:\OllamaModels`. (In Command Prompt use `echo %Ollama_Model%`.)

**One-time only (current session):**  
If you didn’t set it permanently, you can set it for this session:
```powershell
$env:Ollama_Model = "D:\OllamaModels"
```
Then start Ollama in this same window.

---

## Part 4: Start Ollama and pull a Qwen model

1. Start Ollama (if it isn’t already):
   - Either run **Ollama** from the Start menu (it runs in the background), or
   - In terminal: `ollama serve` (leave this window open).

2. In a **new** terminal, pull a Qwen model.  
   Ollama often has **Qwen 2.5**; **Qwen 3.5** may appear later. Use what’s available:

   **Small (less disk, less RAM):**
   ```powershell
   ollama pull qwen2.5:3b
   ```
   (About 2 GB; good for 20GB or tight space.)

   **Medium (better quality):**
   ```powershell
   ollama pull qwen2.5:7b
   ```
   (About 4–5 GB.)

   **Larger:**
   ```powershell
   ollama pull qwen2.5:14b
   ```
   (About 9 GB; only if D: has enough space.)

   Wait until the download finishes. Files go to `D:\OllamaModels` if `Ollama_Model` is set correctly.

3. Check that the model is there:
   ```powershell
   ollama list
   ```
   You should see e.g. `qwen2.5:3b` or `qwen2.5:7b`.

---

## Part 5: Run Qwen in Ollama

1. **Chat in the terminal:**
   ```powershell
   ollama run qwen2.5:3b
   ```
   (Replace with `qwen2.5:7b` or the tag you pulled.)  
   Type your message and press Enter. Type `/bye` or Ctrl+D to exit.

2. **Use the Ollama app (if installed):**  
   Open Ollama from the Start menu, pick the same model (e.g. Qwen 2.5 3B), and chat in the UI.

3. **Use from your Python app (e.g. ModelTA):**  
   Ollama exposes a local API. In your code, use:
   - Base URL: `http://localhost:11434/v1`
   - No API key for local use.

   Example with `openai`-compatible client (if your `llm_recommendations.py` points to Ollama):
   - Set `LLM_API_BASE=http://localhost:11434/v1` and use model name e.g. `qwen2.5:3b` (no key needed).

---

## Part 6: Use Ollama with ModelTA (llm_recommendations.py)

`llm_recommendations.py` is already wired for Ollama. Do this when you want recommendations to use your local Qwen:

1. **Ollama must be running** (app from Start menu or already started).

2. **Set environment variables** (PowerShell, same session where you run Python):
   ```powershell
   $env:LLM_API_BASE = "http://localhost:11434/v1"
   $env:LLM_MODEL = "qwen2.5:3b"
   ```
   Use the model tag you pulled (e.g. `qwen2.5:7b` if you use the 7B model).

3. **Run a script that calls the LLM**, e.g. current conditions + recommendations:
   ```powershell
   cd C:\Users\User\ModelTA
   python Main_model/combine_forecast_realtime.py
   ```
   That script reads realtime data, builds a short summary, and sends it to the LLM (Ollama). You’ll see the summary and the recommendation text.

4. **Optional – make it permanent:** Add the two variables in System Properties → Environment Variables (same as `OLLAMA_MODELS`) so you don’t have to set them every time:
   - `LLM_API_BASE` = `http://localhost:11434/v1`
   - `LLM_MODEL` = `qwen2.5:3b`

---

## Quick reference

| Step | What to do |
|------|------------|
| 1 | Clone repo (optional): `git clone https://github.com/QwenLM/Qwen3.5.git D:\Qwen3.5` |
| 2 | Install Ollama from https://ollama.com/download |
| 3 | Set `Ollama_Model=D:\OllamaModels` (Environment Variables) and restart terminal |
| 4 | `ollama pull qwen2.5:14b` (or 7b / 14b) |
| 5 | `ollama run qwen2.5:3b` to chat; or use `http://localhost:11434/v1` in your app |

---

## Troubleshooting

- **`ollama` not found:** Reopen terminal after install; or add Ollama’s install folder to PATH.
- **Models still on C::** Check `echo $env:Ollama_Model` and that you restarted the terminal (and Ollama) after setting it.
- **Out of memory when running:** Use a smaller model (e.g. `qwen2.5:3b`).
- **Qwen 3.5 in Ollama:** Check `ollama pull qwen3.5:4b` (or search https://ollama.com/library) when available; steps are the same.
