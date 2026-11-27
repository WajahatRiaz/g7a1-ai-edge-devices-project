# Long Context Tool-Calling Agent

AI624 · AI on Edge Devices · Fall 2026  
Track A — Project A1 · 8 GB Raspberry Pi 5

Wajahat Riaz  
26 Nov 2026

Live charts: [wajahatriaz.github.io/g7a1-ai-edge-devices-project](https://wajahatriaz.github.io/g7a1-ai-edge-devices-project/)  
How to run it: [README.md](../README.md)  
Course handout: [Project_Description_AI624_2026.pdf](Project_Description_AI624_2026.pdf)

This is the talk, written for someone who has not seen the project. It has three parts. **Hardware setup** is the Raspberry Pi itself. **Software setup** is every program we install and the commands that start a run. **The project** is the experiment: a model that calls tools, why long chats are hard, and what we measure. Numbers from 26 Nov 2026 are from a run that had not finished. They are observations, not the final score.

---

## Part 1. Hardware setup

### The board

The computer under test is a **Raspberry Pi 5 Model B Rev 1.0**. A Raspberry Pi is a small single-board computer, about the size of a deck of cards. "8 GB" is the working memory (RAM). A laptop often has 16 GB or more. This board has 4 processor cores (ARM Cortex-A76). Its name on the network is `g7a1-machine`. It runs Raspberry Pi OS, kernel `6.18.60+rpt-rpi-2712`.

There is no separate graphics card and no external power meter. The model runs on the Pi's own processor. Watts, degrees, and the "am I going too slow to stay cool?" flag are read from chips already on the board, with a Raspberry Pi command called `vcgencmd`.

| What we read | Command | Plain meaning |
|---|---|---|
| Power | `vcgencmd pmic_read_adc` | Voltage times current on each supply rail that reports both. About once a second during a run |
| Temperature | `vcgencmd measure_temp` | Degrees Celsius of the main chip |
| Throttle | `vcgencmd get_throttled` | Whether the chip has cut its own speed |

**Throttling** is the board protecting itself. When the chip gets too hot it lowers the clock. It can sit near 95°C, draw only a few watts, and get less work done per second. A speed measured in that state is a hot-board speed.

Rails named `EXT5V` and `BATT` report a voltage and no current. They are left out of the watt sum. A watt figure needs both voltage and current.

A second computer, called the **host** in the course notes, is allowed for one job only: scoring whether the model got the exam questions right, using the same model file. Speed, memory, heat, and power stay on this Pi.

### What has to fit in 8 GB

The model file is about 3.8 GB and stays in memory the whole time. On top of that, the model keeps notes for every piece of the chat it has read. Those notes are the **KV cache**. They grow with the chat and live in the same 8 GB.

A **token** is one piece of text. A common word may be one token. In this model each digit is its own token, so a long number is many tokens. "10k tokens" means about ten thousand pieces.

| What is in memory | Size |
|---|---|
| The model file (BF16, explained in the software section) | about 3.8 GB |
| Notes kept per token | about 112 KB each |
| A 16,384-token chat (course baseline) | about 1.75 GB of notes |
| An 8,192-token chat | about 0.9 GB of notes |
| A 16,384-token chat | about 1.75 GB of notes |
| A 32,768-token chat | about 3.5 GB of notes |

The 112 KB figure comes from the shape of this model: 28 layers, 8 note-heads, 128 numbers wide, two bytes each, and both a key and a value. 2 × 28 × 8 × 128 × 2 bytes = 114,688 bytes, which is 112 KiB.

The course PDF baseline is a **16,384-token** cache (fits beside the 3.8 GB file). **32,768** does not fit on 8 GB. `./scripts/start_server.sh` serves **`-c 16384`** to match that baseline.

```mermaid
flowchart LR
  subgraph board [8 GB on the Raspberry Pi 5]
    W[Model file 3.8 GB]
    K[Chat notes 112 KB per token]
    O[Linux and the server]
  end
  W --- K --- O
```

---

## Part 2. Software setup

The software is everything installed on that board so the model can answer, call a fake app, and write down watts and degrees.

### What gets installed

From a fresh Raspberry Pi OS, the build needs a compiler, CMake, git, and Python:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config \
  libcurl4-openssl-dev python3-venv python3-dev
```

Then the project itself:

```bash
git clone --recurse-submodules --shallow-submodules \
  git@github.com:WajahatRiaz/g7a1-ai-edge-devices-project.git
cd g7a1-ai-edge-devices-project
./setup.sh
```

`./setup.sh` does four jobs:

1. Downloads two third-party code trees if they are missing. **llama.cpp** is the program that runs the model on a normal processor, without a large NVIDIA GPU. The copy here is commit `d1eb0de`. **Gorilla** is the Berkeley tool-use exam; we use only its fake apps.
2. Creates `.venv`, a private Python library folder, and installs two small libraries: `requests` (to talk to the model server) and `PyYAML` (to read the settings file).
3. Builds four programs into `build/llama.cpp/bin/`: `llama-server` (answers chat requests), `llama-cli` (one question in the terminal), `llama-quantize` (makes a smaller copy of the model later), and `llama-imatrix` (records which weights matter, also later).
4. Cuts the made-up TradingBot padding in `long_context.py` if that cut is not already there. The same file is used by our runner and by the exam's checker.

A second Python folder, `.venv-convert`, is created only when the model is downloaded. It holds the heavier libraries (`torch`, `numpy`) that turn the published model into our file. The harness does not need them.

The model weights are not in git. They are downloaded from [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) and converted locally:

```bash
./scripts/download_and_convert_baseline.sh
```

**Qwen3-1.7B** is the model, from Alibaba's Qwen team. "1.7B" means about 1.7 billion learned numbers. Those numbers are the **weights**. The converted file is `models/qwen3-1.7b-bf16.gguf`, about 3.8 GB.

**GGUF** is the file format llama.cpp reads. **BF16** is the 16-bit layout of the starting file. **FP16** is another 16-bit layout this Pi can multiply faster. FP16 is the first later experiment. It does not replace the BF16 file. A smaller GGUF such as Q8 or Q4 is **quantization**: fewer bits per number, smaller and slightly less exact. Those copies are not built yet. The official GGUF published for this model is a Q8 file. That file is not our starting point.

### How a run is started

Two terminals. The first starts the server and is left alone:

```bash
./scripts/start_server.sh
```

That is this command:

```bash
llama-server \
  -m models/qwen3-1.7b-bf16.gguf \
  -c 16384 --jinja --webui \
  -t 2 -tb 4 -np 1 \
  --host 127.0.0.1 --port 8080
```

| Flag | Meaning |
|---|---|
| `-m` | Which model file |
| `-c 16384` | The longest chat the server holds — **16k**, the course A1 baseline |
| `--jinja` | Use Qwen's official chat template, including the scratchpad called thinking |
| `--webui` | Also serve a browser chat at `http://127.0.0.1:8080/` |
| `-t 2` | Two cores write new tokens |
| `-tb 4` | All four cores read the incoming chat |
| `-np 1` | Prepare memory for one chat. The default of four would reserve four copies of the chat notes and slow this model down |
| `--host 127.0.0.1` | This machine only. The model is not on the public internet |

`--flash-attn auto` is left unset, so the server uses its default. Flash attention is a faster way to do the step where each token looks at the others. The default turns it on when the model supports it.

The second terminal, once the server says it is listening, runs the measurement:

```bash
./scripts/run_baseline.sh
```

### The programs, one by one

| Program | What a person would use it for |
|---|---|
| `setup.sh` | Install libraries, build llama.cpp, and apply the TradingBot padding cut |
| `scripts/download_and_convert_baseline.sh` | Download Qwen3 and write the BF16 file |
| `scripts/ensure_convert_venv.sh` | Create the heavier Python folder used only for that conversion |
| `scripts/prepare_tasks.py` | Copy the exam questions into train, test, and extra files |
| `scripts/start_server.sh` | Start the model and leave it running |
| `scripts/chat_once.sh` | Ask one question in the terminal. This is not a multi-step tool chat |
| `scripts/run_harness.py` | Play a saved task: send the chat, run tools, write timings. `run_harness.sh` is the same thing |
| `scripts/run_baseline.sh` | The Week 1 measurement: one short task, then a reading-speed sweep, with power logging |
| `scripts/characterize_baseline.py` | The sweep alone: dummy prompts of 512, 2,048, and 4,096 tokens |
| `scripts/sample_device.py` | Read watts, degrees, throttle, and server memory about once a second. `sample_device.sh` wraps another command with that log |
| `scripts/export_dashboard.py` | Turn the logs into `docs/data/dashboard.json`. `export_dashboard.sh` is the same thing |
| `scripts/run_host_accuracy.sh` | Score all 12 exam tasks. Meant for a normal computer, not for timing on the Pi |
| `src/a1_agent/client.py` | The code that sends a chat to llama-server |
| `src/a1_agent/schemas.py` | Loads the 30 tool descriptions |
| `src/a1_agent/backends.py` | Runs the fake trading account and inbox |
| `src/a1_agent/parse.py` | Checks that a tool request names a real tool and fills in the required arguments |
| `src/a1_agent/harness.py` | The loop: chat, tool, chat again, and the per-turn timings |
| `configs/baseline.yaml` | The starting settings, so a later experiment changes a file instead of a habit |
| `docs/index.html` | The public page of past runs, from GitHub Pages |

```mermaid
flowchart LR
  subgraph files [Files in the project]
    Y[baseline.yaml settings]
    T[Saved exam questions]
    H2[Our harness]
  end
  subgraph built [Programs built on the Pi]
    L[llama-server runs the model]
    Q[llama-quantize makes smaller copies later]
    I[llama-imatrix measures which weights matter later]
  end
  Y --> L
  T --> H2
  H2 --> L
```

`llama-quantize` and `llama-imatrix` are already compiled. The starting measurement uses the full BF16 file only.

---

## Part 3. The project

### The project in one page

A **language model** is a program that continues text. You give it a conversation. It predicts the next piece of text, then the next, until it stops.

An **agent** is that model plus a rule: when the text is a request to use a program, actually run the program, paste the result back into the conversation, and let the model continue. The programs are called **tools**. In this project the tools are a pretend stock-trading account and a pretend inbox.

The whole agent has to run on one small computer, a **Raspberry Pi 5** with 8 GB of memory. Nothing is sent to a bigger computer to do the thinking. The course forbids that.

The hard part is length. Every tool reply is pasted onto the conversation. The next question sends the entire conversation again. A short chat becomes a very long document, and the small computer has to read all of it, every time, and still produce a correct tool request at the end.

```mermaid
flowchart LR
  Q[A person asks something] --> M[The model reads the whole chat]
  M --> T[The model asks a tool to do something]
  T --> R[The tool replies with text]
  R --> M
  M --> A[The model answers, or asks another tool]
```

Two jobs are measured separately:

- **Speed, memory, heat, and power** are measured on the Pi. That is the point of the course.
- **Did it get the tasks right?** The full exam can be scored on a normal computer, using the same model file and the same programs. Timing that exam on the Pi would take many hours.

---

### Words for the experiment

Parts 1 and 2 already define the board, tokens, the model file, and the programs. These are the words the experiment adds.

**Prompt.** The text we send in: the standing instructions plus the whole conversation so far.

**Prefill.** The model reading that prompt. On a long chat this is most of the wait.

**Decode.** The model writing new tokens, one by one, including its scratchpad and the tool request. Speed is quoted in tokens per second.

**Tool, schema, tool call.** A tool is a function the model may ask to run, such as "add MSFT to the watchlist." A schema is the written description of that function. A tool call is the model naming one and filling in the arguments. The harness checks the name and the required arguments, then runs it.

**BFCL.** The Berkeley Function Calling Leaderboard, a public exam for "can a model use tools?" It ships fake apps with real code, so a call can be executed and the resulting account can be checked. We use that code. We do not submit scores to their website.

**TradingBot and MessageAPI.** Two of those fake apps. TradingBot is a stock account with 20 actions. MessageAPI is an inbox with 10. The course requires TradingBot. MessageAPI brings the set to 30, the top of the allowed range.

**Turn.** One user message, plus everything the model and the tools do before the next user message.

**Session.** The whole task, several turns long.

**Prefix cache off.** A prefix is the start of the prompt that is identical to last time. The server will reuse that reading unless we say no. The starting experiment says no (`cache_prompt: false`). Every turn, the full conversation is sent and read again. That is the "before" picture. Turning the cache on is a later experiment.

**Thinking mode.** Qwen3 can write a private scratchpad before it answers. With thinking off, the model said it would add money to an account and then stopped without calling `fund_account`. With thinking on, it called `fund_account(amount=2203.4)`. The starting setup leaves thinking on. That is also Qwen's default.

**Sampling.** When several next tokens are plausible, sampling picks one. Temperature 0.6, top-p 0.95, and top-k 20 are Qwen's recommendation for thinking mode. Temperature 0 would always pick the single most likely token. The course note asked us not to use that here. If thinking is compared later with thinking off, that recipe is temperature 0.7, top-p 0.8, top-k 20.

**System prompt.** The standing instructions at the top of every chat, including the 30 tool descriptions.

**Train split and test split.** 26 tasks we may later learn from, and 12 exam tasks we score and do not tune on. The exam tasks are held out.

**Early detail.** A fact near the start of a long chat that a later question asks for. Two exam tasks are built this way.

**Baseline.** The starting configuration, before any speedup. Later experiments change one setting and say whether the change was kept.

---

### What the course asks for, in order

The handout wants a before-and-after story. First measure the plain setup. Then change one thing, measure again on the same exam, and keep the change only if it helps.

| When | Due | What "done" means, in plain words |
|---|---|---|
| Week 1 | 29 Nov 2026 | The tools, the questions, the harness, and a measured starting run on the Pi |
| Week 2 | 6 Dec 2026 | Reuse work the model already did, and try smaller copies of the weights |
| Final talk and demo | 20 Dec 2026 | One more way to shrink the model, one way to handle long chats, and a live demo |
| Written report | 24 Dec 2026 | The code and a report that matches the numbers |

What Week 1 includes:

- **The questions.** 20 to 30 tools, TradingBot included, chats that get long, at least five tasks meant to reach 16k–32k tokens, at least two where the answer depends on something said early, and an exam set we do not tune on.
- **The harness.** For every turn, write down how many tokens were sent, how many were read again, how long reading took, how long writing took, how long the tool took, and whether the tool call was well formed.
- **The characterization.** Time reading and writing at three or more lengths. Say how much memory was used. Check whether writing speed is limited by memory bandwidth. Report the clock time of a turn and of a whole session.

What every later change must still report: did the session succeed, were the calls valid, the time split into reading / writing / tool, how long until the first call on a cold start and on a warm start, peak memory, energy for the session, and whether the chip got hot enough to slow down.

---

### How one question moves through the system

```mermaid
flowchart TB
  subgraph pi [On the Raspberry Pi]
    U[A saved user message] --> H[Our harness]
    H -->|sends the entire chat, and says do not reuse last time| S[llama-server]
    S --> M[Qwen3 reads it and writes]
    M --> C{Did it ask for a tool?}
    C -->|yes| B[TradingBot or the inbox runs]
    B --> H
    C -->|no| E[This turn is over]
    V[Once a second: watts, degrees, memory] --> J[A log line]
    H --> J
  end
  J --> D[A JSON file for the website]
  D --> P[The public dashboard]
  H -.->|same model file, on a normal computer| A[Score all 12 exam tasks]
```

Step by step:

1. The harness builds the message. At the top are the standing instructions and all 30 tool descriptions. Under that is every earlier user message, model reply, and tool result.
2. It sends that to llama-server with `cache_prompt: false`, so the server reads it from the first token.
3. The model first writes its thinking, then either a tool call or a normal reply. It may write up to 2,048 new tokens (`max_tokens`). An earlier cap of 512 was small enough that thinking could use the whole budget and the tool call never appeared.
4. If there is a tool call, the harness checks the name and the required arguments, runs the fake app, and appends the app's reply.
5. The next request includes that reply. The old text is sent again.

The dotted arrow is the accuracy run. It uses the same weights and the same harness. It does not have to be timed on the Pi. The command is `./scripts/run_host_accuracy.sh`.

---

### The questions and the two fake apps

The list of tools is `configs/tool_subset.json`.

| Fake app | How many actions | What a person would recognize |
|---|---|---|
| TradingBot | 20 | Look up a stock, see the balance, place or cancel an order, edit a watchlist, read past deposits and withdrawals |
| MessageAPI | 10 | List messages, send one, delete one, similar inbox actions |

The questions come from BFCL's long-chat set, file `BFCL_v4_multi_turn_long_context.json`. A question is kept only when every app it needs is one of those two, and it does use TradingBot. `scripts/prepare_tasks.py` copies them into three files.

| Pile | How many | File | What it is for |
|---|---|---|---|
| Questions kept from BFCL | 33 | | The raw pool |
| Practice | 26 | `data/tasks/train.jsonl` | May be used later to build a smaller model. Not the reported exam |
| Exam | 12 | `data/tasks/test.jsonl` | Scored, and not used to tune |
| Extra long tasks we wrote | 5 | `data/tasks/long_extra.jsonl` | Meant to push length. Two of them are inside the exam because they hide an early fact |

The timed run on the Pi uses one short extra task, `long_extra_04_watchlist`. The user speaks twice. First: look up the transaction history, look at the watchlist, and add the symbol MSFT. Second: look at the watchlist again and confirm MSFT is there. MSFT is the ticker for Microsoft. The task is short on purpose. A longer trading task, `long_extra_03_fund_after_dump`, previously made the Pi read about 36,000 tokens. At about 1.4 tokens per second that is about 7 hours. The course note said to treat anything well above about 10,000 tokens as too long for this starting setup, and to leave those chats for the later speedups.

#### Why some tool replies are enormous

Long-context mode in TradingBot pads some replies with made-up extra records. The real answer is at the front. The padding is not saved into the account. It exists so the chat gets long.

Because this model spends a whole token on each digit, a page of fake prices and timestamps becomes a huge prompt. A course note measured the uncapped tools:

| Reply | Before we cut the padding | After the cut |
|---|---|---|
| One stock quote, `get_stock_info` | about 31,600 tokens | about 7,900 tokens |
| Transaction history | about 49,300 tokens | about 4,900 tokens |
| A typical TradingBot chat, counting tool text only | middle of the pack 32,000–36,000 tokens, longest about 67,000 | middle 8,000–11,000, longest about 18,000 |

The other fake apps in that note stayed near 3,000 tokens. The "~11,000 tokens" in the course handout matches the cut version, not the raw padding.

`setup.sh` adds four lines to the end of `long_context.py` if they are not already there:

```python
MA_5_EXTENSION = MA_5_EXTENSION[:500]
MA_20_EXTENSION = MA_20_EXTENSION[:500]
WATCH_LIST_EXTENSION = WATCH_LIST_EXTENSION[:500]
TRANSACTION_HISTORY_EXTENSION = TRANSACTION_HISTORY_EXTENSION[:100]
```

In ordinary language: keep the first 500 padded moving-average points, the first 500 padded watchlist entries, and the first 100 padded transactions. Drop the rest.

That file is the one both sides import. The harness runs the tools from it. BFCL's checker, the program that decides whether the final account looks right, imports the same file on this machine. If only one copy were cut, a run could do the right thing and still be marked wrong, because the checker would be simulating a different, longer tool. There is no second copy under the Python environment. Both `import bfcl_eval` and the harness point at `third_party/gorilla/.../long_context.py`.

The course note replayed the official answers for all 50 TradingBot long-context tasks with and without the cut. The final accounts matched, and no official call started failing. Our own check: after the cut, `get_stock_info("AAPL")` returned 7,983 characters of compact JSON, in the same range as "about 7,900 tokens" once digits are split apart.

The timed watchlist task is started without the long-padding switch. The cut still matters as soon as we turn that switch on.

---

### The starting settings, one by one

These live in `configs/baseline.yaml`. Later experiments change one row and say whether the change was kept.

| Setting | Value | What it means if you have not seen it |
|---|---|---|
| Weights | the BF16 GGUF | The full starting file. A faster 16-bit layout is the first experiment, and it does not replace this file |
| Context | 16,384 tokens | Course baseline cache; how long one chat may grow on the server |
| Reuse earlier reading | off | Every turn is read from the first word again |
| Tool descriptions | all 30, in the instructions, and also sent as a tool list | The model can see them, and the server can return a structured call |
| Thinking | on | The model may write scratch work before the call. Turning it off made it skip a required call |
| How the next word is chosen | temperature 0.6, top-p 0.95, top-k 20 | Qwen's published recipe for thinking mode. The non-thinking recipe, if we compare later, is 0.7, 0.8, and 20 |
| Longest reply | 2,048 new tokens | Enough room for scratch work and the call |
| Forced output format | off | We do not make the server reject text that fails a grammar. The baseline is "let the model speak" |
| Cores | 2 for writing, 4 for reading, 1 chat slot | Matches the 4-core board without reserving four copies of the cache |

```mermaid
sequenceDiagram
  participant H as Harness
  participant S as llama-server
  participant B as TradingBot
  H->>S: Entire chat so far, please read it all
  S-->>H: Scratch work, then a tool request
  H->>B: Run it, if the request is well formed
  B-->>H: Text result
  H->>S: The same chat again, with that result stuck on the end
```

The Pi script that measures this starting point is `./scripts/run_baseline.sh` (steps are listed under `pi_scenarios` in `configs/baseline.yaml` and can be run separately with `--only session`, `--only characterize`, or `--only export`). By default it does two things, aimed at about 10,000 tokens of work altogether:

1. The short watchlist task above.
2. A speed sweep: feed the model a dummy prompt of 512 tokens, then 2,048, then 4,096, and let it write 32 tokens each time. That separates "how long does reading take?" from the complications of a real tool chat.

A wrapper, `./scripts/sample_device.sh`, starts the power and temperature log before each of those and stops it when the command exits. `./scripts/export_dashboard.sh` copies the summary into the file the website reads.

---

### What gets written down

| What | Everyday meaning | Where it comes from |
|---|---|---|
| Prompt tokens | How long the chat was when we sent it | The server's own timing line |
| Cached tokens | How much of that the server skipped because it matched last time | Same line. In the baseline this should stay at zero, because we asked it not to skip |
| Generated tokens | How much the model wrote, including thinking | Same line |
| Prefill time | How long the reading took | Harness, per turn |
| Decode time | How long the writing took | Harness, per turn |
| Tool time | How long the fake app took. These apps are instant compared with the model | Harness |
| Wall time | Reading plus writing plus the tool, as a clock would see it | Harness |
| Call validity | The function exists, and the required arguments were filled in | Our checker |
| Session success | The account ended in the state the task asked for. Examples: the balance equals a number, the watchlist contains a symbol, an early fact was recalled | Per-task checks |
| Power | Watts, from voltage times current on each supply rail that reports both | About one sample a second. Energy in joules is the area under that curve |
| Temperature | Degrees Celsius of the main chip | `measure_temp` |
| Throttle | Whether the chip has cut its speed | `get_throttled` |
| Peak memory | The most RAM the server process itself used | High-water mark of that process, not "how full is the 8 GB" |
| Bandwidth check | A back-of-the-envelope: if each new token must reread the whole weight file, does the measured writing speed match the board's memory speed? | Computed after a sweep exists. Not computed yet |

Rails named `EXT5V` and `BATT` report a voltage and no current. They are left out of the watt sum. Inventing an input power from voltage alone would be a guess.

The throttle word is a bit field. The low half says what is true right now: undervoltage, frequency capped, throttled, soft temperature limit. The high half says what has happened at any time since boot. `0x80000` means "the soft temperature limit happened earlier." By itself that does not throw the run away. During the 26 Nov run the value was `0xe0006`, which means the frequency is capped and the CPU is throttled right now, and those events are also in the since-boot history.

The website keeps one card per run, named by date and time. The next run of the same kind is drawn against the previous one. A shorter reading time is better. A higher tokens-per-second number is better.

---

### What the 26 Nov run had shown when this was written

`./scripts/run_baseline.sh` started at 11:39 in the morning, Pakistan time, on 26 Nov 2026. The log folder is `results/baseline/history/2026-11-26T06-39-36Z` (`06:39` UTC is `11:39` PKT). At the time of writing, the watchlist task was still on its first reading. No score file existed yet. The 512 / 2,048 / 4,096 sweep had not started.

The server had finished 2,048 tokens of that first prompt and reported progress 0.37, so the prompt is about 5,500 tokens. Most of that is the 30 tool descriptions, not the user's two sentences.

| What we saw | Value | How to read it |
|---|---|---|
| Reading speed | 2,048 tokens in 3,839 seconds, which is **0.53 token per second** | About two seconds per token. A 5,500-token read is on the order of three hours by itself |
| Chip temperature | 93–95°C | Hot enough that the board is protecting itself |
| Speed limit | frequency capped and throttled, right now | The 0.53 figure is a hot-board figure |
| Power while hot | about 2.9–3.8 W | A few watts. The chip is not doing more work by drawing more power; it has slowed down |
| At the start of the run | 74.7°C, not actively throttled, about 3.6 W | It heated up during the first read |

An earlier, abandoned attempt on the long funding task moved about 36,000 tokens at about 1.4 tokens per second. 0.53 is slower than that. The difference lines up with this board sitting at 95°C. The course note also expects BF16 itself to be slower than FP16 on a Pi, because the chip has faster instructions for FP16. We have not converted the file, so this run does not test that idea. It only shows the starting file, on a hot board.

If the model asks for the three watchlist actions in one reply, the script can finish the same evening. If it asks for them one at a time, each reply causes another full read of a growing chat, and the script runs into the next morning. Either way, this is the "before" number the later experiments have to beat.

---

### What we will try afterwards

None of these have been run. The order follows the handout. FP16 is first because the course note pointed at it as the likely reason the starting speed is low.

```mermaid
flowchart TD
  B[Start here: full BF16 file, thinking on, reread the whole chat, short tasks on the Pi]
  B --> F[Store the same model as FP16 and time it again]
  F --> C[Let the server skip text it already read]
  C --> Q[Make at least two smaller copies of the weights]
  Q --> K[Also shrink the per-token notes, and check a task that needs an early fact]
  K --> R[One more way to shrink or adapt the model]
  R --> X[One way to stop the chat from growing without bound]
  X --> D[A live demo on a question that was not in the exam]
```

What each box means:

- **FP16.** Same model, different 16-bit packing, which this CPU multiplies faster. The BF16 file stays in the repo as the starting point.
- **Prompt cache.** Tell the server it may reuse the reading it did for the unchanged start of the chat. Only the new tool result and the new question should be read. The chat template must stay the one used at the start, or the comparison is unfair.
- **Smaller weight files.** At least two GGUF quantizations. An **importance matrix** (`llama-imatrix`) is a table of which weights the model uses most on our own chats. Quantizing with that table should damage the model less than quantizing with a generic text sample. We report both.
- **Smaller KV cache.** The per-token notes can be stored with fewer bits too. Q4 and Q8 are the obvious first tries. The course wants something beyond those, checked on a task that fails if an early detail was forgotten. Shrinking the notes is a likely way to forget them.
- **Thinking versus not thinking, at each size.** Smaller weights are expected to damage the scratchpad more than they damage a short tool call. If "thinking off" survives quantization better, that can be a step, and the talk has to show how much accuracy it gave up relative to this thinking-on start.
- **One further compression.** The handout allows pruning (deleting parts of the network), distillation (training a smaller model to imitate this one), LoRA (a small add-on trained for tool use), or outlier suppression (special handling for a few unusually large weights). A paper that trains those add-ons for Qwen3-1.7B and Qwen3-4B is listed in the references. If we do that, the training chats should include thinking, because thinking is the mode we are trying to keep.
- **Context management.** Some policy for long chats: summarize old turns, drop old tool output, or keep only the early fact the later question needs. One such policy is required.
- **Demo.** A person types into the web page or the terminal, on a question that was not in the 12-task exam or the practice set.

Flash attention stays on its default unless a measurement shows the server actually turned it off.

---

### Checklist

#### Already in place

- [x] llama.cpp built on this Pi, including the server, a one-shot chat tool, the quantizer, and the importance-matrix tool
- [x] Qwen3-1.7B downloaded and saved as a BF16 GGUF, and the server is running that file
- [x] 30 tools, 26 practice tasks, 12 exam tasks, 5 extra long tasks, 2 of them held out because they depend on an early fact
- [x] Harness that checks calls, refuses prefix reuse, and puts every tool description in the instructions
- [x] Thinking left on, with Qwen's thinking-mode sampling
- [x] TradingBot's made-up padding cut in the one file both the agent and the checker use
- [x] A log of watts, degrees, throttle, and server memory for the whole command
- [x] A website that keeps runs by date

#### Still required before the Week 1 talk

- [ ] Let the watchlist task and the three-length speed sweep finish
- [ ] Report reading time, writing time, peak server memory, energy, and whether the chip throttled
- [ ] Do the bandwidth check once those writing speeds exist
- [ ] Score the 12 exam tasks on a host with this same file
- [ ] Say in the talk that chats well above about 10,000 tokens are too slow for this starting setup, and are what the later experiments are for

#### After Week 1

- [ ] FP16, as the first speed comparison
- [ ] Turn prefix reuse on, with the same chat template
- [ ] Two or more smaller weight files, an importance matrix from our chats, and a smaller KV cache
- [ ] At each size, compare thinking on and thinking off against this starting accuracy
- [ ] One more compression method, and one policy for long chats
- [ ] A live demo on a fresh question
- [ ] The written report, due 24 Dec 2026

---

### A way to give the talk

1. Part 1, the board. Name the Pi 5, 8 GB, 4 cores, and that watts and degrees come from the board itself.
2. The memory picture in Part 1. The model file is 3.8 GB. The notes are 112 KB per token. A 32k chat does not fit. We serve the PDF baseline at 16,384 tokens.
3. Part 2, the programs. Two terminals: start the server, then run the baseline. Point at the program table only if someone asks which file does what.
4. Part 3, one page. One model, tools whose replies make the chat longer, and the whole chat is reread every time.
5. The request diagram. Walk one turn: harness, server, fake app, and the power log beside it.
6. The starting settings. Full file, thinking on, Qwen's sampling, no reuse. Mention the funding task: without thinking, the model promised to add money and never called the tool.
7. The padding story. Digits become tokens. Cutting the padding brings TradingBot back toward the length the handout describes.
8. The 26 Nov snapshot: about 0.53 token per second while the chip was at 93–95°C and throttled. Do not present it as the finished baseline.
9. The "what is next" diagram. Stop there. There are no quantization or caching results yet.
10. The checklist is the backup slide if someone asks what is finished.

---

### References

Each item says what it is, then the citation.

1. **The model.** Qwen Team. *Qwen3 Technical Report.* arXiv:2505.09388, 2025. <https://arxiv.org/abs/2505.09388>. The model card, including the thinking-mode sampling settings, is <https://huggingface.co/Qwen/Qwen3-1.7B>.
2. **The program that runs the model on the Pi.** ggml-org. *llama.cpp.* The copy here is commit `d1eb0de`. <https://github.com/ggml-org/llama.cpp>.
3. **The tool-use exam and the fake apps.** Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. *The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models.* Proceedings of the 42nd International Conference on Machine Learning, PMLR 267, pages 48371–48392, 2025. <https://proceedings.mlr.press/v267/patil25a.html>. How multi-turn chats are judged: <https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html>. Code: <https://github.com/ShishirPatil/gorilla>.
4. **A paper about teaching small Qwen models to use tools, relevant if we train a small add-on after quantization.** Alexander Kovalenko and Artem Redchych. *Small Agents Should Know What Their Tools Do.* arXiv:2610.35981, 2026. <https://arxiv.org/abs/2610.35981>.
5. **The method that paper uses for the add-on (LoRA: train a few extra low-rank matrices instead of all 1.7 billion weights).** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685, 2021. <https://arxiv.org/abs/2106.09685>.
6. **The board.** Raspberry Pi Ltd. *Raspberry Pi 5.* <https://www.raspberrypi.com/products/raspberry-pi-5/>.
7. **The assignment.** AI624 course manual, Fall 2026, Track A, Project A1. [Project_Description_AI624_2026.pdf](Project_Description_AI624_2026.pdf). The uncapped and capped token counts in the questions section are from the course note about TradingBot's padding. They are not a finished measurement from this repository. Our local check is the 7,983-character stock quote.
