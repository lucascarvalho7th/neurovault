# NeuroVault

**A simulated brain-to-vault pipeline: EEG signal → intent detection → thought decoding → Claude → Obsidian.**

NeuroVault is a sandbox for one question: *what would a "second brain" look like if notes came straight from your head, with no typing and no manual "dump" step?*

Real thought-to-text hardware isn't something you can buy yet. Implanted speech BCIs decode inner speech in research labs, and non-invasive wearables are still early. So NeuroVault simulates the parts that don't exist yet and keeps every stage behind a small interface. When real hardware or a real decoder becomes available, you swap one module and the rest of the pipeline stays the same.

```
┌──────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌────────────────┐
│ Signal source│──▶│ Intent gate │──▶│   Decoder   │──▶│  Organizer   │──▶│ Obsidian vault │
│ simulated /  │   │ "am I       │   │ brain → text│   │ Claude fixes,│   │ Ideas/ Tasks/  │
│ BrainFlow    │   │ thinking a  │   │ (mock, with │   │ classifies,  │   │ Decisions/ ... │
│ (EEG)        │   │ note now?"  │   │ real errors)│   │ titles, links│   │ Daily log      │
└──────────────┘   └─────────────┘   └─────────────┘   └──────────────┘   └────────────────┘
```

## What each stage does

| Stage | Today (simulated) | Swap for (real) |
|---|---|---|
| **Signal source** | `SimulatedSource`: numpy EEG with an alpha-dominant resting state and beta bursts when you "intend" to think a note. `BrainFlowSource` with BrainFlow's synthetic board. | Any BrainFlow headset (OpenBCI, Muse, ...) by changing `--board-id`. |
| **Intent gate** | Calibrates a resting baseline, then opens only on *sustained* focus (`beta / (theta+alpha+beta)`). Single-window blips are ignored. This is the "mental on-switch" that keeps private thoughts private. | A trained intent classifier, or a mental password like the one used in Stanford's inner-speech study. |
| **Decoder** | `MockDecoder`: emits scripted "intended" thoughts with a configurable word error rate (substitutions, deletions, repetitions), and garbled low-confidence output on false captures. | A real brain-to-text model (Brain-to-Text '24/'25 baselines, Meta Brain2Qwerty). |
| **Organizer** | Claude repairs decoding errors, classifies (idea / task / decision / question / reflection), writes a title, tags it and links it to existing notes. Two backends with the same prompt and schema: `ClaudeCodeOrganizer` (Claude Code CLI, your subscription) and `ClaudeOrganizer` (API key). `OfflineOrganizer`: keyword rules, no network. | Same. This stage is already real. |
| **Vault** | Plain Markdown + YAML frontmatter, `[[wikilinks]]`, a daily log and a JSON summary per session. | Same. Point `--vault` at your real Obsidian vault. |

## Quick start

```bash
git clone https://github.com/lucascarvalho7th/neurovault.git
cd neurovault
pip install -e ".[dev]"

# Runs fully offline (no API key, no hardware):
python -m neurovault run --organizer offline

# Claude as the organizer, through Claude Code (uses your Claude subscription):
python -m neurovault run --organizer claude-code

# Claude as the organizer, through the API (uses Console credits):
export ANTHROPIC_API_KEY=sk-ant-...        # Windows: set ANTHROPIC_API_KEY=...
python -m neurovault run --organizer claude

# Portuguese thoughts, noisier decoder, real-time pacing:
python -m neurovault run --thoughts examples/thoughts_pt.txt --wer 0.25 --realtime
```

Then open `./demo-vault` in Obsidian (*Open folder as vault*).

Every session also writes a **before/after report** to `Reports/` inside the vault: raw decoder output next to the filed note, plus the word error rate before and after the organizer. Print the latest one with:

```bash
python -m neurovault report --vault ./demo-vault
```

### Example session (offline organizer)

```
Calibrating resting baseline (stay relaxed)...
Baseline set. Gate threshold = 0.220. Capturing.
  [ 11.0s] gate OPEN  (focus 0.36)
  [ 11.0s] decoded (0.62): What if the viult cuuld sugget which old notes I should should revisit each wek
           -> idea       Ideas/What if the viult cuuld sugget which.md
  [ 23.0s] gate OPEN  (focus 0.50)
  [ 23.0s] decoded (0.87): I need renew the domain for my portfolio website before Friday
           -> task       Tasks/I need renew the domain for my.md
...
Session done.
  episodes captured : 6
  notes written     : 6
  discarded/skipped : 0
  gate recall       : 100% of intended thoughts caught
  false captures    : 0
```

The decoded lines are intentionally broken, the way real decoders are. The offline organizer files them as-is, so the typos end up in the titles. That's exactly the gap the Claude organizer fills: it reconstructs the intended sentence, writes a clean title and links the note to related ones, while keeping the raw decoder output in a collapsed callout for reference.

## Results: before and after Claude

Same simulated session, same decoding errors, two organizers. Six Portuguese thoughts, decoder word error rate set to 15%.

| | Offline rules | Claude (Haiku, via Claude Code) |
|---|---|---|
| Word error rate vs. intended thought | 21% | **2%** |
| Decoding errors removed | 0% | **90%** |
| Correct note type | 3 of 6 | **5 of 6**\* |
| Links between notes | 0 | 1 |

| Decoder output (before) | Filed note (after Claude) | Type |
|---|---|---|
| Decudi usar o Clude para a revisão semnal em vez de fazer ne mão | **Usar o Claude para a revisão semanal**<br>Decidi usar o Claude para a revisão semanal em vez de fazer na mão. | decision |
| Prciso ranovar o domínio do meu site de portfólio antes de sexta | **Renovar domínio do site de portfólio até sexta**<br>Preciso renovar o domínio do meu site de portfólio antes de sexta. | task |
| Hoje foi um dia lento mos terminar aquele projeto du automação valeu a peni | **Dia lento, mas projeto de automação concluído**<br>Hoje foi um dia lento, mas terminar aquele projeto de automação valeu a pena. | reflection |

Where the decoder *dropped* a word, Claude can't bring it back: in "testar o headset com a ~~placa~~ de playback" the missing word stays missing, which is the right behavior. Repairing is not inventing.

\*Checked by hand against the intended type. The miss: "Por que eu sempre tenho ideias melhores andando…?" was filed as a reflection instead of a question.

The full report for any session is written to `Reports/` in the vault. The one behind these numbers is in [`docs/sample-report.md`](docs/sample-report.md).

## CLI options

| Option | Default | What it does |
|---|---|---|
| `--vault` | `./demo-vault` | Where notes are written |
| `--source` | `simulated` | `simulated` or `brainflow` |
| `--board-id` | `-1` | BrainFlow board (-1 = synthetic) |
| `--serial-port` | | Port for real boards (e.g. `COM3`) |
| `--duration` | `60` | Session length in seconds |
| `--thoughts` | `examples/thoughts_en.txt` | Script of intended thoughts |
| `--wer` | `0.12` | Simulated word error rate |
| `--organizer` | `auto` | `claude-code` (subscription), `claude` (API key), `offline`, or `auto` (API if a key is set, else offline) |
| `--model` | `haiku` / `claude-haiku-5-5` | Model for Claude Code / the API (or `$NEUROVAULT_MODEL`) |
| `--review-below` | `0.6` | Tag low-confidence notes `needs-review` |
| `--discard-below` | `0.3` | Drop decodes that are probably noise |
| `--realtime` | off | Simulated source runs at wall-clock speed |

> **Note on `--source brainflow`:** BrainFlow's synthetic board produces a steady signal with no intent bursts, so the gate (correctly) captures nothing. It's there to prove the hardware path works end to end. Use `simulated` to see captures, or a real headset.

## Project layout

```
neurovault/
  signal_source.py   SimulatedSource, BrainFlowSource
  intent_gate.py     band powers, focus index, IntentGate
  decoder.py         Decoder protocol, MockDecoder, word-error simulation
  organizer.py       ClaudeCodeOrganizer (CLI, JSON schema), ClaudeOrganizer (API, forced tool call), OfflineOrganizer
  vault.py           Obsidian writer (notes, daily log, reports, session JSON)
  report.py          before/after report and word error rate
  pipeline.py        the capture loop
  cli.py             command line
examples/            thought scripts (EN / PT)
docs/architecture.md design notes and how to plug in real parts
tests/               pytest suite, no hardware or API key needed
```

## Roadmap

- [ ] Playback real EEG recordings with `BrainFlowSource` (`PLAYBACK_FILE_BOARD`)
- [ ] Plug in a real decoder trained on the public Brain-to-Text dataset
- [ ] Expose the pipeline as an MCP server so Claude can query captures directly
- [ ] Weekly "resurface" job: Claude suggests old notes worth revisiting
- [ ] Test the intent gate on a consumer EEG headset

## Honest limitations

- No consumer device today can read free-form thoughts. Research systems decode **attempted or inner speech** (motor cortex), not memories, and the best results still need surgery.
- The focus index is a simplified EEG signature of mental effort. It's good enough to drive a simulation, not a medical-grade measure.
- The mock decoder knows the "right" answer. It's a stand-in so the rest of the system can be built and tested now.

## License

MIT © Lucas Carvalho

---

## 🇧🇷 Resumo em português

**NeuroVault** é um laboratório simulado de um "segundo cérebro" que recebe notas direto da cabeça, sem digitar e sem a etapa manual de "despejo".

Como o hardware que lê pensamentos ainda não está disponível, o projeto simula as partes que não existem e deixa cada etapa isolada atrás de uma interface simples:

1. **Fonte de sinal:** EEG simulado (ou via BrainFlow, que também aceita headsets reais como OpenBCI e Muse).
2. **Portão de intenção:** só abre quando detecta foco *sustentado*. Ignora picos isolados, funcionando como um "interruptor mental" para não capturar pensamentos privados.
3. **Decodificador:** simula um modelo cérebro→texto com erros reais (palavras trocadas, cortadas, repetidas). No futuro, pode ser trocado por um modelo de verdade treinado nos datasets públicos.
4. **Organizador:** o Claude corrige os erros de decodificação, classifica (ideia, tarefa, decisão, pergunta, reflexão), cria título, tags e links para notas existentes. Funciona pelo Claude Code (usando a assinatura do Claude) ou pela API. Existe também um modo offline, sem rede.
5. **Vault:** notas em Markdown compatíveis com o Obsidian, com log diário e resumo de cada sessão.

**Para rodar:**

```bash
pip install -e ".[dev]"
python -m neurovault run --organizer offline                     # sem chave, sem hardware
python -m neurovault run --organizer claude-code --thoughts examples/thoughts_pt.txt   # com o Claude
```

**Resultado:** na mesma sessão simulada, a taxa de erro de palavras caiu de 21% (texto bruto do decodificador) para 2% depois do Claude, que também acertou a classificação de 5 das 6 notas (contra 3 de 6 no modo offline).

Depois é só abrir a pasta `demo-vault` no Obsidian. Cada sessão também gera um relatório de "antes e depois" na pasta `Reports`, comparando o texto bruto do decodificador com a nota final e mostrando quantos erros foram corrigidos. A ideia é que, quando surgir hardware ou um decodificador real, baste trocar um módulo, sem mexer no resto.
