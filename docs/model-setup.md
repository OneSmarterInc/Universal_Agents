# Model setup

The agents use **Qwen on Ollama** by default, through Ollama's OpenAI-compatible
address. Only `agents_service/ai/` talks to a model, so switching is a settings change.

## Default: Ollama on the same machine

If you already use `kormic-local` for Kormic Careers, the agents can use the same
model. The default settings are:

```
AGENTS_MODEL_PROVIDER=ollama
AGENTS_MODEL_BASE_URL=http://127.0.0.1:11434/v1
AGENTS_MODEL_NAME=kormic-local
AGENTS_MODEL_REASONING_EFFORT=none
AGENTS_MODEL_CONTEXT_TOKENS=16384
```

To create the model on a new machine (laptop or server), use the same Modelfile
settings as `kormic-local`:

```
ollama pull qwen3:1.7b
```

Modelfile:

```
FROM qwen3:1.7b
PARAMETER temperature 0
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER num_ctx 16384
PARAMETER repeat_penalty 1
```

```
ollama create kormic-local -f Modelfile
agents model-check
```

`AGENTS_MODEL_CONTEXT_TOKENS` must equal `num_ctx`. Thinking stays off through
`AGENTS_MODEL_REASONING_EFFORT=none` (the default for Ollama).

### Recommended for servers: the checked Q8_0 file

`ollama pull qwen3:1.7b` gives whatever the Ollama registry holds that day (Q4_K_M).
For a server, download Qwen's own file once, check it, and build `kormic-local` from it:

| | |
|---|---|
| Source | `Qwen/Qwen3-1.7B-GGUF` on Hugging Face, revision `90862c4b9d2787eaed51d12237eafdfe7c5f6077` |
| File | `Qwen3-1.7B-Q8_0.gguf`, 1,834,426,016 bytes |
| SHA256 | `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a` |
| Licence | Apache-2.0 (keep Qwen's licence notice with the file) |

```
sha256sum Qwen3-1.7B-Q8_0.gguf   # must match the SHA256 above
```

Use the Modelfile above with `FROM ./Qwen3-1.7B-Q8_0.gguf` in place of `FROM qwen3:1.7b`, then
`ollama create kormic-local -f Modelfile` and `agents model-check`. Q8_0 is closer to the original
model than Q4_K_M but larger and slower on a CPU; run `agents accuracy` after switching.

## Ollama on another machine

```
AGENTS_MODEL_BASE_URL=https://models.internal.example/v1
```

Use HTTPS. Plain HTTP to another machine is refused unless `AGENTS_MODEL_ALLOW_HTTP=true`,
which is meant only for a private network such as a Docker network.

## Faster: Qwen on a GPU (vLLM)

vLLM serves Qwen on a GPU with the same OpenAI-compatible address and answers many
requests at once. Point the agents at it; nothing else changes:

```
AGENTS_MODEL_PROVIDER=openai
AGENTS_MODEL_BASE_URL=https://gpu-host.internal/v1
AGENTS_MODEL_NAME=Qwen/Qwen3-1.7B
AGENTS_MODEL_API_KEY=<vLLM --api-key value, if set>
```

No `reasoning_effort` is sent to these servers unless `AGENTS_MODEL_REASONING_EFFORT` is set.

## Later: an API key model

```
AGENTS_MODEL_PROVIDER=anthropic
AGENTS_MODEL_NAME=<model id>
AGENTS_MODEL_API_KEY=<key>
```

The limits default to 8000 output and 131072 context tokens for this provider; set
`AGENTS_MODEL_OUTPUT_TOKENS` and `AGENTS_MODEL_CONTEXT_TOKENS` to change them.

Install the extra once: `pip install -e ".[anthropic]"`. Large API models read a
whole document in one request; small local models read it section by section.

## One agent on a different model

```
AGENTS_MODEL_OVERRIDES={"linkedin": {"provider": "anthropic", "name": "<model id>", "api_key": "<key>"}}
```

Any `model_*` setting can be overridden per agent (`provider`, `base_url`, `name`,
`api_key`, `structured_mode`, `reasoning_effort`, `context_tokens`, `output_tokens`,
`allow_http`, `request_seconds`, `text_first`).

An override that changes the provider or the server:

- needs its own `api_key` (the global key is never sent to another server);
- gets that provider's default limits and `reasoning_effort`, unless it sets them.

| Provider | context_tokens | output_tokens | reasoning_effort |
|---|---|---|---|
| ollama | 16384 | 2048 | none |
| openai | 16384 | 2048 | not sent |
| anthropic | 131072 | 8000 | not sent |

## After any model change

1. `agents model-check`
2. `agents accuracy` and compare the score with the previous run
3. Every result's `model` field shows which model produced it.
