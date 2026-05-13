"""Target registry.

Builds the list of ``Provider`` instances the daemon will pick from.
Grouped by category so the operator can disable categories via the
``CATEGORIES`` env var or individual targets via the UI.

All targets are legitimate public AI services reviewed as of April 2026.
API probes send realistic SDK-shaped JSON bodies but intentionally fail
auth (Cato identifies the application regardless; the 401/403 is
expected).

Curation criteria:
* Domain resolves and serves content as of the last review pass.
* Recognized by major SASE / DLP vendors in their AI/GenAI app catalogs.
* Distinct enough from its neighbors to produce a separate app-ID hit.
"""
from __future__ import annotations

from typing import Any

from . import mcp
from .providers import (
    ApiProbe,
    DuckDuckGoChat,
    MCPAuthedProbe,
    PollinationsImage,
    PollinationsText,
    Provider,
    WebProbe,
)


# ---------------------------------------------------------------------------
# Category: llm_api — major LLM vendor API endpoints
# ---------------------------------------------------------------------------

def _llm_api_probes() -> list[Provider]:
    def chat_body(model: str):
        def builder(p: str) -> dict[str, Any]:
            return {
                "model": model,
                "messages": [{"role": "user", "content": p}],
                "temperature": 0.7,
            }
        return builder

    def anthropic_body(p: str) -> dict[str, Any]:
        return {
            "model": "claude-sonnet-4-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": p}],
        }

    def gemini_body(p: str) -> dict[str, Any]:
        return {"contents": [{"parts": [{"text": p}]}]}

    def cohere_body(p: str) -> dict[str, Any]:
        return {
            "model": "command-r",
            "messages": [{"role": "user", "content": p}],
        }

    return [
        # Frontier labs
        ApiProbe(
            name="OpenAI",
            url="https://api.openai.com/v1/chat/completions",
            user_agent="OpenAI/Python 1.58.0",
            body_builder=chat_body("gpt-4o-mini"),
        ),
        ApiProbe(
            name="Anthropic",
            url="https://api.anthropic.com/v1/messages",
            user_agent="anthropic-python/0.42.0",
            body_builder=anthropic_body,
            extra_headers={"anthropic-version": "2023-06-01"},
            send_fake_auth=False,
        ),
        ApiProbe(
            name="Google-Gemini",
            url=("https://generativelanguage.googleapis.com/v1beta/models/"
                 "gemini-1.5-flash:generateContent"),
            user_agent="google-generativeai/0.8.3 gl-python/3.12",
            body_builder=gemini_body,
            send_fake_auth=False,
        ),
        ApiProbe(
            name="xAI-Grok",
            url="https://api.x.ai/v1/chat/completions",
            user_agent="xai-sdk/0.1.0",
            body_builder=chat_body("grok-2-latest"),
        ),
        ApiProbe(
            name="Mistral",
            url="https://api.mistral.ai/v1/chat/completions",
            user_agent="mistralai/1.2.0",
            body_builder=chat_body("mistral-small-latest"),
        ),
        ApiProbe(
            name="Cohere",
            url="https://api.cohere.com/v2/chat",
            user_agent="cohere-python/5.11.0",
            body_builder=cohere_body,
        ),
        ApiProbe(
            name="Perplexity-API",
            url="https://api.perplexity.ai/chat/completions",
            user_agent="perplexity-python/0.1.0",
            body_builder=chat_body("sonar"),
        ),
        ApiProbe(
            name="DeepSeek-API",
            url="https://api.deepseek.com/chat/completions",
            user_agent="OpenAI/Python 1.58.0",
            body_builder=chat_body("deepseek-chat"),
        ),
        # Fast-inference providers
        ApiProbe(
            name="Groq",
            url="https://api.groq.com/openai/v1/chat/completions",
            user_agent="groq/0.13.0",
            body_builder=chat_body("llama-3.3-70b-versatile"),
        ),
        ApiProbe(
            name="TogetherAI",
            url="https://api.together.xyz/v1/chat/completions",
            user_agent="together-python/1.3.3",
            body_builder=chat_body("meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        ),
        ApiProbe(
            name="Fireworks-AI-API",
            url="https://api.fireworks.ai/inference/v1/chat/completions",
            user_agent="fireworks-ai/0.15.0",
            body_builder=chat_body(
                "accounts/fireworks/models/llama-v3p3-70b-instruct"
            ),
        ),
        ApiProbe(
            name="DeepInfra",
            url="https://api.deepinfra.com/v1/openai/chat/completions",
            user_agent="OpenAI/Python 1.58.0",
            body_builder=chat_body("meta-llama/Meta-Llama-3.1-70B-Instruct"),
        ),
        ApiProbe(
            name="Hyperbolic",
            url="https://api.hyperbolic.xyz/v1/chat/completions",
            user_agent="OpenAI/Python 1.58.0",
            body_builder=chat_body("meta-llama/Meta-Llama-3.1-70B-Instruct"),
        ),
        ApiProbe(
            name="Novita-AI",
            url="https://api.novita.ai/v3/openai/chat/completions",
            user_agent="OpenAI/Python 1.58.0",
            body_builder=chat_body("meta-llama/llama-3.1-70b-instruct"),
        ),
        ApiProbe(
            name="SambaNova",
            url="https://api.sambanova.ai/v1/chat/completions",
            user_agent="OpenAI/Python 1.58.0",
            body_builder=chat_body("Meta-Llama-3.1-70B-Instruct"),
        ),
        ApiProbe(
            name="Cerebras",
            url="https://api.cerebras.ai/v1/chat/completions",
            user_agent="cerebras-cloud-sdk/1.16.0",
            body_builder=chat_body("llama3.1-70b"),
        ),
        ApiProbe(
            name="Lepton-AI",
            url="https://api.lepton.ai/api/v1/chat/completions",
            user_agent="OpenAI/Python 1.58.0",
            body_builder=chat_body("llama3-1-70b"),
        ),
        # Model-specific & vendor endpoints
        ApiProbe(
            name="AI21-Labs",
            url="https://api.ai21.com/studio/v1/chat/completions",
            user_agent="ai21-python/2.15.0",
            body_builder=chat_body("jamba-large"),
        ),
        ApiProbe(
            name="Voyage-AI",
            url="https://api.voyageai.com/v1/embeddings",
            user_agent="voyageai/0.3.0 python/3.12",
            body_builder=lambda p: {"input": [p], "model": "voyage-3"},
        ),
        ApiProbe(
            name="Writer",
            url="https://api.writer.com/v1/chat",
            user_agent="writer-sdk/2.0.0",
            body_builder=chat_body("palmyra-x-004"),
        ),
        # Enterprise cloud LLM endpoints (unauth call goes to the edge;
        # Cato sees the vendor-specific hostname pattern either way)
        ApiProbe(
            name="Azure-OpenAI",
            url=("https://hairspray-lab.openai.azure.com/openai/deployments/"
                 "gpt-4o-mini/chat/completions?api-version=2024-10-21"),
            user_agent="openai/1.58.0 azure",
            body_builder=chat_body("gpt-4o-mini"),
        ),
        ApiProbe(
            name="AWS-Bedrock",
            url=("https://bedrock-runtime.us-east-1.amazonaws.com/model/"
                 "anthropic.claude-3-5-sonnet-20241022-v2:0/invoke"),
            user_agent="aws-sdk-python/1.35.0",
            body_builder=lambda p: {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": p}],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Category: chatbot_ui — consumer chatbot front-ends
# ---------------------------------------------------------------------------

_CHATBOT_UIS: list[tuple[str, str]] = [
    # Frontier-lab branded UIs
    ("ChatGPT",              "https://chatgpt.com/"),
    ("Claude.ai",            "https://claude.ai/"),
    ("Google-Gemini-Web",    "https://gemini.google.com/"),
    ("Grok-Web",             "https://grok.com/"),
    ("Perplexity-Web",       "https://www.perplexity.ai/"),
    ("Microsoft-Copilot",    "https://copilot.microsoft.com/"),
    ("Meta-AI",              "https://www.meta.ai/"),
    ("Mistral-Chat",         "https://chat.mistral.ai/"),
    ("DeepSeek-Chat",        "https://chat.deepseek.com/"),
    # Aggregated / multi-model UIs
    ("Poe",                  "https://poe.com/"),
    ("HuggingChat",          "https://huggingface.co/chat/"),
    ("You.com",              "https://you.com/"),
    ("Phind",                "https://www.phind.com/"),
    ("Lmarena",              "https://lmarena.ai/"),
    ("Openrouter-Chat",      "https://openrouter.ai/chat"),
    ("DuckDuckGo-AI",        "https://duckduckgo.com/?q=test&ia=chat"),
    ("Brave-Leo",            "https://leo.brave.com/"),
    # Personality / roleplay / companion
    ("Character.AI",         "https://character.ai/"),
    ("Replika",              "https://replika.com/"),
    ("Janitor-AI",           "https://janitorai.com/"),
    ("Pi-AI",                "https://pi.ai/"),
    # Regional + emerging
    ("Kimi",                 "https://kimi.moonshot.cn/"),
    ("Qwen-Chat",            "https://chat.qwen.ai/"),
    ("Yi-Chat",              "https://www.lingyiwanwu.com/"),
    ("Doubao",               "https://www.doubao.com/chat/"),
    # Search-style
    ("Andi",                 "https://andisearch.com/"),
    ("Komo",                 "https://komo.ai/"),
    # Enterprise / writing-focused
    ("Jasper",               "https://www.jasper.ai/"),
    ("Writesonic",           "https://writesonic.com/"),
    ("Copy.ai",              "https://www.copy.ai/"),
    ("Rytr",                 "https://rytr.me/"),
    ("Sudowrite",            "https://www.sudowrite.com/"),
]


def _chatbot_ui_visits() -> list[Provider]:
    return [WebProbe(name, url, category="chatbot_ui") for name, url in _CHATBOT_UIS]


# ---------------------------------------------------------------------------
# Category: media_gen — image / video / audio / avatar generation
# ---------------------------------------------------------------------------

_MEDIA_GEN: list[tuple[str, str]] = [
    # Image — core
    ("Midjourney",            "https://www.midjourney.com/"),
    ("Ideogram",              "https://ideogram.ai/"),
    ("Recraft",               "https://www.recraft.ai/"),
    ("Leonardo-AI",           "https://leonardo.ai/"),
    ("Stability-AI",          "https://stability.ai/"),
    ("Stable-Diffusion-Web",  "https://stablediffusionweb.com/"),
    ("Black-Forest-Labs",     "https://blackforestlabs.ai/"),
    ("Adobe-Firefly",         "https://firefly.adobe.com/"),
    ("Freepik-AI",            "https://www.freepik.com/ai/image-generator"),
    ("Craiyon",               "https://www.craiyon.com/"),
    ("NightCafe",             "https://creator.nightcafe.studio/"),
    ("Civitai",               "https://civitai.com/"),
    ("Tensor-Art",            "https://tensor.art/"),
    ("PicLumen",              "https://piclumen.com/"),
    ("Playground-AI",         "https://playground.com/"),
    ("DeepAI",                "https://deepai.org/"),
    ("Canva-Magic",           "https://www.canva.com/magic-studio/"),
    # Video
    ("Runway",                "https://runwayml.com/"),
    ("Pika",                  "https://pika.art/"),
    ("Kling-AI",              "https://www.klingai.com/"),
    ("Luma-Labs",             "https://lumalabs.ai/dream-machine"),
    ("Hailuo-AI",             "https://hailuoai.com/"),
    ("Google-Flow",           "https://flow.google/"),
    ("Higgsfield",            "https://higgsfield.ai/"),
    ("Wavespeed-AI",          "https://wavespeed.ai/"),
    ("Viggle",                "https://viggle.ai/"),
    ("Vidu",                  "https://www.vidu.studio/"),
    # Avatar / talking-head
    ("HeyGen",                "https://www.heygen.com/"),
    ("Synthesia",             "https://www.synthesia.io/"),
    ("D-ID",                  "https://www.d-id.com/"),
    ("Captions",              "https://www.captions.ai/"),
    ("Hedra",                 "https://www.hedra.com/"),
    # Audio / music / voice
    ("Suno",                  "https://suno.com/"),
    ("Udio",                  "https://www.udio.com/"),
    ("ElevenLabs",            "https://elevenlabs.io/"),
    ("Murf",                  "https://murf.ai/"),
    ("Play.ht",               "https://play.ht/"),
    ("Speechify",             "https://speechify.com/"),
    ("Resemble-AI",           "https://www.resemble.ai/"),
    ("Descript",              "https://www.descript.com/"),
    ("AssemblyAI",            "https://www.assemblyai.com/"),
    # Design / 3D / misc
    ("Meshy",                 "https://www.meshy.ai/"),
    ("Tripo-AI",              "https://www.tripo3d.ai/"),
    ("Luma-Genie",            "https://lumalabs.ai/genie"),
    ("Spline",                "https://spline.design/ai"),
    ("Krea",                  "https://www.krea.ai/"),
    ("Magnific",              "https://magnific.ai/"),
    ("Topaz-Labs",            "https://www.topazlabs.com/"),
    ("Upscayl",               "https://upscayl.org/"),
]


def _media_gen_visits() -> list[Provider]:
    return [WebProbe(name, url, category="media_gen") for name, url in _MEDIA_GEN]


# ---------------------------------------------------------------------------
# Category: aggregator — platforms, marketplaces, dev tooling, agent builders
# ---------------------------------------------------------------------------

_AGGREGATOR_WEB: list[tuple[str, str]] = [
    # Model hubs / hosting
    ("HuggingFace",               "https://huggingface.co/"),
    ("HuggingFace-Spaces",        "https://huggingface.co/spaces"),
    ("HuggingFace-Models",        "https://huggingface.co/models"),
    ("HuggingFace-Datasets",      "https://huggingface.co/datasets"),
    ("HuggingFace-Papers",        "https://huggingface.co/papers"),
    ("Replicate",                 "https://replicate.com/"),
    ("Modal",                     "https://modal.com/"),
    ("Baseten",                   "https://www.baseten.co/"),
    ("RunPod",                    "https://www.runpod.io/"),
    ("Fal.ai",                    "https://fal.ai/"),
    ("Lambda-Labs",               "https://lambdalabs.com/"),
    ("CoreWeave",                 "https://www.coreweave.com/"),
    ("Anyscale",                  "https://www.anyscale.com/"),
    # Gateways / routing
    ("OpenRouter",                "https://openrouter.ai/"),
    ("Portkey",                   "https://portkey.ai/"),
    ("LiteLLM",                   "https://www.litellm.ai/"),
    ("Martian",                   "https://withmartian.com/"),
    # Dev / observability / evals
    ("LangSmith",                 "https://smith.langchain.com/"),
    ("LangChain",                 "https://www.langchain.com/"),
    ("LlamaIndex",                "https://www.llamaindex.ai/"),
    ("Langfuse",                  "https://langfuse.com/"),
    ("Weights-Biases",            "https://wandb.ai/"),
    ("Helicone",                  "https://www.helicone.ai/"),
    ("Arize",                     "https://arize.com/"),
    ("Braintrust",                "https://www.braintrust.dev/"),
    ("PromptLayer",               "https://promptlayer.com/"),
    # Directories / discovery
    ("TheresAnAIForThat",         "https://theresanaiforthat.com/"),
    ("Futurepedia",               "https://www.futurepedia.io/"),
    ("AITools.fyi",               "https://aitools.fyi/"),
    ("FutureTools",               "https://www.futuretools.io/"),
    # Coding assistants
    ("Cursor",                    "https://cursor.com/"),
    ("Windsurf",                  "https://windsurf.com/"),
    ("GitHub-Copilot",            "https://github.com/features/copilot"),
    ("Bolt.new",                  "https://bolt.new/"),
    ("Lovable",                   "https://lovable.dev/"),
    ("v0.dev",                    "https://v0.dev/"),
    ("Replit-AI",                 "https://replit.com/ai"),
    ("Tabnine",                   "https://www.tabnine.com/"),
    ("Codeium",                   "https://codeium.com/"),
    # Agent builders / workflow
    ("Flowise",                   "https://flowiseai.com/"),
    ("CrewAI",                    "https://www.crewai.com/"),
    ("AutoGPT",                   "https://agpt.co/"),
    ("AgentOps",                  "https://www.agentops.ai/"),
    ("Dify",                      "https://dify.ai/"),
    ("n8n",                       "https://n8n.io/"),
    # Vector DBs (frequently tagged as GenAI infra by SASE vendors)
    ("Pinecone",                  "https://www.pinecone.io/"),
    ("Weaviate",                  "https://weaviate.io/"),
    ("Chroma",                    "https://www.trychroma.com/"),
    ("Qdrant-Cloud",              "https://cloud.qdrant.io/"),
]


def _aggregator_api_probes() -> list[Provider]:
    def openrouter_body(p: str) -> dict[str, Any]:
        return {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": p}],
        }

    def hf_body(p: str) -> dict[str, Any]:
        return {"inputs": p, "parameters": {"max_new_tokens": 128}}

    def fal_body(p: str) -> dict[str, Any]:
        return {"prompt": p, "image_size": "square"}

    def replicate_predict_body(p: str) -> dict[str, Any]:
        return {
            "version": "5c7d5dc6dd8bf75c1acaa8565735e7986bc5b66206b55cca93cb72c9bf15ccaa",
            "input": {"prompt": p},
        }

    return [
        ApiProbe(
            name="OpenRouter-API",
            url="https://openrouter.ai/api/v1/chat/completions",
            user_agent="openrouter-client/1.0",
            body_builder=openrouter_body,
            category="aggregator",
        ),
        ApiProbe(
            name="HuggingFace-Inference-API",
            url=("https://api-inference.huggingface.co/models/"
                 "meta-llama/Meta-Llama-3-8B-Instruct"),
            user_agent="huggingface-hub/0.26.2 python/3.12",
            body_builder=hf_body,
            category="aggregator",
        ),
        ApiProbe(
            name="Replicate-API",
            url="https://api.replicate.com/v1/predictions",
            user_agent="replicate-python/1.0.3",
            body_builder=replicate_predict_body,
            category="aggregator",
        ),
        ApiProbe(
            name="Fal.ai-API",
            url="https://fal.run/fal-ai/flux/schnell",
            user_agent="fal-client/0.5.0 python/3.12",
            body_builder=fal_body,
            category="aggregator",
        ),
        ApiProbe(
            name="LangSmith-API",
            url="https://api.smith.langchain.com/info",
            user_agent="langsmith-py/0.1.140",
            method="GET",
            body_builder=None,
            category="aggregator",
        ),
        ApiProbe(
            name="Portkey-API",
            url="https://api.portkey.ai/v1/chat/completions",
            user_agent="portkey-ai/1.5.0",
            body_builder=openrouter_body,
            category="aggregator",
        ),
    ]


def _aggregator_targets() -> list[Provider]:
    return [
        WebProbe(name, url, category="aggregator") for name, url in _AGGREGATOR_WEB
    ] + _aggregator_api_probes()


# ---------------------------------------------------------------------------
# Category: real_response — keyless providers that return actual LLM output
# ---------------------------------------------------------------------------

def _real_response_providers() -> list[Provider]:
    return [
        PollinationsText(),
        PollinationsImage(),
        DuckDuckGoChat(),
    ]


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Category: mcp_synthetic — Model Context Protocol traffic (slice A)
# ---------------------------------------------------------------------------
#
# Generates well-formed MCP JSON-RPC payloads against two destination
# classes:
#
#   * Reflectors (httpbin.org, postman-echo.com) — these don't pretend
#     to be MCP. The probe tests whether SASE classifies based on
#     payload shape alone.
#
#   * Real public MCP server hostnames, but unauthenticated. The
#     server returns 401/403 immediately; we don't care about a
#     successful handshake. What we care about is that the
#     destination + payload combination produces a classifiable
#     signal in the inspecting fabric.
#
# Both transports (legacy HTTP+SSE and current Streamable HTTP) are
# represented so SASE coverage of the two wire shapes is testable.
#
# The probes use the existing ApiProbe class — no new probe type
# needed. Just feed it the MCP-shaped body builder and the right
# Accept header for each transport.

def _mcp_synthetic_probes() -> list[Provider]:
    """MCP probes spanning methods, transports, and destinations."""
    probes: list[Provider] = []

    # Realistic UA strings — these match what current MCP SDKs emit.
    UA_PYTHON_SDK = "mcp-python-sdk/1.2.0"
    UA_TS_SDK     = "@modelcontextprotocol/sdk/1.0.4"
    UA_INSPECTOR  = "modelcontextprotocol-inspector/0.4.0"

    def _streamable_headers() -> dict[str, str]:
        # New header set per call so MCP-Protocol-Version rotates and
        # Mcp-Session-Id gets a fresh UUID each fire.
        return mcp.headers_for(
            "streamable",
            session_id=mcp.synthetic_session_id(),
        )

    # ---- (1) Reflector targets — payload-shape classification ----
    # Each method gets one probe against the canonical reflector so
    # SASE can classify on body shape with no destination hint.

    method_builders = [
        ("initialize",                mcp.build_initialize_request),
        ("tools/list",                mcp.build_tools_list_request),
        ("tools/call",                mcp.random_tool_call_body),
        ("resources/read",            lambda: mcp.build_resources_read_request("file:///etc/hosts")),
        ("prompts/get",               lambda: mcp.build_prompts_get_request("summarize")),
        ("notifications/initialized", mcp.build_initialized_notification),
        # New in Stage 3: extended method coverage. Each adds a
        # distinct on-the-wire fingerprint a classifier might key on.
        ("ping",                       mcp.build_ping_request),
        ("roots/list",                 mcp.build_roots_list_request),
        ("logging/setLevel",           mcp.build_logging_set_level_request),
        ("completion/complete",        mcp.build_completion_complete_request),
        ("sampling/createMessage",     lambda: mcp.build_sampling_create_message_request("Summarize this.")),
        ("elicitation/elicit",         mcp.build_elicitation_elicit_request),
        ("notifications/progress",     mcp.build_progress_notification),
        ("notifications/cancelled",    mcp.build_cancelled_notification),
        ("notifications/resources/updated", mcp.build_resources_updated_notification),
        ("notifications/tools/list_changed", mcp.build_tools_list_changed_notification),
    ]

    for method, builder in method_builders:
        probes.append(ApiProbe(
            name=f"MCP synthetic ({method}) → reflector",
            url="https://httpbin.org/post",
            user_agent=UA_PYTHON_SDK,
            category="mcp_synthetic",
            body_builder=lambda _p, b=builder: b(),
            extra_headers=_streamable_headers(),
            send_fake_auth=False,
        ))

    # Add the same tools/call shape on the legacy SSE transport so
    # SASE coverage of both wire conventions is exercised.
    probes.append(ApiProbe(
        name="MCP synthetic (tools/call, sse-legacy) → reflector",
        url="https://postman-echo.com/post",
        user_agent=UA_TS_SDK,
        category="mcp_synthetic",
        body_builder=lambda _p: mcp.random_tool_call_body(),
        extra_headers=mcp.headers_for("sse-legacy"),
        send_fake_auth=False,
    ))

    # JSON-RPC batched request (2025-03-26 spec window). Some clients
    # pinned to that version still emit batches; the shape is
    # distinctive (top-level array).
    probes.append(ApiProbe(
        name="MCP synthetic (batch: init + tools/list + tools/call)",
        url="https://httpbin.org/post",
        user_agent=UA_TS_SDK,
        category="mcp_synthetic",
        body_builder=lambda _p: mcp.build_batch(
            mcp.build_initialize_request(protocol_version="2025-03-26"),
            mcp.build_tools_list_request(),
            mcp.random_tool_call_body(),
        ),
        extra_headers=mcp.headers_for("streamable", protocol_version="2025-03-26"),
        send_fake_auth=False,
    ))

    # ---- (2) Real public MCP server hostnames, unauthenticated ----
    # Derive directly from PUBLIC_MCP_SERVERS so adding/removing a
    # destination in mcp.py automatically flows through to the
    # registry without a second list to maintain.

    def _ua_for(host: str) -> str:
        # Inspector and TS SDK alternation gives the classifier
        # variety without overweighting any one UA.
        if "cloudflare" in host or "github" in host:
            return UA_PYTHON_SDK
        if "atlassian" in host or "deepwiki" in host:
            return UA_INSPECTOR
        return UA_TS_SDK

    for srv in mcp.PUBLIC_MCP_SERVERS:
        if srv.get("streamable_url"):
            probes.append(ApiProbe(
                name=srv["name"],
                url=srv["streamable_url"],
                user_agent=_ua_for(srv["host"]),
                category="mcp_synthetic",
                # initialize is the universal first-message — what every
                # MCP client sends on session start. Most likely to
                # produce a classifiable signature, even if the server
                # immediately 401s for missing auth.
                body_builder=lambda _p: mcp.build_initialize_request(),
                extra_headers=_streamable_headers(),
                send_fake_auth=False,
            ))

    return probes


def build_registry(
    categories: set[str] | None = None,
    key_store: Any | None = None,
) -> list[Provider]:
    """Return the flat list of providers.

    With ``categories=None`` (the default used at runtime), every provider
    is returned so the live UI can re-enable a category at any time. Pass
    an explicit ``categories`` set to pre-filter (useful for tests).

    ``key_store``: if provided, MCPAuthedProbe instances are constructed
    for each entry in mcp.MCP_KEYED_SERVERS — these are slice B-static,
    real authed traffic against MCP servers that accept token auth.
    Without a key_store the authed probes are omitted (so test code
    that only cares about the deterministic registry shape doesn't
    have to wire one up). The probes themselves fetch the saved token
    at execute() time, so saving/refreshing/deleting a key takes
    effect on the next fire without registry rebuild.
    """
    all_providers: list[Provider] = []
    all_providers += _llm_api_probes()
    all_providers += _chatbot_ui_visits()
    all_providers += _media_gen_visits()
    all_providers += _aggregator_targets()
    all_providers += _real_response_providers()
    all_providers += _mcp_synthetic_probes()
    if key_store is not None:
        for server in mcp.MCP_KEYED_SERVERS:
            all_providers.append(
                MCPAuthedProbe(server=server, key_provider=key_store.mcp_get)
            )
    if categories is None:
        return all_providers
    return [p for p in all_providers if p.category in categories]
