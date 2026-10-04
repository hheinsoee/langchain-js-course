/**
 * Shared model setup for every lesson.
 *
 * WHY THIS FILE EXISTS
 * LangChain's whole selling point is that your application code does not care
 * which provider you use. Every chat model implements the same interface:
 * `.invoke()`, `.stream()`, `.bindTools()`, `.withStructuredOutput()`.
 *
 * So we resolve the provider ONCE here, and every lesson imports from here.
 * Change `.env` and all 10 lessons run against a different model without a
 * single code change. That is the lesson.
 */

import { initChatModel } from "langchain";
import { OpenAIEmbeddings } from "@langchain/openai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { OllamaEmbeddings } from "@langchain/ollama";
import type { ConfigurableModel } from "langchain/chat_models/universal";
import type { Embeddings } from "@langchain/core/embeddings";

type Provider = "openai" | "anthropic" | "google-genai" | "ollama";

/**
 * Any OpenAI-compatible gateway (Sargalay, OpenRouter, Together, LM Studio,
 * vLLM...) is reached by pointing the OpenAI client at a different base URL.
 * The `openai` provider class then works unchanged — that is what "OpenAI
 * compatible" buys you.
 */
const BASE_URL = process.env.OPENAI_BASE_URL;

const DEFAULT_CHAT: Record<Provider, string> = {
  openai: BASE_URL ? "google/gemini-2.5-flash-lite" : "gpt-4o-mini",
  anthropic: "claude-3-5-haiku-latest",
  "google-genai": "gemini-2.0-flash",
  ollama: process.env.OLLAMA_MODEL ?? "qwen3:8b",
};

const DEFAULT_EMBED: Record<Provider, string> = {
  openai: "text-embedding-3-small",
  // Anthropic ships no embedding model; lesson 08 falls back to another one.
  anthropic: "",
  "google-genai": "text-embedding-004",
  ollama: process.env.OLLAMA_EMBED_MODEL ?? "nomic-embed-text",
};

function detectProvider(): Provider {
  const forced = process.env.LC_PROVIDER as Provider | undefined;
  if (forced) return forced;
  if (process.env.OPENAI_API_KEY) return "openai";
  if (process.env.ANTHROPIC_API_KEY) return "anthropic";
  if (process.env.GOOGLE_API_KEY) return "google-genai";
  if (process.env.OLLAMA_MODEL) return "ollama";

  throw new Error(
    [
      "",
      "No model provider configured.",
      "",
      "  cp .env.example .env",
      "",
      "then set ONE of:",
      "  OPENAI_API_KEY=...  (+ OPENAI_BASE_URL for an OpenAI-compatible gateway)",
      "  ANTHROPIC_API_KEY=sk-ant-... https://console.anthropic.com/settings/keys",
      "  GOOGLE_API_KEY=...           https://aistudio.google.com/apikey",
      "  OLLAMA_MODEL=qwen3:8b        local, no key: `ollama serve`",
      "",
    ].join("\n")
  );
}

export const PROVIDER: Provider = detectProvider();
export const MODEL_NAME = process.env.LC_MODEL ?? DEFAULT_CHAT[PROVIDER];

/**
 * The "provider:model" identifier string. `createAgent({ model })` and
 * `initChatModel()` both accept this form directly — handy, but it hides which
 * class you got and takes no parameters. Use `getModel()` when you need either.
 */
export const MODEL_ID = `${PROVIDER}:${MODEL_NAME}`;

/** Extra constructor args forwarded to the provider class. */
const providerParams =
  PROVIDER === "openai" && BASE_URL ? { configuration: { baseURL: BASE_URL } } : {};

/**
 * `initChatModel` dynamically loads the right provider package and returns a
 * `BaseChatModel`. Everything downstream is provider-agnostic from here on.
 *
 * `maxRetries` matters on shared/free model pools: a 429 there means "the
 * upstream pool is busy", not "you exceeded your quota". LangChain retries
 * with exponential backoff automatically — raise the ceiling rather than
 * writing your own retry loop.
 */
export async function getModel(
  params: Record<string, unknown> = {}
): Promise<ConfigurableModel> {
  return initChatModel(MODEL_NAME, {
    modelProvider: PROVIDER,
    maxRetries: 8,
    ...providerParams,
    ...params,
  });
}

/**
 * Embedding models turn text into vectors. Used only by lesson 08 (RAG).
 * Anthropic has no embedding API, so we fall back to whatever else is present.
 */
export function getEmbeddings(): Embeddings {
  let provider = PROVIDER;
  if (!DEFAULT_EMBED[provider]) {
    if (process.env.OPENAI_API_KEY) provider = "openai";
    else if (process.env.GOOGLE_API_KEY) provider = "google-genai";
    else if (process.env.OLLAMA_MODEL) provider = "ollama";
    else
      throw new Error(
        `${PROVIDER} has no embedding model. For lesson 08, also set OPENAI_API_KEY, ` +
          `GOOGLE_API_KEY, or run Ollama (OLLAMA_MODEL=... + \`ollama pull nomic-embed-text\`).`
      );
  }
  const model = process.env.LC_EMBED_MODEL ?? DEFAULT_EMBED[provider];

  switch (provider) {
    case "openai":
      return new OpenAIEmbeddings({
        model,
        ...(BASE_URL ? { configuration: { baseURL: BASE_URL } } : {}),
      });
    case "google-genai":
      return new GoogleGenerativeAIEmbeddings({ model });
    case "ollama":
      return new OllamaEmbeddings({ model });
    default:
      throw new Error(`No embeddings for provider ${provider}`);
  }
}

/** Pretty section header so lesson output is readable. */
export function section(title: string): void {
  console.log(`\n\x1b[1m\x1b[36m${"─".repeat(60)}\n${title}\n${"─".repeat(60)}\x1b[0m`);
}
