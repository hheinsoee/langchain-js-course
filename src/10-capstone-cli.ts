/**
 * LESSON 10 — Capstone: an interactive agent you can actually talk to
 *
 * Everything from lessons 01-09, assembled:
 *   02 messages      - the conversation array
 *   03 streaming     - tokens appear as they're generated
 *   05 tools         - four real tools, including one that touches the FS
 *   06 createAgent   - the loop
 *   07 memory        - checkpointer + thread_id, plus summarization
 *   08 RAG           - a searchable notes file
 *   09 middleware    - a logger showing each step
 *
 * Commands:  /new  /state  /tools  /exit
 *
 * Run: npm run chat
 */

import { createInterface } from "node:readline/promises";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import * as z from "zod";
import {
  createAgent,
  tool,
  createMiddleware,
  summarizationMiddleware,
} from "langchain";
import { MemorySaver } from "@langchain/langgraph";
import { isAIMessage } from "@langchain/core/messages";
import { getModel, MODEL_ID } from "./model.js";

const NOTES_DIR = join(process.cwd(), ".notes");
const NOTES_FILE = join(NOTES_DIR, "notes.md");

const model = await getModel({ temperature: 0 });

// ── Tools ──────────────────────────────────────────────────────────────────

const saveNote = tool(
  async ({ text }) => {
    await mkdir(NOTES_DIR, { recursive: true });
    const line = `- [${new Date().toISOString().slice(0, 16).replace("T", " ")}] ${text}\n`;
    await writeFile(NOTES_FILE, line, { flag: "a" });
    return `Saved: "${text}"`;
  },
  {
    name: "save_note",
    description: "Persist a note to disk. Use when the user asks to remember or jot something down.",
    schema: z.object({ text: z.string().describe("The note, one line") }),
  }
);

const searchNotes = tool(
  async ({ query }) => {
    const body = await readFile(NOTES_FILE, "utf8").catch(() => "");
    if (!body) return "No notes saved yet.";
    const terms = query.toLowerCase().split(/\s+/).filter((t) => t.length > 2);
    const scored = body
      .split("\n")
      .filter(Boolean)
      .map((line) => ({ line, hits: terms.filter((t) => line.toLowerCase().includes(t)).length }))
      .filter((r) => r.hits > 0)
      .sort((a, b) => b.hits - a.hits)
      .slice(0, 5);
    return scored.length ? scored.map((r) => r.line).join("\n") : `No notes matching "${query}".`;
  },
  {
    name: "search_notes",
    description: "Keyword search over previously saved notes. Call before claiming a note does not exist.",
    schema: z.object({ query: z.string().describe("Keywords to look for") }),
  }
);

const calculate = tool(
  ({ expression }) => {
    if (!/^[\d\s+\-*/().%]+$/.test(expression)) return "Rejected: arithmetic characters only.";
    try {
      return String(Function(`"use strict";return (${expression})`)());
    } catch {
      return "Could not evaluate that expression.";
    }
  },
  {
    name: "calculate",
    description: "Evaluate an arithmetic expression. Use for ALL arithmetic, never do it yourself.",
    schema: z.object({ expression: z.string().describe("e.g. '(1200 * 0.19) + 45'") }),
  }
);

const now = tool(
  () => new Date().toString(),
  {
    name: "current_datetime",
    description: "Get the current date and time. Use for anything relative like 'today' or 'next week'.",
    schema: z.object({}),
  }
);

const tools = [saveNote, searchNotes, calculate, now];

// ── Middleware: show the agent's work ──────────────────────────────────────

let verbose = true;

const stepLogger = createMiddleware({
  name: "StepLogger",
  afterModel: (state) => {
    if (!verbose) return;
    const last = state.messages.at(-1);
    const calls = last && isAIMessage(last) ? last.tool_calls : undefined;
    if (calls?.length) {
      for (const c of calls) {
        console.log(`\x1b[2m  · ${c.name}(${JSON.stringify(c.args)})\x1b[0m`);
      }
    }
    return;
  },
});

// ── The agent ──────────────────────────────────────────────────────────────

const checkpointer = new MemorySaver();

const agent = createAgent({
  model,
  tools,
  systemPrompt: [
    "You are a sharp, concise personal assistant running in a terminal.",
    "Use tools rather than guessing: calculate for arithmetic, current_datetime",
    "for anything date-relative, search_notes before saying a note does not exist.",
    "Keep answers short. No markdown headers, no bullet-point walls.",
  ].join(" "),
  middleware: [
    stepLogger,
    // Keeps the context window bounded no matter how long the chat runs.
    summarizationMiddleware({ model, trigger: { messages: 24 }, keep: { messages: 8 } }),
  ],
  checkpointer,
});

// ── REPL ───────────────────────────────────────────────────────────────────

let threadId = crypto.randomUUID();
const rl = createInterface({ input: process.stdin, output: process.stdout });

console.log(`\x1b[1m\x1b[36m
┌──────────────────────────────────────────────────────────┐
│  LangChain course · capstone agent                       │
└──────────────────────────────────────────────────────────┘\x1b[0m
  model   ${MODEL_ID}
  tools   ${tools.map((t) => t.name).join(", ")}
  memory  MemorySaver + summarization @ 24 messages

  /new     start a fresh thread (wipes context)
  /state   inspect stored messages
  /tools   list tools
  /quiet   toggle step logging
  /exit    quit
`);

for (;;) {
  const input = (await rl.question("\x1b[1m\x1b[32myou ›\x1b[0m ")).trim();
  if (!input) continue;

  if (input === "/exit") break;

  if (input === "/new") {
    threadId = crypto.randomUUID();
    console.log(`\x1b[2m  new thread ${threadId.slice(0, 8)}\x1b[0m\n`);
    continue;
  }

  if (input === "/quiet") {
    verbose = !verbose;
    console.log(`\x1b[2m  step logging ${verbose ? "on" : "off"}\x1b[0m\n`);
    continue;
  }

  if (input === "/tools") {
    for (const t of tools) console.log(`  \x1b[1m${t.name}\x1b[0m — ${t.description}`);
    console.log();
    continue;
  }

  if (input === "/state") {
    const snap = await agent.graph.getState({ configurable: { thread_id: threadId } });
    const msgs = snap.values.messages ?? [];
    console.log(`\x1b[2m  thread ${threadId.slice(0, 8)} · ${msgs.length} messages\x1b[0m`);
    for (const m of msgs.slice(-8)) {
      const body = String(m.text || m.content).replace(/\n/g, " ").slice(0, 70);
      console.log(`\x1b[2m    ${m.getType().padEnd(6)} ${body}\x1b[0m`);
    }
    console.log();
    continue;
  }

  // streamMode "messages" yields [token, metadata] tuples from EVERY LLM call
  // in the graph — including the summarization middleware's internal calls.
  // `langgraph_node` tells you which node produced the token; the agent's own
  // model node is named "model_request".
  process.stdout.write("\x1b[1m\x1b[35mbot ›\x1b[0m ");
  const stream = await agent.stream(
    { messages: [{ role: "user", content: input }] },
    { configurable: { thread_id: threadId }, streamMode: "messages" }
  );

  let printed = false;
  for await (const [token, metadata] of stream as AsyncIterable<
    [{ text: string }, { langgraph_node?: string }]
  >) {
    if (metadata.langgraph_node !== "model_request") continue;
    if (token.text) {
      process.stdout.write(token.text);
      printed = true;
    }
  }
  if (!printed) process.stdout.write("\x1b[2m(no text output)\x1b[0m");
  console.log("\n");
}

rl.close();
console.log("\n  Course complete. The .notes/ directory holds anything you saved.\n");
