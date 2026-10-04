/**
 * LESSON 07 — Memory: checkpointers, threads, and middleware
 *
 * CONCEPT: lesson 02 showed that a conversation is an array you resend. A
 * checkpointer automates that: it persists agent state keyed by `thread_id`,
 * loads it before each turn, saves it after. Your app sends one new message;
 * the agent sees the whole history.
 *
 * Swap MemorySaver for PostgresSaver and the exact same code is production
 * multi-user chat. That is the entire design.
 *
 * Run: npm run 07
 */

import * as z from "zod";
import { createAgent, tool, summarizationMiddleware, createMiddleware } from "langchain";
import { MemorySaver } from "@langchain/langgraph";
import { isAIMessage } from "@langchain/core/messages";
import { section, getModel } from "./model.js";

const model = await getModel({ temperature: 0 });

const noteTool = tool(({ note }) => `Noted: ${note}`, {
  name: "save_note",
  description: "Save a short note for the user.",
  schema: z.object({ note: z.string() }),
});

section("07 · Without a checkpointer: total amnesia");

const forgetful = createAgent({ model, tools: [] });
await forgetful.invoke({ messages: [{ role: "user", content: "My name is Ingrid." }] });
const amnesia = await forgetful.invoke({
  messages: [{ role: "user", content: "What's my name?" }],
});
console.log("->", amnesia.messages.at(-1)?.text);

section("With a checkpointer: state persists per thread");

// MemorySaver = in-process Map. Gone on restart. For real apps use
// PostgresSaver / SqliteSaver / RedisSaver — identical interface.
const checkpointer = new MemorySaver();

const remembering = createAgent({
  model,
  tools: [noteTool],
  systemPrompt: "You are a concise personal assistant.",
  checkpointer,
});

// `thread_id` is the conversation key. One per user chat, one per session —
// whatever your app's unit of conversation is.
const alice = { configurable: { thread_id: "alice" } };

await remembering.invoke({ messages: [{ role: "user", content: "My name is Ingrid and I ride a red bicycle." }] }, alice);
const recalled = await remembering.invoke({ messages: [{ role: "user", content: "What colour is my bicycle?" }] }, alice);
console.log("alice thread ->", recalled.messages.at(-1)?.text);

section("Threads are isolated");

const bob = { configurable: { thread_id: "bob" } };
const bobReply = await remembering.invoke(
  { messages: [{ role: "user", content: "What colour is my bicycle?" }] },
  bob
);
console.log("bob thread   ->", bobReply.messages.at(-1)?.text);

section("Inspecting stored state");

// The checkpointer is queryable. `agent.graph` exposes the compiled LangGraph
// underneath the agent — that is where getState/getStateHistory live.
// (`agent.getState` exists but is marked @internal and typed `never`.)
const snapshot = await remembering.graph.getState(alice);
console.log(`alice has ${snapshot.values.messages.length} stored messages`);
console.log("next node to run:", snapshot.next.length ? snapshot.next : "(idle)");

// Every checkpoint is retained — this is the basis of time travel and undo.
let checkpoints = 0;
for await (const _ of remembering.graph.getStateHistory(alice)) checkpoints++;
console.log(`alice has ${checkpoints} checkpoints in history`);

section("The context-window problem");
console.log(`
  Unbounded history eventually exceeds the context window, and you pay for
  every token every turn. Two built-in answers:
`);

// (1) Summarization: when history crosses a threshold, older messages are
// replaced by a model-written summary. Bounded cost, long effective memory.
const summarizing = createAgent({
  model,
  tools: [],
  middleware: [
    summarizationMiddleware({
      model,
      trigger: { messages: 6 },  // summarize once history exceeds 6 messages
      keep: { messages: 2 },     // always keep the 2 most recent verbatim
    }),
  ],
  checkpointer: new MemorySaver(),
});

const longThread = { configurable: { thread_id: "long" } };
const facts = [
  "I live in Porto.",
  "I am allergic to shellfish.",
  "My sister's name is Mira.",
  "I work as a cartographer.",
  "I don't drink coffee.",
  "My favourite season is autumn.",
];
for (const f of facts) {
  await summarizing.invoke({ messages: [{ role: "user", content: `${f} Reply with just "ok".` }] }, longThread);
}

const afterSummary = await summarizing.invoke(
  { messages: [{ role: "user", content: "What am I allergic to, and where do I live?" }] },
  longThread
);
const kept = (await summarizing.graph.getState(longThread)).values.messages.length;
console.log(`after ${facts.length + 1} turns, history compressed to ${kept} messages`);
console.log("still recalls ->", afterSummary.messages.at(-1)?.text);

section("Middleware: your own hook in the loop");

// (2) Middleware intercepts the agent loop. beforeModel/afterModel run per
// model call; wrapModelCall/wrapToolCall wrap them. This is where you put
// logging, retries, guardrails, dynamic prompts, token budgets.
const auditLog = createMiddleware({
  name: "AuditLog",
  beforeModel: (state) => {
    console.log(`  [audit] calling model with ${state.messages.length} messages`);
    return;
  },
  afterModel: (state) => {
    const last = state.messages.at(-1);
    const calls = last && isAIMessage(last) ? last.tool_calls : undefined;
    console.log(`  [audit] model returned ${calls?.length ? "tool calls" : "text"}`);
    return;
  },
});

const audited = createAgent({ model, tools: [noteTool], middleware: [auditLog] });
await audited.invoke({ messages: [{ role: "user", content: "Save a note: buy chain oil." }] });

section("Takeaway");
console.log(`
  checkpointer: new MemorySaver()                  // swap for Postgres in prod
  agent.invoke(input, { configurable: { thread_id } })
  agent.graph.getState(cfg) / getStateHistory(cfg) // inspect + time travel

  summarizationMiddleware({ model, trigger, keep })  // bounded context
  createMiddleware({ beforeModel, afterModel, wrapModelCall, wrapToolCall })

  Memory is not a feature of the model. It is state you persist and replay.

  Next: npm run 08
`);
