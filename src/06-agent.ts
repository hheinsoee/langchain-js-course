/**
 * LESSON 06 — createAgent
 *
 * CONCEPT: lesson 05's hand-written loop, productionised. `createAgent` gives
 * you the same loop plus: recursion limits, parallel tool execution, error
 * handling, streaming, checkpointing, structured output, and middleware hooks.
 *
 * Under the hood it compiles to a LangGraph state machine (lesson 09), which is
 * why `.invoke()` takes `{ messages }` and returns a STATE object, not a message.
 *
 * Run: npm run 06
 */

import * as z from "zod";
import { createAgent, tool } from "langchain";
import type { BaseMessage } from "@langchain/core/messages";
import { isAIMessage } from "@langchain/core/messages";
import { section, getModel } from "./model.js";

const model = await getModel({ temperature: 0 });

const searchFlights = tool(
  ({ from, to }) => {
    const price = 120 + ((from.length * to.length * 37) % 400);
    return JSON.stringify([
      { airline: "Sky", from, to, price, stops: 0 },
      { airline: "Budget", from, to, price: price - 45, stops: 1 },
    ]);
  },
  {
    name: "search_flights",
    description: "Search available flights between two cities.",
    schema: z.object({
      from: z.string().describe("Departure city"),
      to: z.string().describe("Destination city"),
    }),
  }
);

const getBudget = tool(() => "Remaining travel budget: 300 USD.", {
  name: "get_budget",
  description: "Get the traveller's remaining budget in USD.",
  schema: z.object({}),
});

section("06 · The whole of lesson 05, in one call");

const agent = createAgent({
  model,
  tools: [searchFlights, getBudget],
  systemPrompt:
    "You are a travel assistant. Always check the budget before recommending " +
    "a flight. Be concise.",
});

// Input is `{ messages }`. Output is the full agent STATE, whose `messages`
// array contains every intermediate step: AIMessage(tool_calls),
// ToolMessage(result), ..., final AIMessage.
const result = await agent.invoke({
  messages: [{ role: "user", content: "Can I afford to fly from Oslo to Lisbon?" }],
});

console.log("\nFull trace:");
for (const m of result.messages) {
  const kind = m.getType();
  // `isAIMessage` is a type guard: inside the branch, `m.tool_calls` exists.
  if (isAIMessage(m) && m.tool_calls?.length) {
    console.log(`  [ai]    calls: ${m.tool_calls.map((c) => c.name).join(", ")}`);
  } else if (kind === "tool") {
    console.log(`  [tool]  ${String(m.content).slice(0, 70)}`);
  } else {
    console.log(`  [${kind}]${" ".repeat(Math.max(0, 6 - kind.length))}  ${m.text}`);
  }
}

console.log("\nFinal answer:", result.messages.at(-1)?.text);

section("Agent + structured output");

// `responseFormat` constrains the FINAL answer while leaving the tool loop
// free. The parsed object lands on `result.structuredResponse`.
const Recommendation = z.object({
  affordable: z.boolean(),
  airline: z.string(),
  priceUsd: z.number(),
  reasoning: z.string().describe("One sentence"),
});

const structuredAgent = createAgent({
  model,
  tools: [searchFlights, getBudget],
  systemPrompt: "You are a travel assistant. Check the budget before recommending.",
  responseFormat: Recommendation,
});

const structured = await structuredAgent.invoke({
  messages: [{ role: "user", content: "Should I book Oslo to Lisbon?" }],
});
console.log(structured.structuredResponse);

section("Streaming an agent's steps");

// `streamMode: "updates"` yields one object per completed node, keyed by node
// name. This is what you render as "Searching flights..." in a UI.
const stream = await agent.stream(
  { messages: [{ role: "user", content: "Find me a flight from Tokyo to Seoul." }] },
  { streamMode: "updates" }
);

// Each update is keyed by the node that just ran ("model", "tools", ...).
type NodeUpdate = Record<string, { messages?: BaseMessage[] } | undefined>;

for await (const update of stream) {
  for (const [node, payload] of Object.entries(update as NodeUpdate)) {
    const last = payload?.messages?.at(-1);
    if (!last) continue;
    const calls = isAIMessage(last) ? last.tool_calls : undefined;
    if (calls?.length)
      console.log(`  ${node}: requesting ${calls.map((c) => c.name).join(", ")}`);
    else console.log(`  ${node}: ${String(last.content).slice(0, 80)}`);
  }
}

section("Guard rails you get for free");

// Agents can loop forever: a model that keeps requesting tools never reaches a
// final answer. `recursionLimit` caps total graph steps and throws when hit.
const loopy = createAgent({ model, tools: [getBudget] });
try {
  const out = await loopy.invoke(
    { messages: [{ role: "user", content: "Check the budget 50 times in a row." }] },
    { recursionLimit: 4 }
  );
  console.log(
    `finished inside the limit (${out.messages.length} messages) — ` +
      `this model refused to loop, which is also a valid outcome`
  );
} catch (err) {
  console.log("recursionLimit tripped as designed:", (err as Error).name);
}

section("Takeaway");
console.log(`
  createAgent({ model, tools, systemPrompt, responseFormat, checkpointer,
                middleware, contextSchema })

  agent.invoke({ messages })      -> { messages: [...full trace], structuredResponse? }
  agent.stream(input, { streamMode: "updates" | "messages" | "custom" })

  - result.messages holds EVERY step, not just the answer. Great for debugging.
  - responseFormat gives a validated final object without blocking tool use.
  - recursionLimit stops runaway loops.

  Next: npm run 07
`);
