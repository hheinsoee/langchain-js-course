/**
 * LESSON 04 — Structured output
 *
 * CONCEPT: "respond with JSON" in a prompt is a request, not a guarantee.
 * You get markdown fences, prose preambles, trailing commas, hallucinated
 * fields. `.withStructuredOutput(schema)` makes it a constraint instead:
 * LangChain converts your Zod schema to the provider's native JSON-schema or
 * tool-calling mechanism, then parses and validates the result for you.
 *
 * This is the single highest-leverage feature for putting an LLM in a pipeline.
 *
 * Run: npm run 04
 */

import * as z from "zod";
import { isAIMessage } from "@langchain/core/messages";
import { section, getModel } from "./model.js";

const model = await getModel({ temperature: 0 });

section("04 · The problem");

const unstructured = await model.invoke(
  'Extract name and age as JSON: "Dr. Ada Lovelace, 36, mathematician."'
);
console.log("raw string you'd have to parse yourself:\n", JSON.stringify(unstructured.text));

section("The fix: .withStructuredOutput(zodSchema)");

// `.describe()` is not decoration — it is sent to the model as the field
// description. It is your per-field prompt. Use it.
const Person = z.object({
  name: z.string().describe("Full name, including any title"),
  age: z.number().int().describe("Age in years"),
  occupation: z.string().describe("Their profession"),
});

const extractor = model.withStructuredOutput(Person);

// Returns the PARSED OBJECT directly. Not an AIMessage. Already validated.
const person = await extractor.invoke(
  'Extract from: "Dr. Ada Lovelace, 36, mathematician."'
);
console.log(person);
console.log("typeof person.age ->", typeof person.age); // number, not string

section("Nested + optional + enum");

const Ticket = z.object({
  title: z.string(),
  severity: z.enum(["low", "medium", "high", "critical"]),
  // .nullable() lets the model say "not stated" instead of inventing a value.
  affectedVersion: z.string().nullable().describe("Version, or null if unstated"),
  tags: z.array(z.string()).describe("2-4 short lowercase tags"),
  reporter: z.object({
    name: z.string(),
    email: z.string().nullable(),
  }),
});

const triage = model.withStructuredOutput(Ticket);
const ticket = await triage.invoke(`Classify this bug report:

  "Hi, Priya here (priya@acme.io). The checkout page returns a 500 whenever
   I apply a discount code. Every customer is blocked. This is urgent."`);

console.log(JSON.stringify(ticket, null, 2));

section("Keeping the raw message too");

// `includeRaw: true` gives you { raw: AIMessage, parsed: T }. Use it when you
// need token counts or the model's prose alongside the structured data.
const withRaw = model.withStructuredOutput(Person, { includeRaw: true });
const both = await withRaw.invoke('Extract: "Grace Hopper, 45, rear admiral."');
console.log("parsed:", both.parsed);
// `raw` is a BaseMessage; narrow it to read AI-specific fields.
if (isAIMessage(both.raw)) console.log("raw tokens:", both.raw.usage_metadata);

section("Classification: the workhorse pattern");

const Sentiment = z.object({
  sentiment: z.enum(["positive", "neutral", "negative"]),
  confidence: z.number().min(0).max(1),
  reason: z.string().describe("At most 10 words"),
});

const classify = model.withStructuredOutput(Sentiment);

const reviews = [
  "Shipped two weeks late and the box was crushed.",
  "Does exactly what it says. No complaints.",
  "Genuinely the best purchase I've made all year.",
];

// `.batch()` works on the structured chain too — it is still a Runnable.
for (const [i, r] of (await classify.batch(reviews)).entries()) {
  console.log(`${reviews[i].slice(0, 40).padEnd(42)} -> ${r.sentiment} (${r.confidence})`);
}

section("Takeaway");
console.log(`
  model.withStructuredOutput(schema)        -> returns parsed object
  model.withStructuredOutput(s, {includeRaw:true}) -> { raw, parsed }

  - .describe() on every field. It is prompt text, not a comment.
  - .nullable() for "might not be present" - prevents invented values.
  - z.enum() for classification - the model cannot return anything else.
  - Schema validation failures throw, so bad output is loud, not silent.

  Next: npm run 05
`);
