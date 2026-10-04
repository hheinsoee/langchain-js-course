/**
 * LESSON 01 — Your first model call
 *
 * CONCEPT: the chat model is the only irreducible primitive in LangChain.
 * Everything else in this course (agents, RAG, graphs) is scaffolding around
 * this one call.
 *
 * A chat model is a function: messages in -> one message out.
 *
 * Run: npm run 01
 */

import { section, getModel, MODEL_ID } from "./model.js";

const model = await getModel();

section(`01 · First call  [${MODEL_ID}]`);

// `.invoke()` accepts a bare string as a shortcut. LangChain wraps it into a
// HumanMessage for you.
const response = await model.invoke("In one sentence: what is LangChain?");

// WHAT YOU GET BACK is an AIMessage, not a string. Three fields matter:
//
//   .text     -> always a string. Use this 95% of the time.
//   .content  -> the RAW content. May be a string OR an array of content
//                blocks (text, reasoning, images...). Loosely typed on purpose.
//   .contentBlocks -> normalized typed blocks, same shape across providers.
console.log("\n.text:\n", response.text);

console.log("\n.contentBlocks:\n", JSON.stringify(response.contentBlocks, null, 2));

// Token accounting is standardized across every provider.
console.log("\n.usage_metadata:\n", response.usage_metadata);

section("Parameters");

// Model parameters are constructor arguments, not call arguments. Same names
// across providers where the concept exists.
//
// maxTokens caps TOTAL output. On reasoning models, hidden thinking tokens are
// spent from that same budget — set it too low and you get an empty .text
// because the model used every token thinking. 200 leaves room for both.
const creative = await getModel({ temperature: 1.0, maxTokens: 200 });
const precise = await getModel({ temperature: 0, maxTokens: 200 });

const prompt = "Name a color.";
console.log("temperature 1.0 ->", (await creative.invoke(prompt)).text.trim());
console.log("temperature 0.0 ->", (await precise.invoke(prompt)).text.trim());

section("Batching");

// `.batch()` runs requests concurrently. Far faster than awaiting in a loop,
// and it respects the provider's rate limits.
const answers = await model.batch([
  "Capital of France? One word.",
  "Capital of Japan? One word.",
  "Capital of Peru? One word.",
]);
for (const a of answers) console.log(" -", a.text.trim());

section("Takeaway");
console.log(`
  model.invoke(input)  -> AIMessage        (one shot)
  model.batch([...])   -> AIMessage[]      (concurrent)
  model.stream(input)  -> AIMessageChunk*  (lesson 03)

  Read .text for a string. The same three methods exist on EVERY provider,
  which is why the rest of this course never mentions your provider again.

  Next: npm run 02
`);
