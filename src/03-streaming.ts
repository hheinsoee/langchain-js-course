/**
 * LESSON 03 — Streaming
 *
 * CONCEPT: `.invoke()` blocks until the last token. For anything a human
 * watches, that's the difference between a 12-second stare at a blank screen
 * and text appearing immediately. `.stream()` yields AIMessageChunk objects.
 *
 * The key API detail: chunks are not strings, they are PARTIAL MESSAGES, and
 * they compose with `.concat()` back into a whole message.
 *
 * Run: npm run 03
 */

import type { AIMessageChunk } from "langchain";
import { section, getModel } from "./model.js";

const model = await getModel();

section("03 · Token streaming");

const stream = await model.stream(
  "Write two sentences about why streaming matters in chat UIs."
);

// `chunk.text` is the delta — only the new characters, not the running total.
for await (const chunk of stream) {
  process.stdout.write(chunk.text);
}
console.log("\n");

section("Chunks reassemble into a real message");

// `.concat()` merges chunks. The accumulated value is a full AIMessageChunk,
// so you get usage_metadata, tool_calls, and content blocks at the end —
// things a naive string concatenation would throw away.
let full: AIMessageChunk | undefined;
let chunkCount = 0;

for await (const chunk of await model.stream("Name three primary colors.")) {
  full = full ? full.concat(chunk) : chunk;
  chunkCount++;
}

console.log(`${chunkCount} chunks ->`, full?.text.trim());
console.log("accumulated usage:", full?.usage_metadata);

section("Why not just join strings?");
console.log(`
  Because a chunk's .content may be an array of blocks: text, reasoning
  (thinking tokens), tool-call fragments. .concat() merges each block type
  correctly - partial JSON of a tool call gets stitched into valid JSON.
  Joining .text would silently drop reasoning and tool calls.
`);

section("Typed block streaming");

// Branch on block type to render reasoning differently from answer text.
// Providers without reasoning simply never emit that block.
for await (const chunk of await model.stream("What is 17 * 23? Think it through.")) {
  for (const block of chunk.contentBlocks) {
    if (block.type === "text") process.stdout.write(block.text);
    else if (block.type === "reasoning") process.stdout.write(`\x1b[2m${block.reasoning}\x1b[0m`);
  }
}
console.log("\n");

section("Takeaway");
console.log(`
  for await (const chunk of await model.stream(input))   // deltas
  full = full ? full.concat(chunk) : chunk               // rebuild the message

  Note the double await: .stream() returns a Promise of an async iterable.
  Agents stream too, with richer event types - lesson 06.

  Next: npm run 04
`);
