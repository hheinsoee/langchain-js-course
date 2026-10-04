/**
 * LESSON 02 — Messages, roles, and prompts
 *
 * CONCEPT: models are stateless. There is no "conversation" on the server.
 * A conversation is just an array you resend in full on every single turn.
 * Once that clicks, "memory" (lesson 07) stops being mysterious.
 *
 * Run: npm run 02
 */

import { HumanMessage, SystemMessage } from "langchain";
import type { BaseMessage } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { section, getModel } from "./model.js";

const model = await getModel({ temperature: 0 });

section("02 · The four message types");
console.log(`
  SystemMessage  - instructions / persona. Not user-visible. Goes first.
  HumanMessage   - what the user said.
  AIMessage      - what the model said. May carry .tool_calls (lesson 05).
  ToolMessage    - the result of running a tool. Must carry tool_call_id.
`);

section("Class form");

const messages = [
  new SystemMessage("You are a terse Victorian butler. Never exceed 15 words."),
  new HumanMessage("I'm hungry."),
];
const reply = await model.invoke(messages);
console.log("butler:", reply.text);

section("Plain-object form (identical behaviour)");

// LangChain accepts the OpenAI-style { role, content } objects everywhere a
// message is expected. Roles: "system" | "user" | "assistant" | "tool".
// Use whichever you prefer; the class form gives you better autocomplete.
const reply2 = await model.invoke([
  { role: "system", content: "You are a terse pirate. Never exceed 15 words." },
  { role: "user", content: "I'm hungry." },
]);
console.log("pirate:", reply2.text);

section("A multi-turn conversation is just a growing array");

// THIS is the whole trick. We append the model's own reply back into the array,
// then append the next user turn. Nothing is stored anywhere else.
// Type it as BaseMessage[]: the array holds system, human AND ai messages.
const history: BaseMessage[] = [
  new SystemMessage("Answer in one short sentence."),
  new HumanMessage("My favourite number is 7."),
];

const turn1 = await model.invoke(history);
history.push(turn1);                                  // <- AIMessage back in
history.push(new HumanMessage("What is my favourite number, doubled?"));

const turn2 = await model.invoke(history);
console.log("Q1:", history[1].text);
console.log("A1:", turn1.text);
console.log("Q2:", history[3].text);
console.log("A2:", turn2.text);

console.log(`\n  history is now ${history.length} messages, resent in full every turn.`);

section("Drop the array, and the model forgets instantly");

const amnesiac = await model.invoke([
  new HumanMessage("What is my favourite number, doubled?"),
]);
console.log("no history ->", amnesiac.text);

section("Prompt templates: parameterised, reusable prompts");

// Hardcoding user input into a prompt string is the LLM equivalent of SQL
// string concatenation. Templates keep the variable slots explicit.
const template = ChatPromptTemplate.fromMessages([
  ["system", "You translate English to {language}. Output only the translation."],
  ["human", "{text}"],
]);

// `.formatMessages()` shows you exactly what gets sent.
const rendered = await template.formatMessages({
  language: "Japanese",
  text: "Where is the train station?",
});
console.log("rendered:", rendered.map((m) => `${m.getType()}: ${m.text}`));

// `.pipe()` chains a template into a model. This is the Runnable interface:
// anything with .invoke() can be piped into anything else with .invoke().
const translator = template.pipe(model);
console.log("ja ->", (await translator.invoke({ language: "Japanese", text: "Where is the train station?" })).text);
console.log("de ->", (await translator.invoke({ language: "German", text: "Where is the train station?" })).text);

section("Takeaway");
console.log(`
  - Conversations are arrays you resend. The model holds no state.
  - Append the AIMessage you got back before asking the next question.
  - ChatPromptTemplate keeps variables out of your prompt strings,
    and .pipe() composes template -> model into one callable.

  Next: npm run 03
`);
