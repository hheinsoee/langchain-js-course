/**
 * LESSON 05 — Tools, and the loop that makes agents work
 *
 * CONCEPT: a model cannot do anything. It emits text. "Tool calling" means the
 * model emits a structured request — `{name, args, id}` — and YOUR code decides
 * whether and how to execute it. The model never touches your database.
 *
 * This lesson runs the tool loop BY HAND. Lesson 06 replaces all of it with
 * one `createAgent()` call. Doing it manually once is the point: an agent is
 * a while-loop, not magic.
 *
 * Run: npm run 05
 */

import * as z from "zod";
import { tool, HumanMessage, SystemMessage } from "langchain";
import type { BaseMessage } from "@langchain/core/messages";
import type { StructuredToolInterface } from "@langchain/core/tools";
import { section, getModel } from "./model.js";

section("05 · Defining tools");

// A tool is a function + a NAME + a DESCRIPTION + a SCHEMA.
// The name, description and schema are the only things the model ever sees.
// They are prompt engineering surface: vague description = unused tool.
const getWeather = tool(
  ({ city }) => {
    const fake: Record<string, string> = {
      tokyo: "18°C, light rain",
      reykjavik: "2°C, snow",
      cairo: "33°C, clear",
    };
    return fake[city.toLowerCase()] ?? `No weather station for ${city}.`;
  },
  {
    name: "get_weather",
    description: "Get the current weather for a city. Use for any weather question.",
    schema: z.object({
      city: z.string().describe("City name, e.g. 'Tokyo'"),
    }),
  }
);

const calculate = tool(
  ({ expression }) => {
    // A real tool would use a safe expression parser. Never eval model output.
    if (!/^[\d\s+\-*/().]+$/.test(expression)) return "Rejected: unsafe expression.";
    return String(Function(`"use strict";return (${expression})`)());
  },
  {
    name: "calculate",
    description: "Evaluate an arithmetic expression. Use for ALL arithmetic.",
    schema: z.object({
      expression: z.string().describe("Arithmetic only, e.g. '(18 - 2) * 3'"),
    }),
  }
);

// Tools are directly callable — useful for unit testing them without a model.
console.log("direct call:", await getWeather.invoke({ city: "Tokyo" }));

section("Binding tools: the model now has options");

const model = await getModel({ temperature: 0 });
const tools = [getWeather, calculate];
const modelWithTools = model.bindTools(tools);

const first = await modelWithTools.invoke("What's the weather in Reykjavik?");

// The model did NOT answer. It asked you to run something.
console.log(".text:", JSON.stringify(first.text));
console.log(".tool_calls:", JSON.stringify(first.tool_calls, null, 2));

section("The agent loop, written out");

// Index tools by name so we can dispatch on whatever the model asked for.
// `StructuredToolInterface` is the common supertype — without it the array of
// two differently-shaped tools infers as a union whose `.invoke` is uncallable.
const toolsByName: Record<string, StructuredToolInterface> = Object.fromEntries(
  tools.map((t) => [t.name, t as StructuredToolInterface])
);

async function runLoop(question: string, maxTurns = 6): Promise<string> {
  // A system message that INSISTS on tool use. Without it, smaller models
  // happily reply "tell me the temperatures and I'll compare them" instead of
  // calling the tool you just handed them. Tool descriptions alone are a hint;
  // the system prompt is where you make it policy.
  const messages: BaseMessage[] = [
    new SystemMessage(
      "You have tools. Never ask the user for information a tool can supply, " +
        "and never do arithmetic yourself — call calculate."
    ),
    new HumanMessage(question),
  ];

  for (let turn = 0; turn < maxTurns; turn++) {
    const ai = await modelWithTools.invoke(messages);
    messages.push(ai);

    // No tool calls -> the model is done talking. Exit condition.
    if (!ai.tool_calls?.length) return ai.text;

    // Execute every requested call. Passing the whole tool_call object to
    // `.invoke()` returns a ToolMessage already carrying the matching
    // tool_call_id — that id is how the model correlates result to request.
    // Every tool call MUST get a ToolMessage back or the next request errors.
    for (const call of ai.tool_calls) {
      console.log(`  turn ${turn}: ${call.name}(${JSON.stringify(call.args)})`);
      const result = await toolsByName[call.name].invoke(call);
      console.log(`           -> ${result.content}`);
      messages.push(result);
    }
  }
  return "Gave up: hit the turn limit.";
}

console.log("\nQ: Is it colder in Reykjavik than Tokyo, and by how much?");
console.log("A:", await runLoop("Is it colder in Reykjavik than Tokyo, and by how much?"));

section("Parallel tool calls");

// Most providers batch independent calls into ONE response. Note both
// get_weather calls arrive together below — that's one round trip, not two.
const parallel = await modelWithTools.invoke("Weather in Cairo and Tokyo?");
console.log(parallel.tool_calls?.map((c) => `${c.name}(${JSON.stringify(c.args)})`));

section("Takeaway");
console.log(`
  tool(fn, { name, description, schema })   // schema = zod
  model.bindTools([...])                    // model may now request calls
  response.tool_calls                       // [{ name, args, id }]
  await someTool.invoke(toolCall)           // -> ToolMessage w/ tool_call_id

  THE LOOP:  invoke -> tool_calls? -> execute -> append ToolMessages -> repeat
             until the model returns text instead of calls.

  You just wrote an agent. Lesson 06 is this loop, hardened, in one line.

  Next: npm run 06
`);
