/**
 * LESSON 09 — LangGraph: when an agent loop is the wrong shape
 *
 * CONCEPT: `createAgent` gives you ONE control flow — model calls tools in a
 * loop until done. Plenty of real workflows are not that shape: fixed pipelines,
 * branching on classification, retry-with-critique, parallel fan-out, human
 * approval gates.
 *
 * LangGraph is the layer underneath createAgent: a state machine where nodes
 * are functions `(state) => partialStateUpdate` and edges decide what runs next.
 *
 * Built here: a support-ticket pipeline that CLASSIFIES, then ROUTES to one of
 * three handlers, then REVIEWS and loops back if the draft is poor.
 *
 *     START -> classify -> (route) -> billing ─┐
 *                                 -> technical ─┼-> review -> END
 *                                 -> refund   ─┘     └── retry ──┘
 *
 * Run: npm run 09
 */

import * as z from "zod";
import { StateGraph, StateSchema, START, END } from "@langchain/langgraph";
import type { GraphNode } from "@langchain/langgraph";
import { section, getModel } from "./model.js";

const model = await getModel({ temperature: 0 });

section("09 · Defining state");

// State is the single object every node reads from and writes to. Each key is
// a channel: a node returning { draft: "x" } overwrites only `draft`.
const TicketState = new StateSchema({
  ticket: z.string(),
  category: z.enum(["billing", "technical", "refund"]).optional(),
  draft: z.string().optional(),
  critique: z.string().optional(),
  approved: z.boolean().default(false),
  attempts: z.number().default(0),
});

type State = typeof TicketState.State;

console.log("channels:", "ticket, category, draft, critique, approved, attempts");

section("Defining nodes");

// A node is a plain async function. It receives the whole state and returns a
// PARTIAL update. Nothing LangChain-specific is required inside it — you can
// call a database, an HTTP API, or nothing at all.

const Classification = z.object({
  category: z.enum(["billing", "technical", "refund"]),
});

const classify: GraphNode<typeof TicketState> = async (state) => {
  const { category } = await model
    .withStructuredOutput(Classification)
    .invoke(`Classify this support ticket into exactly one category.\n\n${state.ticket}`);
  console.log(`  [classify] -> ${category}`);
  return { category };
};

// Three specialist handlers. In a real system these would have different tools,
// different models, different system prompts — that is the point of routing.
function handler(name: string, persona: string): GraphNode<typeof TicketState> {
  return async (state) => {
    console.log(`  [${name}] drafting (attempt ${state.attempts + 1})`);
    const retry = state.critique
      ? `\n\nYour previous draft was rejected. Fix this: ${state.critique}`
      : "";
    const res = await model.invoke([
      { role: "system", content: persona },
      { role: "user", content: `Ticket:\n${state.ticket}${retry}` },
    ]);
    return { draft: res.text, attempts: state.attempts + 1 };
  };
}

const billing = handler(
  "billing",
  "You handle billing. Reply in under 60 words. Always state the exact next step and a timeframe."
);
const technical = handler(
  "technical",
  "You handle technical issues. Reply in under 60 words. Ask for one diagnostic detail if needed."
);
const refund = handler(
  "refund",
  "You handle refunds. Reply in under 60 words. State the policy and whether this qualifies."
);

const Review = z.object({
  approved: z.boolean(),
  critique: z.string().describe("If rejected, one specific fix. If approved, empty."),
});

// The reviewer is the same model grading its own output — surprisingly
// effective, because critiquing is an easier task than generating.
const review: GraphNode<typeof TicketState> = async (state) => {
  const verdict = await model.withStructuredOutput(Review).invoke(
    `You are a QA reviewer for support replies. Reject if the reply is vague,
     over 60 words, or fails to state a concrete next step.

     TICKET: ${state.ticket}
     REPLY: ${state.draft}`
  );
  console.log(`  [review] ${verdict.approved ? "APPROVED" : `REJECTED — ${verdict.critique}`}`);
  return verdict;
};

section("Wiring the graph");

// Routing functions return the NAME of the next node. They are pure functions
// of state — no model call needed, the classification already happened.
function routeByCategory(state: State): "billing" | "technical" | "refund" {
  return state.category ?? "technical";
}

// Cycle with a bail-out. Without the attempts cap this could loop forever;
// `recursionLimit` is the backstop, but an explicit cap is better design.
function routeAfterReview(state: State): "billing" | "technical" | "refund" | typeof END {
  if (state.approved || state.attempts >= 3) return END;
  return routeByCategory(state);
}

const graph = new StateGraph(TicketState)
  .addNode("classify", classify)
  .addNode("billing", billing)
  .addNode("technical", technical)
  .addNode("refund", refund)
  .addNode("review", review)
  .addEdge(START, "classify")
  // Conditional edge: run the router, go where it says. The third argument is
  // the PATH MAP — the set of nodes this edge may reach. It is optional, but
  // without it LangGraph must assume every node is reachable, which produces a
  // wrong diagram and loses compile-time reachability checking.
  .addConditionalEdges("classify", routeByCategory, ["billing", "technical", "refund"])
  // All three handlers converge on review.
  .addEdge("billing", "review")
  .addEdge("technical", "review")
  .addEdge("refund", "review")
  // The cycle: review may send us back to a handler, or finish.
  .addConditionalEdges("review", routeAfterReview, ["billing", "technical", "refund", END])
  .compile(); // compile() is mandatory — it validates reachability

section("Running it");

const tickets = [
  "I was charged twice for my November subscription. Invoice #8821.",
  "The map export button does nothing on Firefox. Console shows a CORS error.",
  "Bought the pro plan 3 days ago by mistake, never used it. Can I get my money back?",
];

for (const ticket of tickets) {
  console.log(`\n\x1b[1mTICKET:\x1b[0m ${ticket}`);
  const out = await graph.invoke({ ticket }, { recursionLimit: 25 });
  console.log(`\x1b[1mREPLY (${out.category}, ${out.attempts} attempt(s)):\x1b[0m ${out.draft?.trim()}`);
}

section("Inspecting the graph");

// Graphs are introspectable. Useful for docs and for debugging routing.
const drawable = await graph.getGraphAsync();
console.log(drawable.drawMermaid());

section("Takeaway");
console.log(`
  new StateSchema({ ... })                 // channels, zod-typed
  new StateGraph(State)
    .addNode(name, (state) => partialUpdate)
    .addEdge(from, to)                     // unconditional
    .addConditionalEdges(from, router)     // router returns next node name
    .compile()

  - Nodes return PARTIAL state. Returning { draft } leaves everything else alone.
  - Conditional edges + a cycle give you retry/critique loops.
  - Always cap cycles: an explicit attempts counter plus recursionLimit.
  - createAgent IS a compiled graph; reach for StateGraph when your control
    flow is not "model calls tools until done".

  Next: npm run 10
`);
