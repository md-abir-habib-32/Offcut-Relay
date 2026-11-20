"use strict";

const {readFileSync} = require("node:fs");
const {parseScenarioFile} = require("../dist/intake.js");
const {plan,totals,validateScenario} = require("../dist/relay.js");

function calculate(input) {
  const priority = input.priority ?? 0.2;
  if (!Number.isFinite(priority) || priority < 0 || priority > 1) throw new Error("Priority must be between zero and one");
  if (!["csv","json"].includes(input.format)) throw new Error("Unsupported workflow format");
  const {lots,orders} = validateScenario(parseScenarioFile("scenario." + input.format,input.text));
  const outcome = plan(lots,orders,priority);
  return {
    lot_count: lots.length,
    order_count: orders.length,
    priority,
    planned: totals(outcome.chosen),
    nearest_first: totals(outcome.baseline),
    handoffs: outcome.chosen.map(({lot,order,km,slack,score}) => ({
      lot_id:lot.id, order_id:order.id, kg:order.kg, km, slack, estimated_pickup:score
    }))
  };
}

try {
  process.stdout.write(JSON.stringify(calculate(JSON.parse(readFileSync(0,"utf8")))));
} catch (error) {
  process.stderr.write(error.message);
  process.exitCode = 2;
}
