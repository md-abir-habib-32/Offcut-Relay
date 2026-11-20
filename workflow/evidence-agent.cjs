"use strict";

const { execFile } = require("node:child_process");
const { promisify } = require("node:util");
const { join } = require("node:path");
const { Annotation, END, START, StateGraph } = require("@langchain/langgraph");
const { makeIndex, evidenceRecords, retrieve } = require("../dist/audit/search.js");
const execute = promisify(execFile);
const root = join(__dirname,"..");
const records = makeIndex(evidenceRecords(require("../dist/relay.js")));

const State = Annotation.Root({
  question: Annotation(),
  endpoint: Annotation(),
  matches: Annotation(),
  proposed: Annotation(),
  output: Annotation()
});

const evidenceWorkflow = new StateGraph(State)
  .addNode("retrieve", ({question}) => ({matches:retrieve(question,records)}))
  .addNode("abstain", () => ({output:{status:"no_evidence"}}))
  .addNode("select", async ({question,endpoint}) => {
    try {
      const {stdout} = await execute("python3",["-B","-m","workflow.local_assistant",question,"--runtime-url",endpoint],
        {cwd:root, timeout:50000, maxBuffer:65536});
      return {proposed:JSON.parse(stdout)};
    } catch {
      return {proposed:{status:"local_runtime_unavailable"}};
    }
  })
  .addNode("verify", ({matches,proposed}) => {
    const supported = matches.find(record => record.id === proposed.selected?.id);
    const selected = supported ?? matches[0];
    return {output:{status:supported?"model_selected":"search_fallback", evidenceId:selected.id,
      passage:selected.text, source:selected.href.replace("../","/")}};
  })
  .addEdge(START,"retrieve")
  .addConditionalEdges("retrieve", ({matches}) => matches.length ? "select" : "abstain", ["select","abstain"])
  .addEdge("select","verify")
  .addEdge("verify",END)
  .addEdge("abstain",END)
  .compile();

async function run(question, endpoint) {
  if (!question?.trim() || question.length > 160) throw new Error("Enter a question from 1 to 160 characters");
  return (await evidenceWorkflow.invoke({question,endpoint})).output;
}

if (require.main === module) {
  run(process.argv[2],process.argv[3] ?? "http://127.0.0.1:8080").then(result => {
    process.stdout.write(JSON.stringify(result,null,2)+"\n");
  }).catch(error => {
    process.stderr.write(error.message+"\n");
    process.exitCode=1;
  });
}

module.exports = {run};
