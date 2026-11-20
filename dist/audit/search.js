"use strict";

const concepts = [
  ["deadline", "window", "windows", "late", "slack", "timing", "time", "সময়", "সময়সীমা", "দেরি"],
  ["closest", "nearest", "baseline", "greedy", "প্রথমে", "নিকটতম", "কাছের", "কাছে"],
  ["allocated", "allocation", "capacity", "stock", "quantity", "weight", "unfilled", "miss", "মজুত", "ওজন", "পরিমাণ", "বরাদ্দ"],
  ["reliability", "chance", "forecast", "probability", "expected", "score", "সম্ভাবনা", "পূর্বাভাস", "নির্ভরযোগ্যতা"],
  ["field", "pilot", "test", "tested", "validation", "real", "synthetic", "simulated", "যাচাই", "সিমুলেশন", "বাস্তব"],
  ["privacy", "upload", "file", "import", "json", "সুরক্ষা", "ফাইল", "আপলোড"],
  ["fabric", "material", "colour", "color", "shade", "কাপড়", "রং", "উপাদান"]
];

const stopWords = new Set(["a","an","the","was","were","is","why","did","do","how","for","and","what","in","of","an","i","can"]);

function tokenize(text) {
  return (text.normalize("NFC").toLowerCase().match(/[\p{L}\p{M}\p{N}]+/gu) ?? [])
    .filter(word => !stopWords.has(word));
}

const rewrites = new Map([
  ["cutoff", ["deadline","window"]],
  ["leftovers", ["fabric","stock"]],
  ["confidential", ["privacy","local"]],
  ["সময়সীমা", ["deadline","window"]]
]);

function encode(tokens) {
  const set = new Set(tokens);
  return concepts.map(words => Number(words.some(word => set.has(word))));
}

function cosine(left, right) {
  const score = left.reduce((sum, value, index) => sum + value * right[index], 0);
  return score / (Math.hypot(...left) * Math.hypot(...right) || 1);
}

function evidenceRecords({ plan, totals, sites, brier }) {
  const regular = plan(sites.morning.lots, sites.morning.orders, 0.2);
  const reuse = totals(plan(sites.rush.lots, sites.rush.orders, 0).chosen);
  const reliable = totals(plan(sites.rush.lots, sites.rush.orders, 1).chosen);
  const uniform = regular.chosen.find(match => match.order.id === "O02");
  const lateLots = sites.rush.lots.filter(lot => {
    const before = sites.morning.lots.find(normal => normal.id === lot.id);
    return before.ready < lot.ready && regular.chosen.some(match => match.lot.id === lot.id);
  });
  const lateNames = lateLots.map(lot => lot.source);
  const lateOrders = regular.chosen.filter(match => lateLots.some(lot => lot.id === match.lot.id))
    .map(match => match.order.shop);

  return [
    { id:"nearest", title:"Why nearest-first missed the uniform order", context:"Morning shift baseline", entities:[uniform.lot.source, uniform.order.shop],
      tags:"closest greedy unfilled stock order কাছের বরাদ্দ",
      text:`The 24 kg bag order fits two navy cotton lots. Nearest-first uses the larger Tongi lot for the bag order and leaves no lot with enough remaining stock for the 35 kg uniform order. The batch planner reserves ${uniform.lot.source} for uniforms. Total allocated rises from ${totals(regular.baseline).kg} kg to ${totals(regular.chosen).kg} kg in this fixture.`, href:"../docs/#evidence" },
    { id:"rush", title:"Tight windows change what can be collected", context:"Pickup deadline and reliability", entities:[...lateNames, ...lateOrders],
      tags:"deadline slack late tight pickup windows সংগ্রহের সময় কম",
      text:`In the tight-window fixture the denim lot is ready too late. Reuse priority allocates ${reuse.kg} kg with ${reuse.expected.toFixed(1)} kg expected. Reliability priority allocates ${reliable.kg} kg with ${reliable.expected.toFixed(1)} kg expected. The forecast-weighted figure is a model estimate.`, href:"../docs/#evidence" },
    { id:"eligibility", title:"Material and stock are hard constraints", context:"Batch allocation rules",
      tags:"fabric color material quantity stock incompatible constraint কাপড় রং মজুত",
      text:"An eligible assignment must have the exact fabric and color, sufficient kilograms remaining in a single lot, and enough time to load and travel. The estimator only scores pairings that pass those checks.", href:"../docs/#architecture" },
    { id:"validation", title:"Pickup model validation and limitations", context:"Forecast evidence",
      tags:"forecast tested in field pilot real reliability training synthetic model বাস্তব যাচাই",
      text:`Logistic regression uses distance, order weight, and available pickup time. It trained on 700 simulated observations and achieved a Brier score of ${brier.toFixed(3)} on 250 independently seeded synthetic observations. There are no real pilot outcomes yet.`, href:"../docs/#pilot" },
    { id:"privacy", title:"What happens to an imported scenario", context:"Local data handling",
      tags:"privacy file upload import json locally private personal ফাইল আপলোড সুরক্ষা",
      text:"The app parses and validates JSON, CSV, XML or HTML inside the browser. Unsaved data is cleared on reload. The device storage option creates a local copy only when requested. CSV export is generated locally.", href:"../docs/#data" },
    { id:"pilot", title:"What to measure in a pilot", context:"Field test plan",
      tags:"pilot field real measure completion impact waste reuse বাস্তব যাচাই",
      text:"Record allocated kilograms, completed pickups, kilograms actually reused, and time spent coordinating. Compare against the same nearest-first eligibility rules, then assess the model on suppliers held out of retraining.", href:"../docs/#pilot" }
  ];
}

function makeIndex(records) {
  const graph = new Map();
  for (const record of records) {
    for (const label of record.entities ?? []) {
      const entity = tokenize(label).join(" ");
      if (!graph.has(entity)) graph.set(entity, new Set());
      graph.get(entity).add(record.id);
    }
  }
  const items = records.map(record => {
    const title = new Set(tokenize(record.title));
    const all = tokenize([record.title, record.context, record.tags, record.text].join(" "));
    return { record, title, tokens:new Set(all), vector:encode(all) };
  });
  return { items, graph };
}

function retrieve(query, index) {
  const literal = tokenize(query);
  const phrase = ` ${literal.join(" ")} `;
  const terms = [...new Set(literal.flatMap(term => [term, ...(rewrites.get(term) ?? [])]))];
  const linkedRecords = new Set();
  for (const [entity, records] of index.graph) {
    if (phrase.includes(` ${entity} `)) {
      for (const id of records) linkedRecords.add(id);
    }
  }
  const vector = encode(terms);
  const candidates = index.items.map(item => {
    const overlap = terms.filter(term => item.tokens.has(term)).length / Math.max(1, terms.length);
    const relationship = linkedRecords.has(item.record.id) ? 0.65 : 0;
    return { item, score:0.68 * overlap + 0.32 * cosine(vector, item.vector) + relationship };
  }).filter(candidate => candidate.score > 0.12)
    .sort((a,b) => b.score - a.score).slice(0, 5);
  return candidates.map(({ item,score }) => {
    const titleMatch = terms.filter(term => item.title.has(term)).length;
    return { record:item.record, score:score + 0.12 * titleMatch };
  }).sort((a,b) => b.score - a.score).slice(0, 3).map(candidate => candidate.record);
}

if (typeof module !== "undefined") module.exports = { evidenceRecords, makeIndex, retrieve };

if (typeof document !== "undefined") {
  const records = makeIndex(evidenceRecords({ plan, totals, sites, brier }));
  const form = document.getElementById("audit-form");
  const input = document.getElementById("audit-query");
  const status = document.getElementById("audit-status");
  const results = document.getElementById("audit-results");

  form.addEventListener("submit", event => {
    event.preventDefault();
    const matches = retrieve(input.value, records);
    status.textContent = matches.length
      ? `${matches.length} evidence ${matches.length === 1 ? "record" : "records"} found.`
      : "No relevant evidence found. Try a term such as pickup, stock, or import.";
    results.replaceChildren(...matches.map(record => {
      const card = document.createElement("article");
      card.className = "panel audit-result";
      const heading = document.createElement("h2");
      heading.textContent = record.title;
      const context = document.createElement("p");
      context.className = "meta";
      context.textContent = record.context;
      const description = document.createElement("p");
      description.textContent = record.text;
      const link = document.createElement("a");
      link.href = record.href;
      link.textContent = "Read the source";
      card.append(heading, context, description, link);
      return card;
    }));
  });

  for (const button of form.querySelectorAll('button[type="button"]')) {
    button.addEventListener("click", () => {
      input.value = button.textContent;
      form.requestSubmit();
    });
  }
}
