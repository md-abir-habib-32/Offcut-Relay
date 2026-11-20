"use strict";

const sites = {
  morning: {
    lots: [
      { id:"L01", source:"Tongi Knit", fabric:"cotton", color:"navy", kg:38, ready:9 },
      { id:"L02", source:"Bhawal Textiles", fabric:"cotton", color:"navy", kg:26, ready:10 },
      { id:"L03", source:"Konabari Mill", fabric:"denim", color:"indigo", kg:23, ready:9 },
      { id:"L04", source:"Board Bazar Knit", fabric:"jersey", color:"gray", kg:20, ready:11 },
      { id:"L05", source:"Joydebpur Stitch", fabric:"cotton", color:"white", kg:32, ready:10 }
    ],
    orders: [
      { id:"O01", shop:"Tongi Bag Studio", fabric:"cotton", color:"navy", kg:24, deadline:14, km:[2,5,12,8,10] },
      { id:"O02", shop:"Bhawal Uniforms", fabric:"cotton", color:"navy", kg:35, deadline:16, km:[5,3,9,11,7] },
      { id:"O03", shop:"Gazipur Repair Hub", fabric:"denim", color:"indigo", kg:16, deadline:15, km:[8,7,4,10,6] },
      { id:"O04", shop:"College Road Tailors", fabric:"jersey", color:"gray", kg:18, deadline:13, km:[8,11,6,7,4] },
      { id:"O05", shop:"Kapasia Home Goods", fabric:"cotton", color:"white", kg:28, deadline:12, km:[13,12,16,15,23] }
    ]
  }
};

sites.rush = {
  lots: sites.morning.lots.map(lot => ({ ...lot, ready: lot.id === "L03" ? 14 : lot.ready })),
  orders: [
    ...sites.morning.orders.map(order => ({
      ...order, deadline: ({ O01:11, O03:14.5 })[order.id] ?? order.deadline
    })),
    { id:"O06", shop:"Joydebpur Repair Studio", fabric:"cotton", color:"white", kg:18, deadline:16, km:[9,8,11,5,6] }
  ]
};

function randomGenerator(seed) {
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}

const sigmoid = value => 1 / (1 + Math.exp(-value));
const rowFeatures = (km, slack, kg) => [1, km / 20, slack / 8, kg / 45];
const dot = (left, right) => left.reduce((sum, value, index) => sum + value * right[index], 0);

function generateExamples(seed, size) {
  const random = randomGenerator(seed);
  return Array.from({ length:size }, () => {
    const features = rowFeatures(1 + random() * 25, random() * 8, 8 + random() * 38);
    const label = Number(random() < sigmoid(dot(features, [1.05, -3.2, 3.8, -0.45])));
    return { features, label };
  });
}

function train(examples) {
  const weights = [0, 0, 0, 0];
  for (let epoch = 0; epoch < 420; epoch++) {
    const gradient = [0, 0, 0, 0];
    for (const { features, label } of examples) {
      const error = sigmoid(dot(features, weights)) - label;
      features.forEach((feature, index) => { gradient[index] += error * feature; });
    }
    gradient.forEach((change, index) => { weights[index] -= 0.45 * change / examples.length; });
  }
  return weights;
}

const training = generateExamples(2026, 700);
const weights = train(training);
const heldOut = generateExamples(8321, 250);
const brier = heldOut.reduce((sum, row) => sum + (sigmoid(dot(row.features, weights)) - row.label) ** 2, 0) / heldOut.length;

function candidate(lot, order, index) {
  const km = order.km[index];
  const slack = order.deadline - lot.ready - 0.5 - km / 20;
  if (lot.enabled === false || lot.fabric !== order.fabric || lot.color !== order.color || slack < 0) return null;
  return { lot, order, km, slack, score: sigmoid(dot(rowFeatures(km, slack, order.kg), weights)) };
}

function plan(lots, orders, priority) {
  const options = orders.map(order => lots.flatMap((lot, index) => {
    const match = candidate(lot, order, index);
    return match ? [{ ...match, index }] : [];
  }));
  const remaining = lots.map(lot => lot.kg);
  let best = { objective: -1, matches: [] };
  const chosen = [];

  function visit(orderIndex, objective) {
    if (orderIndex === orders.length) {
      if (objective > best.objective) best = { objective, matches:[...chosen] };
      return;
    }
    for (const match of options[orderIndex]) {
      const kg = match.order.kg;
      if (remaining[match.index] < kg) continue;
      remaining[match.index] -= kg;
      chosen.push(match);
      visit(orderIndex + 1, objective + kg * (1 - priority + priority * match.score));
      chosen.pop();
      remaining[match.index] += kg;
    }
    visit(orderIndex + 1, objective);
  }
  visit(0, 0);
  const baseline = [];
  for (const choices of options) {
    const next = [...choices].sort((left, right) => left.km - right.km)
      .find(match => remaining[match.index] >= match.order.kg);
    if (!next) continue;
    remaining[next.index] -= next.order.kg;
    baseline.push(next);
  }
  return { chosen:best.matches, baseline };
}

function totals(matches) {
  return matches.reduce(({ kg, expected }, match) => ({
    kg: kg + match.order.kg, expected: expected + match.order.kg * match.score
  }), { kg:0, expected:0 });
}

function validateScenario(scenario) {
  const records = scenario && [scenario.lots, scenario.orders];
  if (!records || !records.every(list => Array.isArray(list) && list.length >= 1 && list.length <= 6)) {
    throw new Error("Invalid lot or order count");
  }

  const label = text => typeof text === "string" && /^[\p{L}\p{N}][\p{L}\p{M}\p{N} ._&()-]{0,49}$/u.test(text);
  const inRange = (value, minimum, maximum) => Number.isFinite(value) && value >= minimum && value <= maximum;
  const fabrics = ["cotton","denim","jersey"];
  const colors = ["navy","indigo","gray","white"];
  const [lots, orders] = records;
  for (const list of records) {
    if (new Set(list.map(row => row?.id)).size !== list.length) throw new Error("Duplicate ID");
    if (!list.every(row => row && label(row.id) && fabrics.includes(row.fabric) && colors.includes(row.color) &&
        Number.isInteger(row.kg) && inRange(row.kg, 1, 120))) throw new Error("Invalid material record");
  }
  if (!lots.every(lot => label(lot.source) && inRange(lot.ready, 0, 24))) throw new Error("Invalid lot details");
  if (!orders.every(order => label(order.shop) && inRange(order.deadline, 0, 24) &&
      Array.isArray(order.km) && order.km.length === lots.length && order.km.every(km => inRange(km, 0, 25)))) {
    throw new Error("Invalid order details");
  }
  return { lots, orders };
}

if (typeof module !== "undefined") module.exports = { plan, sites, totals, brier, validateScenario };

if (typeof document !== "undefined" && document.getElementById("scenario")) {
  const dictionary = {
    en: {
      demo:"Simulation workspace", kicker:"FABRIC SURPLUS MATCHING", headline:"Give surplus fabric a second shift.",
      lead:"Match factory leftovers to small workshops with compatible orders, enough material, and time to collect it.",
      scenario:"Demo scenario", priority:"Pickup reliability weight", volume:"Reuse more", reliable:"Favor reliable pickups",
      explanation:"Synthetic data. Change quantities or disable a lot to see how the plan adapts.",
      lots:"Available lots", lotsDesc:"Edit the stock available for pickup.", matches:"Recommended handoffs",
      matchesDesc:"Open a match to check the reasoning.", export:"Export CSV", planned:"Fabric allocated", baseline:"Nearest-first allocation",
      expected:"Expected pickup volume", method:"How the plan is calculated", footer:"Prototype for simulated handoffs. Coordinate any real pickup directly with the workshop.",
      fit:"Fabric and color match", pickup:"Simulated pickup likelihood", distance:"Distance", slack:"Time buffer",
      unmatched:"No feasible allocation in this plan", size:"Stock in kilograms", enable:"Enable lot", active:"active",
      kg:"kg", hours:"h", km:"km", cotton:"Cotton", denim:"Denim", jersey:"Jersey", navy:"Navy", indigo:"Indigo", gray:"Gray", white:"White",
      morning:"Gazipur morning shift", rush:"Tight pickup windows", custom:"Imported scenario",
      importLabel:"Import a scenario", template:"Download JSON template", formats:"CSV, XML and HTML templates", docs:"Technical brief", audit:"Search decision evidence",
      saveLocal:"Save plan inputs to this device", loadLocal:"Restore saved inputs", clearLocal:"Remove saved inputs", saved:"Saved on this device.", restored:"Saved inputs restored.", removed:"Saved inputs removed.", unavailable:"Could not access the saved inputs on this device.", missing:"No saved inputs on this device.", localKg:"Kilograms",
      invalid:"Could not load the file. Compare its fields and limits with the template.", imported:"Scenario loaded.",
      methodCopy:"A logistic regression model predicts pickup likelihood from distance, time buffer, and order size. An exact search allocates stock across this small order list and requires matching fabric and color, enough stock, and an on-time pickup. The nearest-first baseline uses the same constraints. Training and test data are simulated, so the score needs validation with real outcomes."
    },
    bn: {
      demo:"সিমুলেশন", kicker:"উদ্বৃত্ত কাপড়ের বরাদ্দ", headline:"উদ্বৃত্ত কাপড় আবার কাজে লাগান।",
      lead:"কারখানার বাড়তি কাপড় এমন ছোট ওয়ার্কশপে বরাদ্দ করুন, যাদের চাহিদা, পরিমাণ ও সংগ্রহের সময় মেলে।",
      scenario:"ডেমোর পরিস্থিতি", priority:"সময়মতো সংগ্রহের গুরুত্ব", volume:"বেশি কাপড় পুনর্ব্যবহার", reliable:"নির্ভরযোগ্য সংগ্রহ",
      explanation:"কৃত্রিম তথ্য। পরিমাণ বদলে বা কোনো লট বন্ধ করে পরিকল্পনার পরিবর্তন দেখুন।",
      lots:"উপলব্ধ লট", lotsDesc:"সংগ্রহযোগ্য কাপড়ের পরিমাণ বদলান।", matches:"প্রস্তাবিত বরাদ্দ",
      matchesDesc:"কারণ দেখতে একটি বরাদ্দ খুলুন।", export:"সিএসভি ডাউনলোড", planned:"বরাদ্দ কাপড়", baseline:"কাছের লট আগে",
      expected:"সংগ্রহের সম্ভাব্য পরিমাণ", method:"পরিকল্পনা যেভাবে তৈরি হয়", footer:"এটি সিমুলেশন। বাস্তব সংগ্রহের জন্য সংশ্লিষ্ট ওয়ার্কশপের সঙ্গে যোগাযোগ করুন।",
      fit:"কাপড় ও রং মিলেছে", pickup:"সংগ্রহের আনুমানিক সম্ভাবনা", distance:"দূরত্ব", slack:"হাতে থাকা সময়",
      unmatched:"এই পরিকল্পনায় উপযুক্ত বরাদ্দ নেই", size:"কেজিতে মজুত", enable:"লট চালু রাখুন", active:"সক্রিয়",
      kg:"কেজি", hours:"ঘণ্টা", km:"কিমি", cotton:"সুতি", denim:"ডেনিম", jersey:"জার্সি", navy:"নেভি", indigo:"ইন্ডিগো", gray:"ধূসর", white:"সাদা",
      morning:"গাজীপুর সকালের শিফট", rush:"সংগ্রহের সময় কম", custom:"আপনার পরিস্থিতি",
      importLabel:"নিজের তথ্য দিন", template:"JSON টেমপ্লেট", formats:"CSV, XML ও HTML টেমপ্লেট", docs:"কারিগরি বিবরণ", audit:"বরাদ্দের কারণ খুঁজুন",
      saveLocal:"এই ডিভাইসে তথ্য সংরক্ষণ করুন", loadLocal:"সংরক্ষিত তথ্য ফেরত আনুন", clearLocal:"সংরক্ষিত তথ্য মুছুন", saved:"এই ডিভাইসে সংরক্ষিত হয়েছে।", restored:"সংরক্ষিত তথ্য ফেরত এসেছে।", removed:"সংরক্ষিত তথ্য মুছে ফেলা হয়েছে।", unavailable:"ডিভাইসে সংরক্ষিত তথ্য খোলা যায়নি।", missing:"সংরক্ষিত তথ্য নেই।", localKg:"কেজি",
      invalid:"ফাইলটি খোলা যায়নি। টেমপ্লেটের সঙ্গে তথ্য মিলিয়ে দেখুন।", imported:"তথ্য লোড হয়েছে।",
      methodCopy:"লজিস্টিক রিগ্রেশন দূরত্ব, হাতে থাকা সময় ও অর্ডারের পরিমাণ থেকে সংগ্রহের সম্ভাবনা অনুমান করে। কাপড়, রং, মজুত ও সময় মিলে গেলে পরিকল্পনা বরাদ্দ বিবেচনা করে। তুলনায় একই শর্তে কাছের লট আগে বেছে নেওয়া হয়। মডেলের তথ্য কৃত্রিম। বাস্তব তথ্য দিয়ে যাচাই বাকি।"
    }
  };

  const byId = id => document.getElementById(id);
  const element = (tag, text, className) => {
    const item = document.createElement(tag);
    if (text) item.textContent = text;
    if (className) item.className = className;
    return item;
  };
  let locale = "en";
  let lots = [];
  let orders = [];
  let matches = [];
  let importStatus = "";
  let storageStatus = "";
  const priority = byId("priority");
  const selector = byId("scenario");
  const strings = () => dictionary[locale];
  const number = value => new Intl.NumberFormat(locale, { maximumFractionDigits:1 }).format(value);

  function results() {
    const t = strings();
    const result = plan(lots, orders, Number(priority.value) / 100);
    matches = result.chosen;
    const summary = totals(matches);
    byId("priority-label").textContent = `${t.priority} · ${number(Number(priority.value))}%`;
    byId("lot-count").textContent = `${number(lots.filter(lot => lot.enabled && lot.kg).length)} ${t.active}`;
    const comparison = [
      [t.planned, summary.kg], [t.baseline, totals(result.baseline).kg], [t.expected, summary.expected]
    ];
    const scale = Math.max(1, ...comparison.map(([, value]) => value));
    byId("stats").replaceChildren(...comparison.map(([label,value]) => {
      const card = element("div", "", "stat");
      const meter = element("meter", "");
      meter.min = 0;
      meter.max = scale;
      meter.value = value;
      meter.setAttribute("aria-label", `${label} (${t.localKg})`);
      card.append(element("strong", `${number(value)} ${t.kg}`), element("span", label), meter);
      return card;
    }));

    const expanded = new Set(Array.from(byId("matches").querySelectorAll("details[open]"), item => item.dataset.order));
    byId("matches").replaceChildren(...orders.map(order => {
      const match = matches.find(item => item.order.id === order.id);
      if (!match) {
        const card = element("div", "", "match");
        card.append(element("strong", `${order.shop} · ${number(order.kg)} ${t.kg}`), element("p", t.unmatched, "meta exception"));
        return card;
      }
      const card = element("details", "", "match");
      card.dataset.order = order.id;
      card.open = expanded.has(order.id);
      const title = element("summary");
      title.append(element("strong", `${order.shop} · ${number(order.kg)} ${t.kg}`));
      card.append(title,
        element("p", `${match.lot.source} · ${t.fit} · ${t[order.fabric]}, ${t[order.color]}`, "status"),
        element("p", `${t.pickup} ${number(Math.round(match.score * 100))}%  ·  ${t.distance} ${number(match.km)} ${t.km}  ·  ${t.slack} ${number(match.slack)} ${t.hours}`, "meta"));
      return card;
    }));
    byId("export").disabled = !matches.length;
  }

  function stock() {
    const t = strings();
    byId("lots").replaceChildren(...lots.map(lot => {
      const row = element("div", "", "lot");
      const name = element("div");
      name.append(element("strong", lot.source), element("span", `${lot.id} · ${t[lot.fabric]} / ${t[lot.color]}`, "meta"));
      const qtyLabel = element("label");
      const qty = element("input");
      qty.type = "number"; qty.min = 0; qty.max = 120; qty.step = 1; qty.value = lot.kg;
      qty.setAttribute("aria-label", `${t.size} · ${lot.id}`);
      qty.addEventListener("input", () => {
        if (qty.value === "" || !qty.validity.valid) return;
        lot.kg = Number(qty.value);
        results();
      });
      qty.addEventListener("change", () => {
        lot.kg = Math.min(120, Math.max(0, Math.round(Number(qty.value) || 0)));
        qty.value = lot.kg;
        results();
      });
      qtyLabel.append(qty, document.createTextNode(t.kg));
      const enable = element("input");
      enable.type = "checkbox"; enable.checked = lot.enabled;
      enable.setAttribute("aria-label", `${t.enable} · ${lot.id}`);
      enable.addEventListener("change", () => { lot.enabled = enable.checked; results(); });
      row.append(name, qtyLabel, enable);
      return row;
    }));
  }

  function reset() {
    const selected = sites[selector.value];
    lots = selected.lots.map(lot => ({ ...lot, enabled:true }));
    orders = selected.orders;
    byId("matches").replaceChildren();
    stock();
    results();
  }

  function translate() {
    const t = strings();
    document.documentElement.lang = locale;
    byId("lang").textContent = locale === "en" ? "বাংলা" : "English";
    byId("lang").lang = locale === "en" ? "bn" : "en";
    for (const [id, key] of Object.entries({
      "demo-label":"demo", kicker:"kicker", headline:"headline", lead:"lead", "scenario-label":"scenario",
      volume:"volume", reliable:"reliable", explanation:"explanation", "lots-title":"lots", "lots-desc":"lotsDesc",
      "matches-title":"matches", "matches-desc":"matchesDesc", export:"export", "method-title":"method",
      "method-copy":"methodCopy", footer:"footer", "docs-link":"docs", "import-label":"importLabel", "sample-link":"template", "audit-link":"audit", "formats-link":"formats", "save-local":"saveLocal", "load-local":"loadLocal", "clear-local":"clearLocal"
    })) byId(id).textContent = t[key];
    Array.from(selector.options).forEach(option => { option.textContent = t[option.value]; });
    byId("import-status").textContent = t[importStatus] ?? "";
    byId("storage-status").textContent = t[storageStatus] ?? "";
    stock();
    results();
  }

  byId("lang").addEventListener("click", () => { locale = locale === "en" ? "bn" : "en"; translate(); });
  selector.addEventListener("change", () => {
    importStatus = "";
    byId("import-status").textContent = "";
    reset();
  });
  function setCustom(scenario) {
    sites.custom = scenario;
    if (!Array.from(selector.options).some(option => option.value === "custom")) {
      const option = element("option", strings().custom);
      option.value = "custom";
      selector.append(option);
    }
    selector.value = "custom";
    reset();
  }

  function keepInputs() {
    return {
      lots: lots.map(lot => ({ ...lot, kg:Math.max(1,lot.kg), editedKg:lot.kg })),
      orders: structuredClone(orders),
      priority: Number(priority.value)
    };
  }

  for (const action of ["save","load","clear"]) {
    byId(action + "-local").addEventListener("click", async () => {
      const buttons = ["save","load","clear"].map(item => byId(item + "-local"));
      for (const button of buttons) button.disabled = true;
      try {
        const state = await savedScenario(action, action === "save" ? keepInputs() : undefined);
        storageStatus = { save:"saved", load:"missing", clear:"removed" }[action];
        if (action === "load" && state) {
          const restored = validateScenario(state);
          const enabled = restored.lots.map(lot => lot.enabled !== false);
          const quantities = restored.lots.map(lot => Number.isInteger(lot.editedKg) && lot.editedKg === 0 ? 0 : lot.kg);
          priority.value = Number.isFinite(state.priority) ? state.priority : 20;
          setCustom(restored);
          lots.forEach((lot,index) => { lot.enabled = enabled[index]; lot.kg = quantities[index]; });
          stock();
          results();
          storageStatus = "restored";
        }
      } catch {
        storageStatus = "unavailable";
      } finally {
        for (const button of buttons) button.disabled = false;
        byId("storage-status").textContent = strings()[storageStatus];
      }
    });
  }

  byId("scenario-file").addEventListener("change", async event => {
    const input = event.currentTarget;
    const [file] = input.files;
    if (!file) return;
    input.disabled = true;
    try {
      if (file.size > 32000) throw new Error("File exceeds 32 KB");
      setCustom(validateScenario(parseScenarioFile(file.name, await file.text())));
      importStatus = "imported";
    } catch {
      importStatus = "invalid";
    } finally {
      input.value = "";
      input.disabled = false;
      byId("import-status").textContent = strings()[importStatus];
    }
  });
  priority.addEventListener("input", results);
  byId("export").addEventListener("click", () => {
    const rows = [["Order","Workshop","Lot","Source","Fabric","Color","Kg","Simulated pickup likelihood"]];
    for (const { order,lot,score } of matches) {
      rows.push([order.id,order.shop,lot.id,lot.source,order.fabric,order.color,order.kg,score.toFixed(3)]);
    }
    const csv = rows.map(row => row.map(cell => '"' + String(cell).replaceAll('"','""') + '"').join(",")).join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type:"text/csv;charset=utf-8" }));
    const link = element("a");
    link.href = url; link.download = "Offcut-Relay-Plan.csv";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  reset();
}
