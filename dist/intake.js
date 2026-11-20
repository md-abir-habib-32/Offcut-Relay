"use strict";

const scenarioHeaders = ["kind","id","name","fabric","color","kg","ready","deadline","km"];

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false, endedQuote = false;
  function nextField() {
    row.push(field);
    field = "";
    endedQuote = false;
  }
  function nextRow() {
    nextField();
    if (row.some(value => value.trim() !== "")) rows.push(row);
    row = [];
  }
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += ch;
        i++;
      } else if (ch === '"') {
        quoted = false;
        endedQuote = true;
      } else field += ch;
    } else if (ch === ",") nextField();
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      nextRow();
    } else if (ch === '"' && !field && !endedQuote) quoted = true;
    else if (endedQuote || ch === '"') throw new Error("Invalid CSV quoting");
    else field += ch;
  }
  if (quoted) throw new Error("Unclosed CSV quote");
  nextRow();
  return rows;
}

function rowsToScenario(rows) {
  const [headers, ...body] = rows;
  if (!headers || headers.length !== scenarioHeaders.length ||
      !scenarioHeaders.every((header, index) => headers[index].trim() === header)) throw new Error("Invalid columns");
  const lots = [], orders = [];
  const numeric = value => {
    if (!/^(?:[0-9]+(?:\.[0-9]+)?)$/.test(value.trim())) throw new Error("Invalid numeric value");
    return Number(value);
  };
  for (const row of body) {
    if (row.length !== headers.length) throw new Error("Invalid row length");
    const [kind,id,name,fabric,color,kg,ready,deadline,km] = row.map(value => value.trim());
    const shared = { id,fabric,color,kg:numeric(kg) };
    if (kind === "lot" && !deadline && !km) lots.push({ ...shared,source:name,ready:numeric(ready) });
    else if (kind === "order" && !ready) orders.push({ ...shared,shop:name,deadline:numeric(deadline),km:km.split("|").map(numeric) });
    else throw new Error("Invalid record kind or unexpected fields");
  }
  return { lots, orders };
}

function parseScenarioFile(filename, rawText) {
  const text = rawText.replace(/^\uFEFF/, "");
  const extension = filename.split(".").at(-1).toLowerCase();
  if (extension === "json") return JSON.parse(text);
  if (extension === "csv") return rowsToScenario(parseCsv(text));
  if (extension === "html" || extension === "htm") {
    const template = document.createElement("template");
    template.innerHTML = text;
    const table = template.content.querySelector("table[data-offcut-scenario]");
    if (!table) throw new Error("Missing scenario table");
    const rows = Array.from(table.rows, row => Array.from(row.cells, cell => cell.textContent));
    return rowsToScenario(rows);
  }
  if (extension === "xml") {
    if (/<!DOCTYPE|<!ENTITY/i.test(text)) throw new Error("XML declarations are unsupported");
    const xml = new DOMParser().parseFromString(text,"application/xml");
    if (xml.querySelector("parsererror") || xml.documentElement.tagName !== "scenario") throw new Error("Invalid XML");
    const rows = Array.from(xml.documentElement.children, element => {
      if (element.tagName !== "row" || Array.from(element.children).some(child => !scenarioHeaders.includes(child.tagName))) {
        throw new Error("Invalid XML element");
      }
      return scenarioHeaders.map(header => {
        const elements = element.querySelectorAll(":scope > " + header);
        if (elements.length > 1 || elements[0]?.childElementCount) throw new Error("Invalid XML field");
        return elements[0]?.textContent ?? "";
      });
    });
    return rowsToScenario([scenarioHeaders,...rows]);
  }
  throw new Error("Unsupported file type");
}

function savedScenario(action, scenario) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("offcut-relay", 1);
    request.onblocked = () => reject(new Error("Local storage blocked"));
    request.onerror = () => reject(request.error);
    request.onupgradeneeded = () => request.result.createObjectStore("scenarios");
    request.onsuccess = () => {
      const db = request.result;
      const tx = db.transaction("scenarios", action === "load" ? "readonly" : "readwrite");
      const store = tx.objectStore("scenarios");
      const result = action === "save" ? store.put(scenario,"saved") :
        action === "clear" ? store.delete("saved") : store.get("saved");
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => { db.close(); reject(tx.error); };
      tx.oncomplete = () => { db.close(); resolve(result.result); };
    };
  });
}

if (typeof module !== "undefined") module.exports = { parseScenarioFile, scenarioHeaders };
