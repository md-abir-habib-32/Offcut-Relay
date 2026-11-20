"use strict";

const {readFileSync} = require("node:fs");
const {makeIndex,evidenceRecords,retrieve} = require("../dist/audit/search");
const query = readFileSync(0,"utf8");
if (query.length > 160) process.exit(2);
const index = makeIndex(evidenceRecords(require("../dist/relay")));
process.stdout.write(JSON.stringify(retrieve(query,index).map(({id,title,text,href}) => ({
  id,title,text,source:href.replace("../","/"),
}))));
