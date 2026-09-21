import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = new URL(".", import.meta.url).pathname;

const labels = {
  A: [
    2,3,3,2,1,2,3,2,1,1, 1,1,3,1,3,3,2,3,2,1,
    3,1,1,1,3,1,1,1,3,2, 2,3,1,1,2,1,1,2,3,1,
    3,2,3,3,3,3,1,1,3,3, 3,3,1,1,3,1,1,1,3,3,
    3,1,2,1,1,1,2,1,1,1, 2,1,1,1,3,3,3,2,3,2,
    3,3,1,1,3,1,3,1,1,1, 3,2,3,2,3,1,3,1,3,1,
  ],
  B: [
    1,1,1,1,3,1,1,1,3,3, 1,1,1,3,1,1,3,3,1,1,
    1,1,1,1,3,1,1,3,1,1, 1,1,1,1,1,3,1,3,1,1,
    1,1,3,3,1,1,3,1,1,1, 1,1,1,1,1,1,1,1,1,1,
    1,1,3,1,1,1,1,3,1,1, 1,1,1,1,1,3,3,1,1,1,
    1,1,1,1,3,3,1,1,3,3, 1,1,3,1,1,1,1,1,1,1,
  ],
  C: [
    3,1,3,3,3,1,1,1,3,1, 3,1,1,3,1,1,1,1,3,1,
    3,3,3,1,1,1,1,3,1,3, 1,3,1,3,3,1,3,3,1,1,
    3,3,1,1,3,2,3,3,3,3, 3,1,1,1,1,3,3,1,3,1,
    3,1,1,3,1,1,2,1,1,3, 1,3,1,1,1,3,3,3,3,1,
    3,1,1,1,1,1,1,1,1,1, 1,1,3,3,1,1,3,3,3,1,
  ],
  E: [
    3,3,1,1,1,2,3,3,1,3, 2,2,1,1,1,3,1,3,2,1,
    3,3,1,2,1,3,2,2,1,2, 2,1,1,3,3,3,3,3,3,1,
    3,3,1,1,2,2,3,3,3,2, 1,2,3,3,3,2,2,2,1,1,
    3,1,1,1,2,1,3,1,1,3, 2,2,2,3,3,3,3,3,3,2,
    2,1,2,3,3,1,1,2,3,1, 1,3,1,3,3,1,1,1,3,3,
  ],
  F: [
    2,3,1,1,2,3,1,1,1,2, 3,2,3,2,1,3,1,1,1,1,
    1,2,2,1,1,2,1,3,1,3, 1,1,3,1,3,3,1,1,1,3,
    1,3,1,1,3,3,1,1,2,1, 3,2,1,3,1,1,1,3,1,1,
    1,3,2,3,1,2,1,1,1,1, 1,1,1,2,1,1,1,1,2,1,
    1,3,1,1,3,2,1,2,1,1, 1,2,2,1,3,3,3,1,1,3,
  ],
  G: [
    3,1,3,3,3,3,1,3,3,3, 1,2,3,3,1,1,1,3,3,1,
    2,1,1,3,2,3,2,2,1,3, 2,3,3,3,3,1,1,1,2,3,
    3,2,1,3,1,1,3,3,2,2, 3,2,2,2,3,1,1,3,1,1,
    3,3,1,1,1,2,1,3,3,3, 3,1,1,2,3,3,1,3,3,1,
    3,3,1,1,3,3,3,1,3,1, 3,1,2,3,2,1,1,3,3,3,
  ],
  I: [
    1,1,1,3,3,1,3,3,1,1, 1,3,1,1,1,1,1,3,1,1,
    3,1,1,1,1,3,1,2,1,3, 3,1,1,3,2,3,3,1,2,3,
    3,1,3,1,1,3,1,3,3,3, 1,1,1,1,1,1,3,1,3,1,
    1,1,3,3,2,3,1,3,1,3, 1,1,3,1,3,2,3,3,3,1,
    3,1,2,1,3,3,1,1,3,3, 1,3,1,3,1,3,1,3,3,1,
  ],
};

for (const [annotator, values] of Object.entries(labels)) {
  if (values.length !== 100) throw new Error(`${annotator} has ${values.length} labels, expected 100`);
  if (values.some((v) => ![1, 2, 3].includes(v))) throw new Error(`${annotator} contains an invalid label`);
}

const rows = [["id", "value"]];
for (const [annotator, values] of Object.entries(labels)) {
  values.forEach((value, index) => rows.push([`${annotator}${index + 1}`, value]));
}

const labelMap = new Map(rows.slice(1));
const requiredChecks = {
  A49: 3,
  A50: 3,
  B31: 1,
  B45: 1,
  B84: 1,
  C82: 1,
  E99: 3,
  F23: 2,
};
for (const [id, expected] of Object.entries(requiredChecks)) {
  const actual = labelMap.get(id);
  if (actual !== expected) throw new Error(`${id}: expected ${expected}, found ${actual}`);
}
console.log(JSON.stringify({ requiredChecks }));

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("labels");
sheet.showGridLines = false;
sheet.getRange(`A1:B${rows.length}`).values = rows;
sheet.freezePanes.freezeRows(1);

sheet.getRange("A1:B1").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: { preset: "outside", style: "thin", color: "#1F4E78" },
};
sheet.getRange("A2:A701").format = { horizontalAlignment: "left" };
sheet.getRange("B2:B701").format = { horizontalAlignment: "center", numberFormat: "0" };
sheet.getRange("A1:A701").format.columnWidth = 14;
sheet.getRange("B1:B701").format.columnWidth = 10;
sheet.getRange("A1:B701").format.rowHeight = 18;

const table = sheet.tables.add("A1:B701", true, "ManualLabelsTable");
table.style = "TableStyleMedium2";
table.showFilterButton = true;

const inspected = await workbook.inspect({
  kind: "table",
  range: "labels!A1:B12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 2,
});
console.log(inspected.ndjson);

const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "final formula error scan",
});
console.log(errorScan.ndjson);

const preview = await workbook.render({ sheetName: "labels", range: "A1:B25", scale: 2, format: "png" });
await fs.writeFile(`${outputDir}/manual_labels_preview.png`, new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${outputDir}/manual_global_relationship_labels.xlsx`);

console.log(JSON.stringify({ rows: rows.length - 1, output: `${outputDir}/manual_global_relationship_labels.xlsx` }));
