import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const outputDir = "/Users/mimi/Documents/Code/Github/GSA-work/explore/sythetic/outputs/manual_branch_labels_20260830";
const workbookPath = `${outputDir}/CMIP6_branch_manual_labels.xlsx`;
const previewDir = `${outputDir}/previews_no_two_band`;
await fs.mkdir(previewDir, { recursive: true });

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const MATRIX_SHEETS = [
  "P_Q_TRIM", "P_Q_RAW", "ET_Q_RAW", "RSDS_Q_RAW", "RLDS_Q_RAW",
  "HFLS_Q_RAW", "HFSS_Q_RAW", "HFSS_Q_TRIM", "P_ET_RAW", "MRSOS_Q_TRIM",
  "RLUS_Q_RAW", "RLUS_Q_TRIM", "MRROS_Q_TRIM", "HFLS_Q_TRIM", "PRSN_Q_TRIM",
];
const STATUSES = ["Branch", "Candidate", "Uncertain", "No branch", "待标注"];
const COLORS = {
  "Branch": "#D9F2E3", "Candidate": "#FFF1B8", "Uncertain": "#E5E7EB",
  "No branch": "#F5F6F8", "待标注": "#FFFFFF",
};
const FONT_COLORS = {
  "Branch": "#166534", "Candidate": "#8A5A00", "Uncertain": "#4B5563",
  "No branch": "#374151", "待标注": "#9CA3AF",
};

function addStatusFormatting(range, anchor) {
  range.conditionalFormats.deleteAll();
  for (const status of STATUSES) {
    range.conditionalFormats.addCustom(`=${anchor}="${status}"`, {
      fill: COLORS[status],
      font: { bold: status !== "待标注", color: FONT_COLORS[status] },
    });
  }
}

// The only existing Two-band label becomes No branch.
const pqTrim = workbook.worksheets.getItem("P_Q_TRIM");
pqTrim.getRange("C5").values = [["No branch"]];

// Remove Two-band from every matrix validation, formatting rule and legend.
for (const name of MATRIX_SHEETS) {
  const sheet = workbook.worksheets.getItem(name);
  const grid = sheet.getRange("B5:G9");
  grid.dataValidation = { rule: { type: "list", values: STATUSES } };
  addStatusFormatting(grid, "B5");

  sheet.getRange("I4:J10").clear({ applyTo: "all" });
  sheet.getRange("I4:J9").values = [
    ["Legend", "Meaning"],
    ["Branch", "明确分支"],
    ["Candidate", "局部/较弱分支"],
    ["Uncertain", "尚需复核"],
    ["No branch", "明确非分支"],
    ["待标注", "无人工判断"],
  ];
  sheet.getRange("I4:J4").format = {
    fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" },
  };
  for (let i = 0; i < STATUSES.length; i++) {
    sheet.getRange(`I${5 + i}:J${5 + i}`).format = {
      fill: COLORS[STATUSES[i]], font: { bold: true, color: FONT_COLORS[STATUSES[i]] },
      borders: { preset: "inside", style: "thin", color: "#D7DEE7" },
    };
  }
  sheet.getRange("I4:I9").format.columnWidthPx = 90;
  sheet.getRange("J4:J9").format.columnWidthPx = 130;
}

// Update the summary legend and remove the Two-band count column.
const summary = workbook.worksheets.getItem("Summary");
summary.getRange("A4:B10").clear({ applyTo: "all" });
summary.getRange("A4:B9").values = [
  ["标签", "含义"],
  ["Branch", "明确分支"],
  ["Candidate", "局部分支或较弱证据"],
  ["Uncertain", "存在冲突或尚未最终确认"],
  ["No branch", "明确不是分支"],
  ["待标注", "尚无人工判断"],
];
summary.getRange("A4:B4").format = {
  fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" },
};
for (let i = 0; i < STATUSES.length; i++) {
  summary.getRange(`A${5 + i}:B${5 + i}`).format = {
    fill: COLORS[STATUSES[i]], font: { color: FONT_COLORS[STATUSES[i]], bold: true },
    borders: { preset: "inside", style: "thin", color: "#D7DEE7" },
  };
}
summary.getRange("A4:A9").format.columnWidthPx = 105;
summary.getRange("B4:B9").format.columnWidthPx = 260;

const summarySource = summary.getRange("A13:C27").values;
summary.getRange("A12:J27").clear({ applyTo: "all" });
summary.getRange("A12:I12").values = [[
  "Worksheet", "Pair", "Version", "Annotated", "Branch", "Candidate",
  "Uncertain", "No branch", "待标注",
]];
summary.getRange("A12:I12").format = {
  fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center",
};
summary.getRange("A13:C27").values = summarySource;
for (let i = 0; i < MATRIX_SHEETS.length; i++) {
  const row = 13 + i;
  const name = MATRIX_SHEETS[i];
  summary.getRange(`D${row}:I${row}`).formulas = [[
    `=30-COUNTIF('${name}'!B5:G9,"待标注")`,
    `=COUNTIF('${name}'!B5:G9,"Branch")`,
    `=COUNTIF('${name}'!B5:G9,"Candidate")`,
    `=COUNTIF('${name}'!B5:G9,"Uncertain")`,
    `=COUNTIF('${name}'!B5:G9,"No branch")`,
    `=COUNTIF('${name}'!B5:G9,"待标注")`,
  ]];
}
summary.getRange("A13:I27").format = {
  borders: { preset: "inside", style: "thin", color: "#E1E6EC" }, verticalAlignment: "center",
};
summary.getRange("D13:I27").format.horizontalAlignment = "center";
summary.getRange("A12:A27").format.columnWidthPx = 125;
summary.getRange("B12:B27").format.columnWidthPx = 110;
summary.getRange("C12:C27").format.columnWidthPx = 90;
summary.getRange("D12:I27").format.columnWidthPx = 82;

// Update the flat label log and its validation.
const log = workbook.worksheets.getItem("Label_Log");
const logRange = log.getRange("A5:H66");
const logRows = logRange.values;
for (const row of logRows) {
  if (row[5] === "Two-band") {
    row[5] = "No branch";
    row[7] = "旧双带类别已移除；按当前判定体系归入 No branch。";
  } else if (typeof row[7] === "string" && row[7].includes("Two-band")) {
    row[7] = "旧双带类别已移除；按当前判定体系归入 No branch。";
  }
}
logRange.values = logRows;
log.getRange("F5:F66").dataValidation = { rule: { type: "list", values: STATUSES.slice(0, 4) } };
addStatusFormatting(log.getRange("F5:F66"), "F5");

console.log((await workbook.inspect({
  kind: "table", range: "P_Q_TRIM!A4:G9", include: "values,formulas",
  tableMaxRows: 10, tableMaxCols: 8, maxChars: 5000,
})).ndjson);
console.log((await workbook.inspect({
  kind: "match", searchTerm: "Two-band",
  options: { useRegex: false, maxResults: 100 }, summary: "Two-band removal scan",
})).ndjson);
console.log((await workbook.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan",
})).ndjson);

for (const name of ["Summary", ...MATRIX_SHEETS, "Label_Log"]) {
  const range = name === "Summary" ? "A1:I27" : name === "Label_Log" ? "A1:H66" : "A1:J10";
  const preview = await workbook.render({ sheetName: name, range, scale: 1.25, format: "png" });
  await fs.writeFile(`${previewDir}/${name}.png`, new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(workbookPath);
console.log(`Updated ${workbookPath}`);
